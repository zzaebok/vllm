# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio import (
    moriio_connector as moriio_connector_module,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    HandshakeError,
    MoRIIOAgentMetadata,
    MoRIIOConnectorMetadata,
    MoRIIOMode,
    MoRIIOTransferAck,
    RemoteAllocInfo,
    ReqMeta,
    WriteTask,
    get_port_offset,
    resolve_peer_tp_size,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnector,
    MoRIIOConnectorScheduler,
    MoRIIOConnectorWorker,
    get_moriio_expected_ack_count,
    get_moriio_remote_tp_rank,
    resolve_moriio_transfer_ack,
    validate_moriio_heterogeneous_tp_kv_heads,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
    MoRIIOWriter,
)


def test_remote_tp_rank_same_tp_maps_to_self():
    assert [get_moriio_remote_tp_rank(rank, 4, 4) for rank in range(4)] == [
        0,
        1,
        2,
        3,
    ]


@pytest.mark.parametrize(("dp_size", "tp_size"), [(2, 2), (2, 4), (4, 2)])
def test_dp_tp_port_offsets_are_injective(dp_size, tp_size):
    offsets = {
        get_port_offset(dp_rank, tp_rank, tp_size)
        for dp_rank in range(dp_size)
        for tp_rank in range(tp_size)
    }

    assert len(offsets) == dp_size * tp_size


def test_port_offset_rejects_unknown_tp_size():
    """0 is the "unknown peer TP" sentinel; it must not silently drop the DP term."""
    with pytest.raises(ValueError, match="tp_size must be positive"):
        get_port_offset(1, 0, 0)


def test_resolve_peer_tp_size_falls_back_when_unadvertised():
    assert resolve_peer_tp_size({}, 4) == 4
    assert resolve_peer_tp_size({"tp_size": 0}, 4) == 4
    assert resolve_peer_tp_size({"tp_size": 2}, 4) == 2
    assert resolve_peer_tp_size({"remote_tp_size": 8, "tp_size": 2}, 4) == 8


def test_early_write_release_uses_producer_tp_size():
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler.tp_size = 4
    sent = []
    scheduler._send_transfer_release = lambda *args: sent.append(args)

    scheduler._release_write_prefill_blocks(
        "req",
        {
            "transfer_id": "tx",
            "remote_dp_rank": 1,
            "remote_host": "producer",
            "remote_notify_port": 7000,
            "tp_size": 2,
        },
    )

    assert sent == [
        ("tx", "producer", 7000 + get_port_offset(1, 0, 2)),
        ("tx", "producer", 7000 + get_port_offset(1, 1, 2)),
    ]


def _writer_with_stub_worker(tp_rank, world_size):
    sent = []
    wrapper = SimpleNamespace(
        lock=threading.Lock(),
        done_req_ids=[],
        done_remote_allocate_req_dict={},
        waiting_for_transfer_complete=lambda _: None,
        send_notify=lambda _, host, port, **kwargs: sent.append((host, port)),
        _mark_transfer_terminal_locked=lambda _: None,
    )
    worker = SimpleNamespace(
        moriio_wrapper=wrapper,
        tp_rank=tp_rank,
        world_size=world_size,
    )
    writer = MoRIIOWriter.__new__(MoRIIOWriter)
    writer._worker_ref = lambda: worker
    writer._write_state_lock = threading.Lock()
    writer._sealed_writes = {}
    writer._clear_transfer_state = lambda _: None
    return writer, worker, sent


def _complete_write(
    writer,
    worker,
    transfer_id,
    decode_dp_rank,
    notify_port,
    decode_tp_size,
    remote_ip="127.0.0.1",
):
    info = RemoteAllocInfo(block_ids=None)  # type: ignore[arg-type]
    info.decode_dp_rank = decode_dp_rank
    worker.moriio_wrapper.done_remote_allocate_req_dict[transfer_id] = info
    writer._execute_write_task(
        WriteTask(
            request_id="req",
            transfer_id=transfer_id,
            dst_engine_id="decode",
            local_block_ids=[0],
            remote_block_ids_hint=None,
            layer_name="layer",
            event=None,  # type: ignore[arg-type]
            remote_notify_port=notify_port,
            remote_ip=remote_ip,
            remote_tp_size=decode_tp_size,
        )
    )
    info.block_ids = [0]
    info.writes_expected = 1
    info.writes_done = 1
    writer._finalize_if_complete(transfer_id, info)


@pytest.mark.parametrize("decode_dp_rank", [0, 1])
def test_write_done_targets_the_exact_decode_rank_that_bound_the_port(decode_dp_rank):
    base, producer_tp, decode_tp = 7000, 4, 2
    bound = {
        base + get_port_offset(dp, tp, decode_tp): (dp, tp)
        for dp in range(2)
        for tp in range(decode_tp)
    }

    for producer_rank in range(producer_tp):
        writer, worker, sent = _writer_with_stub_worker(producer_rank, producer_tp)
        _complete_write(
            writer,
            worker,
            f"tx-{producer_rank}",
            decode_dp_rank,
            base,
            decode_tp,
        )
        ((_, port),) = sent
        expected_tp = get_moriio_remote_tp_rank(producer_rank, producer_tp, decode_tp)
        assert bound[port] == (decode_dp_rank, expected_tp)


def test_write_done_uses_per_request_decode_tp_size():
    base = 7000
    writer, worker, sent = _writer_with_stub_worker(tp_rank=0, world_size=8)

    _complete_write(writer, worker, "tx-a", 1, base, 2, remote_ip="host-a")
    _complete_write(writer, worker, "tx-b", 1, base, 4, remote_ip="host-b")

    assert sent == [
        ("host-a", base + get_port_offset(1, 0, 2)),
        ("host-b", base + get_port_offset(1, 0, 4)),
    ]


def test_remote_tp_rank_p4_d8_floor_maps_decode_to_prefill():
    assert [get_moriio_remote_tp_rank(rank, 8, 4) for rank in range(8)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]


def test_remote_tp_rank_p8_d4_maps_to_first_prefill_rank_per_pair():
    assert [get_moriio_remote_tp_rank(rank, 4, 8) for rank in range(4)] == [
        0,
        2,
        4,
        6,
    ]


@pytest.mark.parametrize(
    ("local_tp_rank", "local_tp_size", "remote_tp_size"),
    [
        (0, 6, 4),
        (0, 4, 6),
    ],
)
def test_remote_tp_rank_invalid_non_multiple_tp_raises(
    local_tp_rank: int, local_tp_size: int, remote_tp_size: int
):
    with pytest.raises(ValueError, match="multiple"):
        get_moriio_remote_tp_rank(local_tp_rank, local_tp_size, remote_tp_size)


@pytest.mark.parametrize(
    ("local_tp_size", "remote_tp_size", "total_num_kv_heads"),
    [
        (4, 4, 8),
        (8, 4, 4),
        (4, 8, 4),
    ],
)
def test_heterogeneous_tp_head_guard_allows_supported_layouts(
    local_tp_size: int, remote_tp_size: int, total_num_kv_heads: int
):
    validate_moriio_heterogeneous_tp_kv_heads(
        local_tp_size,
        remote_tp_size,
        total_num_kv_heads,
        is_mla=False,
    )


def test_heterogeneous_tp_head_guard_allows_mla_layouts():
    validate_moriio_heterogeneous_tp_kv_heads(
        local_tp_size=2,
        remote_tp_size=4,
        total_num_kv_heads=4,
        is_mla=True,
    )


@pytest.mark.parametrize(
    ("local_tp_size", "remote_tp_size", "total_num_kv_heads"),
    [
        (4, 2, 4),
        (2, 4, 4),
    ],
)
def test_heterogeneous_tp_head_guard_rejects_split_kv_heads(
    local_tp_size: int, remote_tp_size: int, total_num_kv_heads: int
):
    with pytest.raises(NotImplementedError, match="replicated KV heads"):
        validate_moriio_heterogeneous_tp_kv_heads(
            local_tp_size,
            remote_tp_size,
            total_num_kv_heads,
            is_mla=False,
        )


def test_expected_ack_count_for_homogeneous_or_smaller_consumer_tp_is_one():
    assert get_moriio_expected_ack_count(4, 4) == 1
    assert get_moriio_expected_ack_count(8, 4) == 1


def test_expected_ack_count_for_decode_fan_in():
    assert get_moriio_expected_ack_count(4, 8) == 2


def test_expected_ack_count_rejects_non_multiple_fan_in():
    with pytest.raises(ValueError, match="multiple"):
        get_moriio_expected_ack_count(4, 6)


def test_plain_string_ack_is_backward_compatible_single_ack():
    notification_counts: dict[str, int] = {}
    completed_transfer_ids: set[str] = set()

    assert (
        resolve_moriio_transfer_ack(
            "tx-plain",
            producer_tp_size=4,
            live_transfer_ids={"tx-plain"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        == "tx-plain"
    )
    assert notification_counts == {}
    assert completed_transfer_ids == {"tx-plain"}


def test_structured_release_ack_waits_for_all_expected_acks():
    ack = MoRIIOTransferAck("tx-fanin", consumer_tp_size=8)
    notification_counts: dict[str, int] = {}
    completed_transfer_ids: set[str] = set()

    assert (
        resolve_moriio_transfer_ack(
            ack,
            producer_tp_size=4,
            live_transfer_ids={"tx-fanin"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        is None
    )
    assert notification_counts == {"tx-fanin": 1}
    assert completed_transfer_ids == set()

    assert (
        resolve_moriio_transfer_ack(
            ack,
            producer_tp_size=4,
            live_transfer_ids={"tx-fanin"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        == "tx-fanin"
    )
    assert notification_counts == {}
    assert completed_transfer_ids == {"tx-fanin"}


def test_duplicate_ack_after_completion_does_not_resolve_twice():
    ack = MoRIIOTransferAck("tx-dup", consumer_tp_size=8)
    notification_counts: dict[str, int] = {}
    completed_transfer_ids: set[str] = set()

    assert (
        resolve_moriio_transfer_ack(
            ack,
            producer_tp_size=4,
            live_transfer_ids={"tx-dup"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        is None
    )
    assert (
        resolve_moriio_transfer_ack(
            ack,
            producer_tp_size=4,
            live_transfer_ids={"tx-dup"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        == "tx-dup"
    )
    assert (
        resolve_moriio_transfer_ack(
            ack,
            producer_tp_size=4,
            live_transfer_ids={"tx-dup"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        is None
    )
    assert notification_counts == {}
    assert completed_transfer_ids == {"tx-dup"}


def test_ack_for_non_live_transfer_is_ignored():
    notification_counts: dict[str, int] = {}
    completed_transfer_ids: set[str] = set()

    assert (
        resolve_moriio_transfer_ack(
            MoRIIOTransferAck("tx-stale", consumer_tp_size=8),
            producer_tp_size=4,
            live_transfer_ids={"tx-live"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        is None
    )
    assert notification_counts == {}
    assert completed_transfer_ids == set()


def test_worker_get_finished_counts_structured_release_fan_in():
    class FakeWrapper:
        def __init__(self):
            self.batches = [
                [MoRIIOTransferAck("tx-fanin", consumer_tp_size=8)],
                [MoRIIOTransferAck("tx-fanin", consumer_tp_size=8)],
            ]

        def pop_finished_req_ids(self):
            return self.batches.pop(0)

        def shutdown(self):
            pass

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.is_producer = True
    worker.mode = MoRIIOMode.READ
    worker.world_size = 4
    worker.moriio_wrapper = FakeWrapper()
    worker.transfer_id_to_request_id = {"tx-fanin": "req-fanin"}
    worker._consumer_notification_counts = {}
    worker._completed_consumer_notifications = set()
    worker._pending_unmapped_acks = []

    assert worker.get_finished() == (set(), set())
    assert worker._consumer_notification_counts == {"tx-fanin": 1}

    assert worker.get_finished() == ({"req-fanin"}, set())
    assert worker._consumer_notification_counts == {}
    assert worker._completed_consumer_notifications == {"tx-fanin"}


def test_read_completion_sends_structured_release_with_consumer_tp_size():
    class DoneStatus:
        def Succeeded(self):
            return True

        def Failed(self):
            return False

    class FakeWrapper:
        def __init__(self):
            self.lock = threading.Lock()
            self.sent = []

        def send_notify(
            self,
            transfer_id,
            host,
            port,
            message_type=None,
            message_fields=None,
        ):
            self.sent.append((transfer_id, host, port, message_type, message_fields))

        def shutdown(self):
            pass

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.world_size = 8
    worker.moriio_wrapper = FakeWrapper()
    worker._recving_transfers = {"req": {"layer0": DoneStatus()}}
    worker._recving_transfers_callback_addr = {
        "req": ("127.0.0.1", "7000", "tx-release")
    }
    # Transfer-timeout reaping state consulted by _pop_done_transfers.
    worker._recving_transfers_start = {}

    assert worker._pop_done_transfers() == {"tx-release"}
    assert worker.moriio_wrapper.sent == [
        (
            "tx-release",
            "127.0.0.1",
            "7000",
            "release",
            {"consumer_tp_size": 8},
        )
    ]
    assert worker._recving_transfers == {}
    assert worker._recving_transfers_callback_addr == {}


def test_requested_cudagraph_mode_is_never_overridden():
    # The configured cudagraph mode is always honored: the barrier fires when
    # the operator sets cudagraph_mode=PIECEWISE, and READ mode with full
    # graphs only warns instead of silently forcing PIECEWISE.
    assert (
        MoRIIOConnector.requires_piecewise_for_cudagraph({"read_mode": True}) is False
    )
    assert (
        MoRIIOConnector.requires_piecewise_for_cudagraph({"read_mode": False}) is False
    )


def test_aborted_write_transfer_is_terminal_and_releases_producer_state():
    """A WRITE with no schedulable layers must finish producer cleanup."""
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.moriio_config = SimpleNamespace(defer_timeout=5.0)
    worker.moriio_wrapper = MoRIIOWrapper()
    worker.moriio_wrapper.done_remote_allocate_req_dict["tx"] = object()
    writer = MoRIIOWriter(worker)

    writer.abort_transfer("tx")
    writer._scheduled_writes["tx"] = 1
    writer._scheduled_layers["tx"] = {"layer0"}
    writer._sealed_writes["tx"] = 1
    writer.abort_transfer("tx")

    assert worker.moriio_wrapper.pop_finished_req_ids() == [MoRIIOTransferAck("tx")]
    assert "tx" not in worker.moriio_wrapper.done_remote_allocate_req_dict
    assert "tx" in worker.moriio_wrapper._terminal_transfer_ids
    assert "tx" not in writer._scheduled_writes
    assert "tx" not in writer._scheduled_layers
    assert "tx" not in writer._sealed_writes


def test_handshake_receive_has_a_real_socket_deadline(monkeypatch):
    """An absent peer must not occupy the handshake worker forever."""

    class TimeoutSocket:
        def __init__(self):
            self.options = {}

        def send(self, payload):
            pass

        def setsockopt(self, option, value):
            self.options[option] = value

        def recv_multipart(self):
            raise moriio_connector_module.zmq.Again()

    class SocketContext:
        def __init__(self, sock):
            self.sock = sock

        def __enter__(self):
            return self.sock

        def __exit__(self, exc_type, exc, traceback):
            return False

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.world_size = 1
    worker.tp_rank = 0
    worker.moriio_config = SimpleNamespace(transfer_timeout=0.05)
    sock = TimeoutSocket()
    monkeypatch.setattr(
        moriio_connector_module,
        "zmq_ctx",
        lambda socket_type, path: SocketContext(sock),
    )

    with pytest.raises(HandshakeError, match="receiving agent metadata"):
        worker._moriio_handshake("127.0.0.1", 6301, 1, "engine_dp0")

    timeout_ms = sock.options[moriio_connector_module.zmq.RCVTIMEO]
    assert 1 <= timeout_ms <= 50


def test_partial_handshake_does_not_register_remote_engine(monkeypatch):
    """A missing frame retry must not duplicate remote registration."""
    metadata = MoRIIOAgentMetadata(
        engine_id="remote",
        agent_metadata=b"agent",
        kv_caches_base_addr=[0],
        num_blocks=1,
        block_len=1,
        attn_backend_name="test",
    )
    first_frame = [
        b"",
        moriio_connector_module.msgspec.msgpack.encode(metadata),
    ]

    class PartialSocket:
        def __init__(self):
            self.frames = [first_frame]

        def send(self, payload):
            pass

        def setsockopt(self, option, value):
            pass

        def recv_multipart(self):
            if self.frames:
                return self.frames.pop(0)
            raise moriio_connector_module.zmq.Again()

    class SocketContext:
        def __enter__(self):
            return PartialSocket()

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeWrapper:
        def __init__(self):
            self.registered = []

        def register_remote_engine(self, agent_metadata):
            self.registered.append(agent_metadata)
            return "agent-name"

        def shutdown(self):
            pass

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.world_size = 1
    worker.tp_rank = 0
    worker.moriio_config = SimpleNamespace(transfer_timeout=0.05)
    worker.moriio_wrapper = FakeWrapper()
    worker.local_kv_cache_metadata = []
    worker.remote_kv_cache_metadata = []
    worker.layer_name_to_remote_kv_cache_metadata = {}
    worker.remote_moriio_metadata = {}
    monkeypatch.setattr(
        moriio_connector_module,
        "zmq_ctx",
        lambda socket_type, path: SocketContext(),
    )

    with pytest.raises(HandshakeError, match="receiving KV-cache metadata"):
        worker._moriio_handshake("127.0.0.1", 6301, 1, "engine_dp0")

    assert worker.moriio_wrapper.registered == []


class _ImmediateExecutor:
    def submit(self, function, *args):
        future: Future[Any] = Future()
        try:
            future.set_result(function(*args))
        except Exception as error:
            future.set_exception(error)
        return future

    def shutdown(self, wait=True):
        pass


class _QueuedExecutor:
    def __init__(self):
        self.tasks = []

    def submit(self, function, *args):
        future: Future[Any] = Future()
        self.tasks.append((future, function, args))
        return future

    def run_next(self):
        future, function, args = self.tasks.pop(0)
        try:
            future.set_result(function(*args))
        except Exception as error:
            future.set_exception(error)

    def shutdown(self, wait=True):
        pass


def _first_contact_worker():
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.world_size = 1
    worker._remote_agents = {}
    worker._engine_handshake_futures = {}
    worker._handshake_lock = threading.RLock()
    worker._handshake_initiation_executor = _ImmediateExecutor()
    worker._failed_handshake_requests = set()
    worker.aborted_transfers = []
    worker._writer = SimpleNamespace(abort_transfer=worker.aborted_transfers.append)
    worker._eager_handshaked_engines = set()
    worker._reqs_to_send = {}
    worker.moriio_config = SimpleNamespace(transfer_timeout=5.0)
    worker.mode = MoRIIOMode.READ
    worker.is_producer = False
    worker.handshake_calls = []
    worker._eager_handshake_all_dp_ranks = lambda metadata: None

    def fake_handshake(
        host,
        port,
        tp_size,
        engine_id,
        dp_rank,
        tp_rank=None,
        deadline=None,
    ):
        worker.handshake_calls.append(engine_id)
        return {f"agent-{engine_id}"}

    worker._moriio_handshake = fake_handshake
    return worker


def _meta(remote_dp_size=1, host="127.0.0.1", port=6301):
    return SimpleNamespace(
        remote_host=host,
        remote_handshake_port=port,
        remote_dp_size=remote_dp_size,
        tp_size=1,
        remote_engine_id=None,
        transfer_id="tx",
        local_block_ids=[1],
        remote_block_ids=[1],
        remote_notify_port=7000,
        remote_dp_rank=0,
        multi_pod_hosts=None,
        remote_dp_size_local=remote_dp_size,
    )


def test_write_first_contact_schedules_first_layer_for_every_request():
    """Every cold-peer WRITE includes the first attention layer."""
    worker = _first_contact_worker()
    worker.mode = MoRIIOMode.WRITE
    worker.is_producer = True
    started = []
    worker._write_blocks_for_req = lambda req_id, meta, layer_name, kv_layer: (
        started.append((req_id, layer_name))
    )

    metadata = SimpleNamespace(reqs_to_save={f"req-{i}": _meta() for i in range(4)})
    worker.save_kv_layer(metadata, "layer0", object(), None)

    assert sorted(started) == [
        ("req-0", "layer0"),
        ("req-1", "layer0"),
        ("req-2", "layer0"),
        ("req-3", "layer0"),
    ]
    assert worker.handshake_calls == ["127.0.0.1:6301_dp0"]


def test_batch_waits_once_for_a_shared_handshake():
    """One failed WRITE peer consumes one wait for the whole batch."""
    worker = _first_contact_worker()
    worker.mode = MoRIIOMode.WRITE
    worker.is_producer = True
    shared_future: Future[Any] = Future()
    worker._background_moriio_handshake = lambda engine_id, meta: shared_future
    wait_calls = []

    def fail_wait(future, req_id, remote_engine_id):
        wait_calls.append((future, req_id, remote_engine_id))
        return False

    worker._wait_for_handshake = fail_wait
    worker._write_blocks_for_req = lambda *args: pytest.fail(
        "started a write after the shared handshake failed"
    )
    metadata = SimpleNamespace(reqs_to_save={f"req-{i}": _meta() for i in range(4)})

    worker.save_kv_layer(metadata, "layer0", object(), None)

    assert wait_calls == [(shared_future, "req-0", "127.0.0.1:6301")]
    assert worker._failed_handshake_requests == {
        "req-0",
        "req-1",
        "req-2",
        "req-3",
    }
    assert worker.aborted_transfers == ["tx", "tx", "tx", "tx"]


def test_queued_engine_gets_full_timeout_when_execution_starts(monkeypatch):
    """A queued engine must not inherit an earlier peer's deadline."""
    worker = _first_contact_worker()
    executor = _QueuedExecutor()
    worker._handshake_initiation_executor = executor
    now = [0.0]
    deadlines = {}
    monkeypatch.setattr(moriio_connector_module.time, "monotonic", lambda: now[0])

    def record_deadline(
        host,
        port,
        tp_size,
        engine_id,
        dp_rank,
        tp_rank=None,
        deadline=None,
    ):
        deadlines[engine_id] = deadline
        return {f"agent-{engine_id}"}

    worker._moriio_handshake = record_deadline
    worker._background_moriio_handshake("engine-a", _meta(host="engine-a"))
    worker._background_moriio_handshake("engine-b", _meta(host="engine-b"))

    now[0] = 100.0
    executor.run_next()
    executor.run_next()
    now[0] = 200.0
    executor.run_next()
    executor.run_next()

    assert deadlines == {"engine-a_dp0": 105.0, "engine-b_dp0": 205.0}


def test_remote_engine_ready_requires_every_dp_rank():
    worker = _first_contact_worker()
    worker._remote_agents = {"eng_dp0": {"a"}}

    assert not worker._remote_engine_ready("eng", 2)

    worker._remote_agents["eng_dp1"] = {"b"}
    assert worker._remote_engine_ready("eng", 2)


def test_partial_dp_failure_retries_only_missing_rank():
    worker = _first_contact_worker()
    failed = {"127.0.0.1:6301_dp1"}

    def flaky(
        host,
        port,
        tp_size,
        engine_id,
        dp_rank,
        tp_rank=None,
        deadline=None,
    ):
        worker.handshake_calls.append(engine_id)
        if engine_id in failed:
            raise RuntimeError("dp1 unreachable")
        return {f"agent-{engine_id}"}

    worker._moriio_handshake = flaky
    future = worker._background_moriio_handshake("127.0.0.1:6301", _meta(2))
    assert not worker._wait_for_handshake(future, "req-0", "127.0.0.1:6301")
    assert sorted(worker.handshake_calls) == [
        "127.0.0.1:6301_dp0",
        "127.0.0.1:6301_dp1",
    ]

    failed.clear()
    worker.handshake_calls.clear()
    worker._begin_handshake_step()
    future = worker._background_moriio_handshake("127.0.0.1:6301", _meta(2))
    assert worker._wait_for_handshake(future, "req-0", "127.0.0.1:6301")
    assert worker.handshake_calls == ["127.0.0.1:6301_dp1"]


def test_write_handshake_failure_is_not_retried_per_layer():
    worker = _first_contact_worker()
    worker.mode = MoRIIOMode.WRITE
    worker.is_producer = True

    def always_fails(
        host,
        port,
        tp_size,
        engine_id,
        dp_rank,
        tp_rank=None,
        deadline=None,
    ):
        worker.handshake_calls.append(engine_id)
        raise RuntimeError("unreachable")

    worker._moriio_handshake = always_fails
    worker._write_blocks_for_req = lambda *args: pytest.fail(
        "scheduled a layer after the handshake failed"
    )
    metadata = SimpleNamespace(reqs_to_save={"req-0": _meta()})

    worker.save_kv_layer(metadata, "layer0", object(), None)
    worker.save_kv_layer(metadata, "layer1", object(), None)

    assert worker.handshake_calls == ["127.0.0.1:6301_dp0"]
    assert worker._failed_handshake_requests == {"req-0"}
    assert worker.aborted_transfers == ["tx"]


def test_eager_handshake_selects_wide_ep_pod_and_local_dp_rank(monkeypatch):
    calls = []

    class ImmediateExecutor:
        def submit(self, function, *args):
            calls.append(args)
            future: Future[Any] = Future()
            future.set_result(function(*args))
            return future

        def shutdown(self, wait=False):
            pass

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.world_size = 1
    worker.tp_rank = 0
    worker.use_mla = False
    worker.moriio_config = SimpleNamespace(transfer_timeout=5.0)
    worker.tp_group = SimpleNamespace(cpu_group=object())
    worker._eager_handshaked_engines = set()
    worker._remote_agents = {}
    worker.layer_name_to_remote_kv_cache_metadata = {}
    worker._handshake_lock = threading.RLock()
    worker._handshake_initiation_executor = ImmediateExecutor()
    worker._moriio_handshake = lambda *args, **kwargs: {args[3]}
    monkeypatch.setattr("torch.distributed.all_reduce", lambda *args, **kwargs: None)
    metadata = MoRIIOConnectorMetadata()
    metadata.reqs_to_recv["req"] = ReqMeta(
        transfer_id="tx",
        local_block_ids=[1],
        remote_block_ids=[2],
        remote_host="pod0",
        remote_port=6000,
        remote_handshake_port=6000,
        remote_notify_port=7000,
        remote_engine_id="engine",
        tp_size=2,
        remote_dp_size=4,
        multi_pod_hosts=["pod0", "pod1"],
        remote_dp_size_local=2,
    )

    worker._eager_handshake_all_dp_ranks(metadata)

    assert [(args[0], args[4]) for args in calls] == [
        ("pod0", 0),
        ("pod0", 1),
        ("pod1", 0),
        ("pod1", 1),
    ]


@pytest.mark.parametrize("task_source", ["queued", "deferred"])
def test_terminal_write_task_discard_clears_scheduled_state(task_source):
    writer = MoRIIOWriter.__new__(MoRIIOWriter)
    writer._write_state_lock = threading.Lock()
    writer._scheduled_writes = {"tx": 2}
    writer._scheduled_layers = {"tx": {"dense0", "indexer"}}
    writer._sealed_writes = {"tx": 2}
    writer._is_transfer_terminal = lambda transfer_id: transfer_id == "tx"
    task = SimpleNamespace(transfer_id="tx")

    if task_source == "deferred":
        writer._deferred_tasks = [task]
        writer._defer_timeout = 1.0
        writer._process_deferred_tasks()
        assert writer._deferred_tasks == []
    else:
        tasks = iter([task])

        def get_task(timeout):
            try:
                return next(tasks)
            except StopIteration as error:
                raise RuntimeError("stop worker loop") from error

        writer._deferred_tasks = []
        writer._process_deferred_tasks = lambda: None
        writer._write_task_q = SimpleNamespace(get=get_task)
        with pytest.raises(RuntimeError, match="stop worker loop"):
            writer._write_worker_loop()

    assert "tx" not in writer._scheduled_writes
    assert "tx" not in writer._scheduled_layers
    assert "tx" not in writer._sealed_writes
