# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_common
from vllm.distributed.kv_transfer.kv_connector.v1.moriio import (
    moriio_connector as moriio_connector_module,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOMode,
    MoRIIOTransferAck,
    MoRIIOWriteAck,
    RemoteAllocInfo,
    WriteTask,
    get_port_offset,
    resolve_peer_tp_size,
    validate_moriio_write_tp_topology,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnector,
    MoRIIOConnectorScheduler,
    MoRIIOConnectorWorker,
    get_moriio_expected_ack_count,
    get_moriio_expected_write_ack_count,
    get_moriio_remote_tp_rank,
    resolve_moriio_transfer_ack,
    resolve_moriio_write_ack,
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
            "remote_dp_size": 2,
            "remote_dp_size_local": 2,
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


def test_write_completion_maps_producer_tp_to_decode_pod_and_port():
    writer = MoRIIOWriter.__new__(MoRIIOWriter)
    writer._worker_ref = lambda: SimpleNamespace(tp_rank=5, world_size=8)
    task = SimpleNamespace(
        remote_tp_size=4,
        remote_ip="pod0",
        remote_hosts=("pod0", "pod1"),
        remote_notify_port=7000,
        remote_dp_size_local=8,
    )

    assert writer._resolve_write_completion_endpoint(task, 11) == ("pod1", 7014)


def test_write_tasks_keep_each_requests_decode_topology():
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.world_size = 8
    scheduled = []
    worker.schedule_write_blocks = lambda **kwargs: scheduled.append(kwargs)

    def meta(host, tp_size, dp_local, hosts):
        return SimpleNamespace(
            transfer_id=f"tx-{host}",
            remote_engine_id=f"engine-{host}",
            local_block_ids=[1],
            remote_block_ids=[2],
            remote_notify_port=7000,
            remote_host=host,
            tp_size=tp_size,
            remote_dp_size=dp_local,
            remote_dp_size_local=dp_local,
            multi_pod_hosts=hosts,
        )

    worker._write_blocks_for_req(
        "req-a", meta("host-a", 2, 4, ["a0", "a1"]), "layer", object()
    )
    worker._write_blocks_for_req(
        "req-b", meta("host-b", 8, 2, ["b0"]), "layer", object()
    )

    assert [task["remote_tp_size"] for task in scheduled] == [2, 8]
    assert [task["remote_dp_size_local"] for task in scheduled] == [4, 2]
    assert [task["remote_hosts"] for task in scheduled] == [
        ("a0", "a1"),
        ("b0",),
    ]
    assert not hasattr(worker, "multi_pod_hosts")
    assert not hasattr(worker, "remote_dp_size_local")


def test_write_completion_waits_for_every_mapped_producer():
    ack = MoRIIOWriteAck("tx-write", producer_tp_size=8)
    notification_counts: dict[str, int] = {}
    completed_transfer_ids: set[str] = set()
    results = [
        resolve_moriio_write_ack(
            ack,
            decode_tp_size=4,
            live_transfer_ids={"tx-write"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        for _ in range(3)
    ]

    assert results == [None, "tx-write", None]
    assert notification_counts == {}
    assert completed_transfer_ids == {"tx-write"}


def test_write_completion_rejects_decode_tp_larger_than_producer():
    with pytest.raises(NotImplementedError, match="decode TP larger"):
        get_moriio_expected_write_ack_count(4, 8)


def test_worker_counts_write_done_fan_in_and_retries_early_ack():
    class FakeWrapper:
        def __init__(self):
            self.batches = [
                [MoRIIOWriteAck("tx-write", 8)],
                [MoRIIOWriteAck("tx-write", 8)],
                [MoRIIOWriteAck("tx-early", 4)],
                [],
            ]

        def pop_finished_write_req_ids(self):
            return self.batches.pop(0)

        def shutdown(self):
            pass

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.is_producer = False
    worker.mode = MoRIIOMode.WRITE
    worker.world_size = 4
    worker.moriio_wrapper = FakeWrapper()
    worker.transfer_id_to_request_id = {"tx-write": "req-write"}
    worker._pending_unmapped_write_acks = []
    worker._write_notification_counts = {}
    worker._completed_write_notifications = set()

    assert worker.get_finished() == (set(), set())
    assert worker.get_finished() == (set(), {"req-write"})
    assert worker.get_finished() == (set(), set())
    assert worker._pending_unmapped_write_acks == [MoRIIOWriteAck("tx-early", 4)]

    worker.transfer_id_to_request_id = {"tx-early": "req-early"}
    assert worker.get_finished() == (set(), {"req-early"})
    assert worker._pending_unmapped_write_acks == []


def test_write_completion_queue_preserves_duplicate_transfer_ids():
    ack = MoRIIOWriteAck("tx-write", 8)
    wrapper = MoRIIOWrapper.__new__(MoRIIOWrapper)
    wrapper.lock = threading.Lock()
    wrapper.done_write_cache_req_ids = [ack, ack]

    assert wrapper.pop_finished_write_req_ids() == [ack, ack]


def test_write_done_message_carries_producer_tp_size():
    previous_role = moriio_common.get_role()
    try:
        moriio_common.set_role(moriio_common.ROLE.CONSUMER)
        wrapper = MoRIIOWrapper.__new__(MoRIIOWrapper)
        wrapper.lock = threading.Lock()
        wrapper.done_write_cache_req_ids = []

        wrapper._handle_write_done_message(
            {"transfer_id": "tx-write", "producer_tp_size": 8}
        )

        assert wrapper.done_write_cache_req_ids == [MoRIIOWriteAck("tx-write", 8)]
    finally:
        moriio_common.set_role(previous_role)


def test_advertised_notify_port_remains_the_cluster_base(monkeypatch):
    monkeypatch.setattr(moriio_common, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        moriio_common, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(moriio_common, "get_open_port", lambda: 32000)
    config = moriio_common.MoRIIOConfig.from_vllm_config(
        SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_connector="MoRIIOConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "host_ip": "127.0.0.1",
                    "http_port": 8000,
                    "handshake_port": 6000,
                    "notify_port": 7000,
                    "read_mode": True,
                },
            ),
            parallel_config=SimpleNamespace(
                data_parallel_rank=1,
                data_parallel_size=2,
                data_parallel_size_local=2,
            ),
        )
    )

    assert config.notify_port == 7003
    assert config.base_notify_port == 7000
    assert moriio_common.get_moriio_notification_endpoint(
        "127.0.0.1", [], config.base_notify_port, 1, 2, 1, 2
    ) == ("127.0.0.1", config.notify_port)
    assert moriio_common.get_moriio_notification_endpoint(
        "127.0.0.1", [], config.notify_port, 1, 2, 1, 2
    ) == ("127.0.0.1", 7006)


def test_prefill_notify_endpoints_cover_the_producer_width(monkeypatch):
    monkeypatch.setattr(
        moriio_connector_module,
        "get_peer_zmq_from_request_id",
        lambda *args, **kwargs: None,
    )
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler.tp_size = 4

    endpoints = scheduler._prefill_notify_endpoints(
        "req",
        {
            "remote_host": "10.0.0.1",
            "remote_notify_port": 7000,
            "remote_tp_size": 8,
            "remote_dp_rank": 1,
            "remote_dp_size": 2,
            "remote_dp_size_local": 2,
        },
    )

    assert endpoints == [("10.0.0.1", port) for port in range(7008, 7016)]


@pytest.mark.parametrize(
    ("producer_tp_size", "decode_tp_size"), [(8, 4), (4, 4), (0, 4)]
)
def test_write_topology_is_validated_before_allocation(
    producer_tp_size, decode_tp_size
):
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler.is_producer = False
    scheduler.mode = MoRIIOMode.WRITE
    scheduler.tp_size = decode_tp_size
    request = SimpleNamespace(
        prompt_token_ids=list(range(17)),
        kv_transfer_params={"remote_tp_size": producer_tp_size},
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (17, True)


@pytest.mark.parametrize(
    ("producer_tp_size", "decode_tp_size", "error"),
    [(2, 4, NotImplementedError), (4, 3, ValueError)],
)
def test_write_topology_rejects_unsupported_layout_before_allocation(
    producer_tp_size, decode_tp_size, error
):
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler.is_producer = False
    scheduler.mode = MoRIIOMode.WRITE
    scheduler.tp_size = decode_tp_size
    request = SimpleNamespace(
        prompt_token_ids=list(range(17)),
        kv_transfer_params={"remote_tp_size": producer_tp_size},
    )

    with pytest.raises(error):
        scheduler.get_num_new_matched_tokens(request, 0)


def _write_decode_scheduler(tp_size=4):
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler.mode = MoRIIOMode.WRITE
    scheduler.is_producer = False
    scheduler.tp_size = tp_size
    scheduler._global_dp_rank = 0
    scheduler._is_kv_master = True
    scheduler._reqs_need_save = {}
    scheduler._req_kv_params = {}
    scheduler.map_request_id = lambda request_id, transfer_id: None
    return scheduler


def test_write_block_ready_reaches_every_producer_rank():
    scheduler = _write_decode_scheduler()
    sent = []
    scheduler.send_notify_block = lambda **kwargs: sent.append(kwargs)
    params = {
        "do_remote_prefill": True,
        "transfer_id": "tx-write",
        "remote_host": "127.0.0.1",
        "remote_notify_port": 7000,
        "remote_tp_size": 8,
        "remote_dp_rank": 0,
        "remote_dp_size": 1,
        "remote_dp_size_local": 1,
    }
    request = SimpleNamespace(request_id="req-write", kv_transfer_params=params)
    blocks = SimpleNamespace(get_block_ids=lambda: [[11, 12]])

    scheduler.update_state_after_alloc(request, blocks, num_external_tokens=1)

    assert [message["port"] for message in sent] == list(range(7000, 7008))
    assert {tuple(message["block_notify_list"]) for message in sent} == {(11, 12)}


def test_unchosen_write_connector_releases_instead_of_staging_empty_write():
    scheduler = _write_decode_scheduler()
    released = []
    staged = []
    scheduler._release_write_prefill_blocks = lambda *args: released.append(args)
    scheduler.send_notify_block = lambda **kwargs: staged.append(kwargs)
    params = {
        "do_remote_prefill": True,
        "transfer_id": "tx-write",
        "remote_host": "127.0.0.1",
        "remote_notify_port": 7000,
        "remote_tp_size": 8,
    }
    request = SimpleNamespace(request_id="req-write", kv_transfer_params=params)
    blocks = SimpleNamespace(get_block_ids=lambda: [[11, 12]])

    scheduler.update_state_after_alloc(request, blocks, num_external_tokens=0)

    assert released == [("req-write", params)]
    assert staged == []


def test_write_topology_validator_rejects_nonpositive_sizes():
    with pytest.raises(ValueError, match="positive"):
        validate_moriio_write_tp_topology(0, 4)


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
