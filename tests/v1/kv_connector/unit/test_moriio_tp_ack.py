# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOMode,
    MoRIIOTransferAck,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnector,
    MoRIIOConnectorWorker,
    get_moriio_expected_ack_count,
    get_moriio_remote_tp_rank,
    resolve_moriio_transfer_ack,
    validate_moriio_heterogeneous_tp_kv_heads,
)


def test_remote_tp_rank_same_tp_maps_to_self():
    assert [get_moriio_remote_tp_rank(rank, 4, 4) for rank in range(4)] == [
        0,
        1,
        2,
        3,
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
    from types import SimpleNamespace

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
    worker.moriio_config = SimpleNamespace(transfer_timeout=5.0)
    worker.moriio_wrapper = FakeWrapper()
    worker._recving_transfers = {"req": {"layer0": DoneStatus()}}
    worker._recving_transfers_callback_addr = {
        "req": ("127.0.0.1", "7000", "tx-release")
    }
    # Transfer-timeout reaping state consulted by _pop_done_transfers.
    worker._recving_transfers_start = {}
    worker._recving_block_ids = {"req": {0}}
    worker._invalid_block_ids = set()
    worker._unsynchronized_read_requests = set()

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


def test_read_mode_requires_piecewise_cudagraphs():
    assert MoRIIOConnector.requires_piecewise_for_cudagraph({"read_mode": True})
    assert MoRIIOConnector.requires_piecewise_for_cudagraph({"read_mode": "1"})
    assert (
        MoRIIOConnector.requires_piecewise_for_cudagraph({"read_mode": False}) is False
    )


class _Status:
    """Minimal stand-in for a MoRIIO transfer status handle."""

    def __init__(self, succeeded=False, failed=False, message="", code=0):
        self._succeeded = succeeded
        self._failed = failed
        self._message = message
        self._code = code

    def Succeeded(self):
        return self._succeeded

    def Failed(self):
        return self._failed

    def Message(self):
        return self._message

    def Code(self):
        return self._code


class _RecordingWrapper:
    def __init__(self):
        self.lock = threading.Lock()
        self.sent = []

    def send_notify(
        self, transfer_id, host, port, message_type=None, message_fields=None
    ):
        self.sent.append((transfer_id, message_type))

    def shutdown(self):
        pass


def _read_worker(status_list, wrapper=None, block_ids=(10, 11)):
    from types import SimpleNamespace

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.is_producer = False
    worker.mode = MoRIIOMode.READ
    worker.world_size = 2
    worker.moriio_config = SimpleNamespace(transfer_timeout=5.0)
    worker.moriio_wrapper = wrapper or _RecordingWrapper()
    # One status per layer, matching how _read_blocks registers transfers.
    worker._recving_transfers = {
        "req": {f"layer{i}": status for i, status in enumerate(status_list)}
    }
    worker._recving_transfers_callback_addr = {"req": ("127.0.0.1", "7000", "tx-read")}
    worker._recving_transfers_start = {}
    worker._recving_block_ids = {"req": set(block_ids)}
    worker._invalid_block_ids = set()
    worker._unsynchronized_read_requests = set()
    worker.transfer_id_to_request_id = {"tx-read": "req"}
    return worker


def test_read_failure_releases_and_reports_its_blocks_as_invalid():
    """A failed layer frees prefill's blocks and invalidates the decode blocks.

    Synchronous READ requests are already RUNNING, so get_finished() must not
    report them as asynchronous completions. The invalid IDs in the same worker
    pass make the scheduler discard this step's output and recompute the blocks.
    """
    worker = _read_worker(
        [_Status(succeeded=True), _Status(failed=True, message="io", code=7)],
        block_ids=(10, 11),
    )

    assert worker.get_finished() == (set(), set())
    assert worker.moriio_wrapper.sent == [("tx-read", "release")]
    assert worker._recving_transfers == {}
    assert worker.get_block_ids_with_load_errors() == {10, 11}


def test_successful_read_reports_no_invalid_blocks():
    worker = _read_worker([_Status(succeeded=True), _Status(succeeded=True)])

    assert worker.get_finished() == (set(), set())
    assert worker.get_block_ids_with_load_errors() == set()


def test_invalid_block_ids_are_drained_once():
    """The scheduler must not see the same bad block twice."""
    worker = _read_worker([_Status(failed=True, message="io", code=7)], block_ids=(3,))
    worker.get_finished()

    assert worker.get_block_ids_with_load_errors() == {3}
    assert worker.get_block_ids_with_load_errors() == set()


def test_read_failure_waits_for_every_layer_before_invalidating(monkeypatch):
    """A block cannot be reused while another layer may still DMA into it."""
    pending = _Status()
    worker = _read_worker(
        [_Status(failed=True, message="io", code=7), pending],
        block_ids=(3,),
    )
    worker._recving_transfers_start = {"req": 0.0}
    sleep_calls = []

    def finish_pending_layer(delay):
        sleep_calls.append(delay)
        pending._succeeded = True

    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio."
        "moriio_connector.time.sleep",
        finish_pending_layer,
    )
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio."
        "moriio_connector.time.monotonic",
        lambda: 1.0,
    )

    worker.get_finished()

    assert sleep_calls == [0.001]
    assert worker.get_block_ids_with_load_errors() == {3}
    assert worker.moriio_wrapper.sent == [("tx-read", "release")]


def test_read_barrier_timeout_invalidates_even_if_rdma_later_succeeds(monkeypatch):
    """A late Success cannot validate KV that attention may already have read."""
    from types import SimpleNamespace

    from vllm.config import CUDAGraphMode

    pending = _Status()
    worker = _read_worker([pending], block_ids=(3,))
    monotonic_values = iter((0.0, 6.0, 6.0))
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio."
        "moriio_connector.get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE),
    )
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio."
        "moriio_connector.time.monotonic",
        lambda: next(monotonic_values),
    )

    worker.wait_for_layer_load("layer0")
    pending._succeeded = True
    worker.get_finished()

    assert worker.get_block_ids_with_load_errors() == {3}
    assert worker.moriio_wrapper.sent == [("tx-read", "release")]


def test_read_timeout_fails_closed_without_releasing_live_dma(monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
        TransferError,
    )

    worker = _read_worker([_Status()], block_ids=(3,))
    worker._recving_transfers_start = {"req": 0.0}
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio."
        "moriio_connector.time.monotonic",
        lambda: 6.0,
    )

    with pytest.raises(TransferError, match="cannot be safely cancelled"):
        worker.get_finished()

    assert worker._recving_transfers
    assert worker.get_block_ids_with_load_errors() == set()
    assert worker.moriio_wrapper.sent == []
