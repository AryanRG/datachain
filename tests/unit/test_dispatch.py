import queue as queue_mod
import threading
from queue import Empty, Full
from types import SimpleNamespace
from unittest.mock import MagicMock

from datachain.query.dispatch import (
    FINISHED_STATUS,
    STOP_SIGNAL,
    UDFDispatcher,
    UDFWorker,
    get_from_queue,
    put_into_queue,
)


class MockQueue:
    def __init__(self) -> None:
        self.return_empty_once = True
        self.return_full_once = True
        self.put_signal: str | None = None
        self.put_count: int = 0

    def put_nowait(self, task: str) -> None:
        if self.return_full_once:
            self.return_full_once = False
            raise Full
        assert task == STOP_SIGNAL
        self.put_signal = task
        self.put_count += 1

    def get_nowait(self) -> str:
        if self.return_empty_once:
            self.return_empty_once = False
            raise Empty
        return STOP_SIGNAL


def test_get_from_queue():
    mock_queue = MockQueue()

    assert get_from_queue(mock_queue) == STOP_SIGNAL


def test_put_into_queue():
    mock_queue = MockQueue()

    assert put_into_queue(mock_queue, STOP_SIGNAL) is None
    assert mock_queue.put_signal == STOP_SIGNAL


class ThreadQueue(queue_mod.Queue):
    def close(self) -> None:
        pass

    def join_thread(self) -> None:
        pass


class ThreadProcess:
    def __init__(self, name: str, target) -> None:
        self.name = name
        self.thread = threading.Thread(target=target, daemon=True)
        self.exitcode = None

    def start(self) -> None:
        self.thread.start()

    def is_alive(self) -> bool:
        return self.thread.is_alive()


def test_dispatcher_runs_more_batches_than_its_buffer():
    dispatcher = UDFDispatcher.__new__(UDFDispatcher)
    dispatcher.ctx = SimpleNamespace(Queue=ThreadQueue, Process=ThreadProcess)
    dispatcher.is_batching = False
    dispatcher.buffer_size = 2
    dispatcher.input_batch_size = lambda n_workers: 3
    seen = []
    catalog = MagicMock()
    catalog.warehouse.clone().dataset_rows_select_from_ids = lambda _, ids, __: ids

    def run_worker() -> None:
        worker = UDFWorker(
            catalog,
            MagicMock(),
            dispatcher.task_queue,
            dispatcher.done_queue,
            MagicMock(),
            MagicMock(),
            cache=False,
            is_batching=False,
            batch_size=None,
            udf_fields=[],
        )
        seen.extend(worker.get_inputs())
        worker.stats.send(FINISHED_STATUS)

    dispatcher._run_worker = run_worker
    runner = threading.Thread(
        target=dispatcher.run_udf_parallel,
        args=(2, [[i] for i in range(300)]),
        daemon=True,
    )
    runner.start()
    runner.join(timeout=30)

    assert not runner.is_alive()
    assert sorted(seen) == list(range(300))
