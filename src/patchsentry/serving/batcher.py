"""Dynamic micro-batching: trade a small bounded wait for much higher throughput under load.

A CNN forward pass costs far less per image in a batch than alone. The first request in an empty
queue starts a timer of ``max_wait_ms``; requests arriving meanwhile join its batch, up to
``max_batch_size``. Under light traffic, batches stay small and the added latency is at most the
timer. Under heavy traffic, batches fill up before the timer fires, so the wait disappears. The
batch size therefore adapts to the load with no tuning.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class QueueFullError(Exception):
    """The queue is at capacity. The caller should shed load (HTTP 503)."""


@dataclass
class _Pending(Generic[RequestT, ResultT]):
    request: RequestT
    future: asyncio.Future[ResultT]


class MicroBatcher(Generic[RequestT, ResultT]):
    """Collects concurrent requests into batches for a blocking, batch-capable function.

    The function runs on one dedicated thread, for two reasons: the event loop stays free to accept
    requests, and only one batch runs at a time, so ONNX Runtime's own thread pool is not
    oversubscribed.
    """

    def __init__(
        self,
        function: Callable[[list[RequestT]], Sequence[ResultT]],
        max_batch_size: int = 16,
        max_wait_ms: float = 5.0,
        max_queue_size: int = 256,
        on_batch: Callable[[int], None] | None = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        self._function = function
        self._max_batch_size = max_batch_size
        self._max_wait = max_wait_ms / 1000.0
        self._max_queue_size = max_queue_size
        self._on_batch = on_batch
        self._queue: asyncio.Queue[_Pending[RequestT, ResultT]] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._executor: ThreadPoolExecutor | None = None

    async def start(self) -> None:
        self._queue = asyncio.Queue(maxsize=self._max_queue_size)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        self._worker = asyncio.create_task(self._run(), name="micro-batcher")

    async def stop(self) -> None:
        """Stop accepting work, finish the queued requests, then shut down."""
        queue, self._queue = self._queue, None
        if queue is not None:
            await queue.join()
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize() if self._queue is not None else 0

    async def submit(self, request: RequestT) -> ResultT:
        if self._queue is None:
            raise RuntimeError("MicroBatcher is not running.")
        future: asyncio.Future[ResultT] = asyncio.get_running_loop().create_future()
        try:
            self._queue.put_nowait(_Pending(request, future))
        except asyncio.QueueFull as error:
            raise QueueFullError from error
        return await future

    async def _collect(self, queue: asyncio.Queue) -> list[_Pending[RequestT, ResultT]]:
        batch = [await queue.get()]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._max_wait
        while len(batch) < self._max_batch_size:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(queue.get(), remaining))
            except TimeoutError:
                break
        return batch

    async def _run(self) -> None:
        queue = self._queue
        assert queue is not None
        loop = asyncio.get_running_loop()
        while True:
            batch = await self._collect(queue)
            # A client that disconnected has a cancelled future: do not spend inference on it.
            live = [item for item in batch if not item.future.done()]
            try:
                if live:
                    results = await loop.run_in_executor(
                        self._executor, self._function, [item.request for item in live]
                    )
                    if len(results) != len(live):
                        raise RuntimeError(f"Got {len(results)} results for {len(live)} requests.")
                    if self._on_batch is not None:
                        self._on_batch(len(live))
                    for item, result in zip(live, results, strict=True):
                        if not item.future.done():
                            item.future.set_result(result)
            except Exception as error:
                # One failed batch must fail only its own requests and leave the worker running.
                logger.exception("Batch of %d failed", len(live))
                for item in live:
                    if not item.future.done():
                        item.future.set_exception(error)
            finally:
                for _ in batch:
                    queue.task_done()
