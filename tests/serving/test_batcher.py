from __future__ import annotations

import asyncio
import threading
import time

import pytest

from patchsentry.serving.batcher import MicroBatcher, QueueFullError


def _double(batches: list[int]):
    def function(items: list[int]) -> list[int]:
        batches.append(len(items))
        return [item * 2 for item in items]

    return function


def test_concurrent_requests_share_batches_and_keep_their_results() -> None:
    async def scenario() -> None:
        sizes: list[int] = []
        batcher = MicroBatcher(_double(sizes), max_batch_size=8, max_wait_ms=50)
        await batcher.start()
        results = await asyncio.gather(*(batcher.submit(i) for i in range(20)))
        await batcher.stop()
        assert results == [i * 2 for i in range(20)], "every caller gets its own result"
        assert max(sizes) == 8, "batches must fill up to the limit under load"
        assert sum(sizes) == 20

    asyncio.run(scenario())


def test_a_lone_request_waits_no_longer_than_the_timer() -> None:
    async def scenario() -> None:
        sizes: list[int] = []
        batcher = MicroBatcher(_double(sizes), max_batch_size=8, max_wait_ms=20)
        await batcher.start()
        started = time.perf_counter()
        assert await batcher.submit(21) == 42
        elapsed = time.perf_counter() - started
        await batcher.stop()
        assert sizes == [1]
        assert 0.015 <= elapsed < 0.5

    asyncio.run(scenario())


def test_full_batch_does_not_wait_for_the_timer() -> None:
    async def scenario() -> None:
        batcher = MicroBatcher(_double([]), max_batch_size=4, max_wait_ms=5000)
        await batcher.start()
        started = time.perf_counter()
        await asyncio.gather(*(batcher.submit(i) for i in range(4)))
        elapsed = time.perf_counter() - started
        await batcher.stop()
        assert elapsed < 1.0

    asyncio.run(scenario())


def test_failed_batch_fails_its_requests_and_the_worker_survives() -> None:
    async def scenario() -> None:
        calls = 0

        def flaky(items: list[int]) -> list[int]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("boom")
            return items

        batcher = MicroBatcher(flaky, max_batch_size=1, max_wait_ms=0)
        await batcher.start()
        with pytest.raises(RuntimeError, match="boom"):
            await batcher.submit(1)
        assert await batcher.submit(2) == 2
        await batcher.stop()

    asyncio.run(scenario())


def test_wrong_result_count_is_an_error_not_a_mismatched_response() -> None:
    async def scenario() -> None:
        batcher = MicroBatcher(lambda items: [], max_batch_size=1, max_wait_ms=0)
        await batcher.start()
        with pytest.raises(RuntimeError, match="0 results for 1"):
            await batcher.submit(1)
        await batcher.stop()

    asyncio.run(scenario())


def test_full_queue_sheds_load() -> None:
    async def scenario() -> None:
        release = threading.Event()

        def blocked(items: list[int]) -> list[int]:
            release.wait(5)
            return items

        batcher = MicroBatcher(blocked, max_batch_size=1, max_wait_ms=0, max_queue_size=2)
        await batcher.start()
        first = asyncio.create_task(batcher.submit(0))
        await asyncio.sleep(0.05)  # the worker is now blocked on the first request
        queued = [asyncio.create_task(batcher.submit(i)) for i in (1, 2)]
        await asyncio.sleep(0.01)
        with pytest.raises(QueueFullError):
            await batcher.submit(3)
        release.set()
        await asyncio.gather(first, *queued)
        await batcher.stop()

    asyncio.run(scenario())


def test_cancelled_requests_are_skipped_without_inference() -> None:
    async def scenario() -> None:
        seen: list[int] = []
        release = threading.Event()

        def function(items: list[int]) -> list[int]:
            release.wait(5)
            seen.extend(items)
            return items

        batcher = MicroBatcher(function, max_batch_size=1, max_wait_ms=0)
        await batcher.start()
        first = asyncio.create_task(batcher.submit(1))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(batcher.submit(2))
        await asyncio.sleep(0.01)
        second.cancel()
        await asyncio.gather(second, return_exceptions=True)
        release.set()
        assert await first == 1
        await batcher.stop()
        assert seen == [1]

    asyncio.run(scenario())


def test_stop_drains_queued_requests() -> None:
    async def scenario() -> None:
        batcher = MicroBatcher(_double([]), max_batch_size=2, max_wait_ms=1)
        await batcher.start()
        tasks = [asyncio.create_task(batcher.submit(i)) for i in range(6)]
        await asyncio.sleep(0)
        await batcher.stop()
        assert await asyncio.gather(*tasks) == [i * 2 for i in range(6)]

    asyncio.run(scenario())


def test_submit_when_stopped_is_an_error() -> None:
    async def scenario() -> None:
        batcher = MicroBatcher(_double([]))
        with pytest.raises(RuntimeError, match="not running"):
            await batcher.submit(1)

    asyncio.run(scenario())
