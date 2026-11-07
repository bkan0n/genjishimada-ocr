"""Native OCR must remain serialized even when its response has timed out."""

import asyncio
import contextlib
import time

import numpy as np
from fastapi import FastAPI

import main


def test_worker_waits_for_native_inference_after_timeout(monkeypatch):
    calls = []

    def inference(image, **kwargs):
        calls.append(("start", time.perf_counter()))
        time.sleep(0.08)
        calls.append(("end", time.perf_counter()))
        return {"extracted": {}}

    monkeypatch.setattr(main, "run_ocr_pipeline", inference)
    monkeypatch.setattr(main, "OCR_TIMEOUT_S", 0.02)

    async def exercise():
        app = FastAPI()
        app.state.ocr_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        futures = [loop.create_future(), loop.create_future()]
        for future in futures:
            await app.state.ocr_queue.put(
                main._OcrJob(np.zeros((1, 1, 3)), "", future, time.perf_counter(), "", None, [])
            )
        worker = asyncio.create_task(main._ocr_worker(app))
        results = await asyncio.gather(*futures, return_exceptions=True)
        assert all(isinstance(result, TimeoutError) for result in results)
        await asyncio.wait_for(app.state.ocr_queue.join(), timeout=2)
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker

    asyncio.run(exercise())
    assert [kind for kind, _ in calls] == ["start", "end", "start", "end"]
    assert calls[2][1] >= calls[1][1]
