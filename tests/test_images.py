"""Opt-in real PaddleOCR corpus; images and measurements stay outside tracked files."""

import importlib
import json
import os
import time
import urllib.request
from pathlib import Path

import pytest

CORPUS = json.loads(Path(__file__).with_name("corpus.json").read_text(encoding="utf-8"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_OCR_INTEGRATION") != "1", reason="set RUN_OCR_INTEGRATION=1"),
]


@pytest.fixture(scope="session")
def image_engine():
    module_name = os.getenv("OCR_BENCHMARK_MODULE", "main")
    module = importlib.import_module(module_name)
    # For controlled parser/pipeline comparisons, optionally use identical mobile
    # model files with the unmodified baseline as well as the updated service.
    model_root = os.getenv("OCR_BENCHMARK_MODELS")
    if model_root:
        root = Path(model_root)

        def model_dirs(lang):
            rec = {
                "en": "en_PP-OCRv5_mobile_rec",
                "korean": "korean_PP-OCRv5_mobile_rec",
                "ch": "PP-OCRv5_mobile_rec",
                "japan": "PP-OCRv5_mobile_rec",
            }[lang]
            return str(root / "PP-OCRv5_mobile_det"), str(root / rec), "PP-OCRv5", "PP-OCRv5_mobile_det", rec

        setattr(module, "_model_dirs_for_language_code", model_dirs)
    module.warm_ocr_engines()
    return module


@pytest.mark.parametrize("case", CORPUS, ids=[c["id"] for c in CORPUS])
def test_real_screenshot(case, image_engine, tmp_path):
    cache = Path(os.getenv("OCR_FIXTURE_DIR", str(tmp_path)))
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (case["id"] + Path(case["path"]).suffix)
    if not path.exists():
        request = urllib.request.Request(
            "https://cdn.genji.pk/screenshots/" + case["path"], headers={"User-Agent": "Genji-OCR-regression/1.0"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            path.write_bytes(response.read())
    started = time.perf_counter()
    result = image_engine.run_ocr_pipeline(
        image_engine.decode_image_bytes(path.read_bytes()), code=case["code"], time=case["time"], names=case["names"]
    )
    elapsed = time.perf_counter() - started
    record = {
        "case": case["id"],
        "classification": case["classification"],
        "elapsed": round(elapsed, 3),
        "result": result,
    }
    report = Path(os.getenv("OCR_REPORT_DIR", str(tmp_path)))
    report.mkdir(parents=True, exist_ok=True)
    (report / (case["id"] + ".json")).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    if os.getenv("OCR_CAPTURE_BASELINE") == "1":
        return  # baseline is measured, not expected to satisfy corrected safety rules
    extracted = result["extracted"]
    assert extracted["code"] in (None, case["code"])
    assert extracted["time"] in (None, case["time"])
    if case["names"]:
        assert extracted["name"] is None or extracted["name"] in case["names"]
    if case["classification"] == "confirmed_positive":
        assert extracted["name"] in case["names"]
        assert extracted["time"] == case["time"]
        assert extracted["code"] == case["code"]
    if case["id"] == "D":
        assert extracted["code"] is None
    if case["id"] == "E":
        assert extracted["time"] is None
    if extracted["time"] is not None:
        assert extracted["sources"].get("timeVerifiedBy") in ("banner", "topRight.top5")
