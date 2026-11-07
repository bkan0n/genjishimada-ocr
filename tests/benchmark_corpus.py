"""Capture each corpus case in a fresh process with a hard native-inference timeout.

Usage: python tests/benchmark_corpus.py --report-dir .ocr-cache/updated
For an unmodified baseline, also pass --baseline-ref <commit>.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref")
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--fixture-dir", default=".ocr-cache")
    parser.add_argument("--models", help="Optional directory with the four mobile models; use identically in both runs")
    parser.add_argument("--timeout", type=float, default=75, help="Per-process budget including imports/model warmup")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = Path(args.report_dir).resolve()
    report.mkdir(parents=True, exist_ok=True)
    corpus = json.loads((root / "tests/corpus.json").read_text(encoding="utf-8"))
    env = os.environ.copy()
    env.update(
        RUN_OCR_INTEGRATION="1",
        OCR_FIXTURE_DIR=str(Path(args.fixture_dir).resolve()),
        OCR_REPORT_DIR=str(report),
        PYTHONIOENCODING="utf-8",
    )
    failures = 0
    with tempfile.TemporaryDirectory(prefix="genji-ocr-baseline-") as temporary:
        if args.baseline_ref:
            source = subprocess.check_output(["git", "show", f"{args.baseline_ref}:main.py"], cwd=root)
            Path(temporary, "baseline_main.py").write_bytes(source)
            env.update(OCR_BENCHMARK_MODULE="baseline_main", OCR_CAPTURE_BASELINE="1")
        else:
            env.update(OCR_BENCHMARK_MODULE="main", OCR_CAPTURE_BASELINE="0")
        if args.models:
            env["OCR_BENCHMARK_MODELS"] = str(Path(args.models).resolve())
        search = [temporary, str(root), *[p for p in sys.path if p.endswith("site-packages")]]
        for case in corpus:
            label = case["id"]
            script = f"import sys; sys.path[:0]={search!r}; import pytest; sys.exit(pytest.main("
            script += repr(["-q", f"tests/test_images.py::test_real_screenshot[{label}]"]) + "))"
            record_path = report / (label + ".json")
            error = None
            with (report / (label + ".log")).open("w", encoding="utf-8") as log:
                try:
                    # Use the real executable directly on Windows so killing a
                    # timed-out venv launcher cannot orphan native OCR children.
                    result = subprocess.run(
                        [getattr(sys, "_base_executable", sys.executable), "-c", script],
                        env=env,
                        cwd=root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout,
                    )
                    if result.returncode:
                        error = f"process exited {result.returncode}; see {label}.log"
                except subprocess.TimeoutExpired:
                    error = f"exceeded {args.timeout}s process budget (includes model warmup)"
            if error:
                failures += 1
                record = (
                    json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {"case": label}
                )
                record["error"] = error
                record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            print(label, error or "captured", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(run())
