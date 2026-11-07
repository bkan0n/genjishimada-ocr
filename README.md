# GenjiPK OCR

FastAPI service for player names, completion/leaderboard times, and Workshop map codes in Overwatch parkour screenshots. Uses PaddleOCR 3.3.3 and explicitly selected PP-OCRv5 mobile models on CPU.

## API

`GET /ping` reports health and warmed language models. `POST /extract` downloads an image URL and queues OCR. The request uses **snake_case** `image_url`; it does not accept a base64 upload.

```bash
curl -X POST http://localhost:8000/extract \
  -H 'Content-Type: application/json' \
  -d '{"image_url":"https://cdn.genji.pk/screenshots/example.png","code":"WZJ79","time":1092.35,"names":["Snowmän"]}'
```

`code`, `time`, and `names` are optional. Names are allowed aliases, not proof of identity. Omitting `time` (or passing `0` for compatibility) requests extraction without a submitted-time constraint.

```json
{
  "extracted": {
    "name": "Snowmän",
    "time": 1092.35,
    "code": "WZJ79",
    "sources": {
      "name": "hint",
      "nameVerifiedBy": "banner",
      "code": "hint",
      "codeVerifiedBy": "topLeft",
      "time": "hint",
      "timeVerifiedBy": "banner"
    },
    "texts": {
      "topLeft": {"en":"MAP CODE: WZJ79\nTIME 1092.35 SEC","whiteMaskEn":"","cyanMaskEn":""},
      "banner": {"en":"SNOWMÄN MISSION COMPLETE! TIME 1092.35 SEC","text":"SNOWMÄN MISSION COMPLETE! TIME 1092.35 SEC"},
      "topRight": {"en":"SNOWMÄN\n1092.35 SEC","top5":"SNOWMÄN\n1092.35 SEC"},
      "bottomLeft": {"en":"SNOWMÄN"}
    },
    "diagnostics": {
      "reasons": [],
      "observedName": "SNOWMÄN",
      "codeCandidates": ["WZJ79"],
      "codeEvidence": [],
      "timeCandidates": [],
      "ocrPasses": 4
    }
  }
}
```

The example abbreviates candidate diagnostics. Actual candidates include confidence, ROI, language, preprocessing variant, and screenshot-relative bounds. `texts` retains existing nested ROI keys and adds per-pass strings; newlines separate reconstructed rows. Text blobs are diagnostic only. The displayed OCR name is preserved in `diagnostics.observedName`; `name` returns an allowed alias only when recognition supports it. Schemas are available at `/docs` and `/redoc`.

Unconfirmed fields are `null`. Reason codes include `name_not_detected`, `name_mismatch`, `code_not_detected`, `code_mismatch`, `time_mismatch`, `name_time_pair_unconfirmed`, and `completion_not_confirmed`. A confirmed code alone does not verify the record.

## Verification rules

- A name/time pair must come from one localized completion banner or one spatially reconstructed leaderboard entry. Workshop layouts with a time directly below its player name are supported. Fragments from different OCR passes never form a name/time pair.
- Without submitted aliases, an explicit named completion banner can identify the player. Leaderboard extraction requires a matching HUD identity; an arbitrary leaderboard row cannot identify the submitter.
- Live top-left timers, checkpoint splits, and a name appearing elsewhere do not prove completion. A matching completion banner takes priority over older leaderboard entries. Conflicting readings for the same identity/source remain unconfirmed.
- Time comparison uses exact integer centiseconds and an absolute tolerance, default `0`. Range validation defaults to 10–15360 seconds. A mismatching submitted time returns `null`; hints never supply missing OCR evidence.
- Workshop codes are literal 4–6 alphanumeric characters, including all-letter codes. `O/0`, `I/1`, partial codes, and different final characters are distinct. An enlarged or localized pass must independently read a complete code to resolve ambiguity. Localized code-label geometry can establish the field's location for another pass that reads the literal code there.
- Name comparison normalizes Unicode compatibility forms, case, spacing, and Latin diacritics separately from display. A single familiar Latin digit/letter confusion in a name of at least five characters may match. Short CJK names and arbitrary character edits are not fuzzy-matched.

## Running

Python 3.11+ is required. Linux is recommended for deployment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. Alternatively:

```bash
docker build -t genjishimada-ocr .
docker run --rm -p 8000:8000 genjishimada-ocr
```

The process warms English, Korean, Japanese, and Chinese models at startup. Existing Docker model directories under `~/.paddleocr/whl` are supported; absent directories trigger Paddle's normal model download. `PADDLE_PDX_CACHE_HOME` can relocate downloaded models. Models are reused, and OCR calls are serialized through one FIFO worker per process. A timed-out request retains the inference slot until native OCR finishes. More workers duplicate models and inference capacity; start with one.

## Configuration

| Variable | Default | Behavior |
|---|---|---|
| `OCR_MKLDNN` | `1` | oneDNN enabled; recognized PIR errors retry once with a cached engine with oneDNN disabled |
| `OCR_CPU_THREADS` | auto, at most 4 | CPU threads per engine |
| `OCR_TIMEOUT_S` | `60` | Inference response deadline; no new fallback starts after this request budget |
| `OCR_QUEUE_MAXSIZE` | `32` | Queued jobs; `0` means unlimited |
| `OCR_QUEUE_WAIT_S` | `180` | Queue wait allowance |
| `OCR_DEBUG_TEXTS` | `0` | Include detailed polygons/provenance in response diagnostics |
| `HINT_TIME_ABS_TOL` | `0` | Absolute time tolerance in seconds, never a percentage |
| `HINT_MATCH_THRESHOLD` | `0.90` | Name comparison threshold; map codes always require literal equality |
| `MIN_VALID_TIME_SECONDS` | `10.0` | Lower bound for time extraction |

`FAST_OCR`, `ACCURATE_OCR`, `OCR_EARLY_EXIT`, and `OCR_TRUST_HINTS` remain import-compatible with legacy helpers. The evidence pipeline uses bounded, targeted fallbacks; these legacy settings cannot enable live-timer verification or trust unsupported hints. Set `OCR_MKLDNN=0` if necessary for CPU/oneDNN compatibility. OpenCV threading is disabled; BLAS/OMP defaults are set before Paddle is imported.

## Tests and investigation

```bash
pip install pytest ruff pyright
ruff check .
pyright
pytest -q
git diff --check
```

Unit tests load no Paddle models and require no network. They cover Unicode names, localized banners, geometry/transforms, split fields, code disagreements, strict time mismatches, live-timer rejection, aliases, response compatibility, and serialization after timeout.

Image tests are **opt-in** and run the actual pipeline. They download only the nine supplied screenshots into temporary storage, or an explicitly chosen ignored cache. Models may download on first use. Classifications and manually transcribed aliases are in `tests/corpus.json`; recognition targets allow `null` and are not asserted as guaranteed positives. Positive cases assert complete fields; mismatch cases assert rejection.

```bash
RUN_OCR_INTEGRATION=1 OCR_FIXTURE_DIR=.ocr-cache OCR_REPORT_DIR=.ocr-cache/results \
  pytest -q tests/test_images.py
```

PowerShell:

```powershell
$env:RUN_OCR_INTEGRATION = '1'
$env:OCR_FIXTURE_DIR = '.ocr-cache'
$env:OCR_REPORT_DIR = '.ocr-cache/results'
pytest -q tests/test_images.py
```

Set `OCR_BENCHMARK_MODULE` to an importable baseline module and `OCR_CAPTURE_BASELINE=1` to capture its behavior without corrected assertions. `OCR_BENCHMARK_MODELS` optionally points at a directory containing the four named mobile models so both versions use identical weights. Per-image measurements include latency and the response. Run sequentially on otherwise idle hardware for meaningful latency comparisons.

See [the investigation and corpus report](docs/ocr-evidence-validation.md) for root causes, validation results, remaining limitations, and baseline comparisons.

## Layout

- `main.py`: API, queue, preprocessing, inference adapters, orchestration, and compatible text helpers.
- `evidence.py`: model-independent Unicode/code/time parsing, structured detections, spatial grouping, and relationship validation.
- `tests/`: deterministic tests and the opt-in image corpus.

MIT. Built with [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), FastAPI, OpenCV, NumPy, and Pillow.
