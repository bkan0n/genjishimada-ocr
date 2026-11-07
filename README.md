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
      "ocrPasses": 8
    }
  }
}
```

The example abbreviates candidate diagnostics. Actual candidates include confidence, ROI, language, preprocessing variant, and screenshot-relative bounds. `texts` retains existing nested ROI keys and adds per-pass strings; newlines separate reconstructed rows. Text blobs are diagnostic only. The displayed OCR name is preserved in `diagnostics.observedName`; `name` returns an allowed alias only when recognition supports it. Schemas are available at `/docs` and `/redoc`.

Unconfirmed fields are `null`. Reason codes include `name_not_detected`, `name_mismatch`, `code_not_detected`, `code_mismatch`, `time_mismatch`, `name_time_pair_unconfirmed`, and `completion_not_confirmed`. A confirmed code alone does not verify the record.

## Verification rules

- A name/time pair must come from one localized completion banner or one spatially reconstructed leaderboard entry. Workshop layouts with a time directly below its player name are supported. Fragments from different OCR passes never form a name/time pair.
- Without submitted aliases, a named completion banner or matching HUD identity can identify the player. A unique named row below an observed TOP 5 header can also support the submitted time. This reports the screenshot's row owner; it does not prove account ownership. Multiple players at the same time remain ambiguous.
- Live top-left timers, checkpoint splits, and a name appearing elsewhere do not prove completion. A matching completion banner takes priority over older leaderboard entries. Conflicting readings for the same identity/source remain unconfirmed.
- Time comparison uses exact integer centiseconds and an absolute tolerance, default `0`. Range validation defaults to 10–15360 seconds. A mismatching submitted time returns `null`; hints never supply missing OCR evidence.
- Workshop codes are literal 4–6 alphanumeric characters, including all-letter codes. Verification resolves targeted field readings before consulting the submitted code. A truncated prefix cannot override an independently read complete field; conflicting token readings remain unconfirmed. Localized label geometry can establish the field's location for another pass.
- `O/0`, `I/1`, and different final characters are distinct. The tested HUD font produces confident `O` readings for real zeros across multiple models. An isolated code-token reading containing `O` therefore keeps the code unconfirmed (`code_glyph_ambiguous`), including genuine letter-O codes; it is never converted to zero. Without token readings, the same guard applies to focused field readings. Codes whose isolated readings consistently show `0` can still be confirmed despite a whole-line misreading. This conservative limitation needs better glyph evidence before it can be relaxed.
- Name comparison normalizes Unicode compatibility forms, case, spacing, and Latin diacritics separately from display. Japanese voicing marks and identity punctuation remain significant; unsupported symbols are rejected rather than erased. Rank removal requires an explicit leaderboard role and separator. A single familiar Latin digit/letter confusion in a name of at least five characters may match. Short CJK names and arbitrary character edits are not fuzzy-matched.
- Short Hangul names with an ambiguous `ㅏ/ㅑ` reading require an independent name-cell read before confirmation. A mobile banner or HUD reading alone cannot verify them, even when it matches the submitted alias. Unresolved cases return `name_glyph_ambiguous`; this also affects genuine short `ㅏ` names when an independent cell reading is unavailable.

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

The process warms English, Korean, Japanese, and Chinese mobile models at startup. A high-capacity CJK recognizer is loaded lazily for small, ambiguous Japanese/Chinese name cells. A bounded Tesseract Korean fallback reads the same actual name cell with an independent model when needed; it never receives a supplied alias. Docker includes Tesseract and a checksum-pinned official Korean model. Local Python installations need `tesseract` on PATH and `kor.traineddata` under `~/.paddleocr/whl/rec/tesseract` to enable that fallback; otherwise it abstains.

Existing Docker model directories under `~/.paddleocr/whl` are supported; absent Paddle directories trigger its normal model download. `PADDLE_PDX_CACHE_HOME` can relocate downloaded Paddle models. OCR calls, including targeted fallbacks, stay serialized through one FIFO worker per process. A timed-out request retains the inference slot until native OCR finishes. More workers duplicate model memory and inference capacity; start with one.

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

## Validation

Validation results and remaining recognition limitations are recorded in the pull request. Test fixtures, test code and investigation reports are retained locally and are not distributed with the service.

## Layout

- `main.py`: API, queue, preprocessing, inference adapters, orchestration, and compatible text helpers.
- `evidence.py`: model-independent Unicode/code/time parsing, structured detections, spatial grouping, and relationship validation.
- `fallback_ocr.py`: optional, bounded Korean name-cell recognition without name hints.

MIT. Built with [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [Tesseract](https://github.com/tesseract-ocr/tesseract), FastAPI, OpenCV, NumPy, and Pillow.
