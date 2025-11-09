# GenjiPK OCR (FastAPI + PaddleOCR)

FastAPI microservice that extracts **player name**, **run time (seconds)**, and **map code** from Overwatch parkour screenshots using PaddleOCR (PP-OCRv3) with CPU-safe defaults.
Responses use **camelCase** via Pydantic aliases; internal code stays **snake_case**.

## Features

* 🔤 Multi-language OCR (en, ch, korean, japan) with prewarmed models
* 🧠 Script-aware name selection (Hangul/Kana/Han/Latin weighting)
* ⏱️ Robust time parsing with OCR noise normalization
* 🗺️ Multi-heuristic map code extraction (strict → loose fallback)
* 🧪 Deterministic CPU runtime (MKL/oneDNN off; thread caps)
* 🐍 Clean Pydantic models with camelCase JSON output

---

## API

### `GET /ping`

Health + warmed models.

**Response**

```json
{
  "ok": true,
  "models": ["en", "ch", "korean", "japan"]
}
```

### `POST /extract`

Accepts a base64 image, returns extracted fields + raw region texts.

**Request**

```json
{
  "image_b64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

**Response**

```json
{
  "extracted": {
    "name": "DAHMX",
    "time": 123.45,
    "code": "A1B2C3",
    "texts": {
      "topLeft": "MAP CODE: A1B2C3",
      "topLeftWhite": "MAP CODE",
      "topLeftCyan": "A1B2C3",
      "banner": "MISSION COMPLETE TIME 123.45 SEC",
      "topRight": "123.45 SEC",
      "bottomLeft": "250 DAHMX"
    }
  }
}
```

> JSON is camelCase thanks to `alias_generator=to_camel` and `populate_by_name=True`.

Open interactive docs at **`/docs`** and **`/redoc`**.

---

## Quickstart

### Option A: Docker (recommended)

```bash
# 1) Build (multi-stage pulls PP-OCRv3 weights)
docker build -t genjishimada-ocr:latest .

# 2) Run
docker run --rm -p 8000:8000 genjishimada-ocr:latest

# 3) Test
curl http://localhost:8000/ping
```

### Option B: Local Python

> Python 3.10; Linux is recommended for PaddlePaddle CPU wheels.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

If you see CPU ISA or MKL errors, see **Troubleshooting** below.

---

## Example Requests

**cURL**

```bash
# Base64-encode an image and POST
IMG_B64=$(base64 -w0 sample.png)
curl -sX POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d "{\"image_b64\":\"data:image/png;base64,${IMG_B64}\"}" | jq .
```

**Python**

```python
import base64, requests
b64 = "data:image/png;base64," + base64.b64encode(open("sample.png","rb").read()).decode()
r = requests.post("http://localhost:8000/extract", json={"image_b64": b64})
print(r.json())
```

---

## How It Works

1. **Decoding & ROI**
   The service decodes the base64 image and crops regions of interest (top-left, banner, top-right, bottom-left).

2. **Preprocessing**

   * Contrast enhancement (CLAHE)
   * White/cyan masks for high-salience overlays
   * Adaptive thresholding variant for the banner

3. **OCR (PP-OCRv3)**
   Engines are prewarmed per language and reused. `cls` is disabled for speed.

4. **Parsing**

   * **Time:** lexes near “TIME/SEC”, corrects OCR look-alike digits, scores proximity.
   * **Name:** ASCII bottom-left heuristics first; otherwise script-aware CJK candidate selection with ROI weights.
   * **Map Code:** strict “MAP CODE: XXXX” → short window after “MAP” → global 4–6 token scan with validation.

5. **Response Models**
   Pydantic models output **camelCase**; internal attributes remain snake_case.

---

## Configuration & Performance

* **CPU flags** (already set):
  `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `FLAGS_use_mkldnn=0`
  (Docker runtime uses `FLAGS_use_mkldnn=1` with AVX2 hints; OpenBLAS/OMP still capped.)
* **Model cache path:** `/root/.paddleocr/whl` (prepopulated in Docker stage).
* **Workers:** `--workers 4` is a good default; tune for your CPU.
* **Memory:** PP-OCRv3 is modest on CPU; each worker loads its own models.

---

## Project Layout

```
.
├─ main.py                 # FastAPI app, OCR logic, parsing heuristics
├─ Dockerfile              # Multi-stage build with pre-fetched PP-OCRv3 models
├─ requirements.txt        # Runtime deps (pinned)
├─ pyproject.toml          # Packaging + Ruff config
├─ pyrightconfig.json      # Type checking (basic)
└─ README.md
```

---

## Development

### Lint & Type Check

```bash
# Ruff (configured in pyproject)
ruff check .

# Pyright (basic mode)
pyright
```

---

## Troubleshooting

* **`paddlepaddle` wheel/CPU ISA issues (AVX/AVX2):**

  * Prefer the Docker image (pre-tested).
  * On bare metal, ensure your CPU supports the wheel you install; otherwise fall back to compatible wheels or use Docker.

* **OpenCV GUI errors:**
  Using `opencv-python-headless` avoids GUI deps. If you install full OpenCV, add `libgl1`/`libglib2.0-0`.

* **MKL/oneDNN perf variance:**
  If you want absolute determinism, keep `FLAGS_use_mkldnn=0`. If you want extra speed and your CPU supports it, leave the Docker defaults (AVX2 hints) and benchmark.

* **`503 OCR models not ready yet`:**
  Ensure the app started completely (the lifespan manager warms models at startup). Check `/ping`.

---

## Extending

* Add new ROIs or languages: update the `SUPPORTED_LANGUAGES`, `_v3_dirs_for_language_code`, and Docker model fetch stage.
* Add more HUD heuristics: create dedicated `extract_*` helpers and unit tests per heuristic.

---

## License

MIT

---

## Acknowledgements

* [PaddleOCR / PP-OCRv3](https://github.com/PaddlePaddle/PaddleOCR)
* FastAPI / Pydantic / OpenCV / NumPy / Pillow
