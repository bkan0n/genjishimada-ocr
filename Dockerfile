FROM debian:bookworm-slim AS models
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates curl tar && \
    rm -rf /var/lib/apt/lists/*

ENV PWHL=/root/.paddleocr/whl

RUN set -eux; \
    mkdir -p \
    "$PWHL/det/ppocrv5" \
    "$PWHL/rec/ppocrv5" \
    "$PWHL/rec/en" \
    "$PWHL/rec/korean"

RUN set -eux; \
    fetch() { \
    url="$1"; dest="$2"; name="$3"; \
    archive="${url##*/}"; model_dir="$dest/${archive%.tar}"; \
    echo "-> $name -> $dest"; \
    curl --fail --show-error --location "$url" | tar -x -C "$dest"; \
    test -f "$model_dir/inference.pdiparams"; \
    test -f "$model_dir/inference.pdmodel" || test -f "$model_dir/inference.json" || test -f "$model_dir/inference.yml"; \
    }; \
    fetch "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar" "$PWHL/det/ppocrv5" "v5 mobile det"; \
    fetch "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar" "$PWHL/rec/ppocrv5" "v5 mobile rec"; \
    fetch "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv5_mobile_rec_infer.tar" "$PWHL/rec/en" "v5 mobile rec (en)"; \
    fetch "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/korean_PP-OCRv5_mobile_rec_infer.tar" "$PWHL/rec/korean" "v5 mobile rec (korean)"; \
    fetch "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar" "$PWHL/rec/ppocrv5" "targeted CJK name recognizer"

RUN set -eux; \
    mkdir -p "$PWHL/rec/tesseract"; \
    curl --fail --show-error --location \
    "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/kor.traineddata" \
    -o "$PWHL/rec/tesseract/kor.traineddata"; \
    echo "f888d4038348a0c3d25151e7f452bda0d74ca275b18cab146798bcbb94084fff  $PWHL/rec/tesseract/kor.traineddata" | sha256sum -c -

FROM python:3.11-slim
WORKDIR /app

ENV OCR_MKLDNN=1 \
    OCR_CPU_THREADS=0 \
    OCR_MKLDNN_CACHE=20 \
    DISABLE_MODEL_SOURCE_CHECK=True \
    HOME=/root

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 libstdc++6 curl tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=models /root/.paddleocr /root/.paddleocr

COPY main.py evidence.py fallback_ocr.py .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
