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
    echo "-> $name -> $dest"; \
    curl --fail --show-error --location "$url" | tar -x -C "$dest"; \
    test -f "$dest"/*/inference.pdiparams; \
    test -f "$dest"/*/inference.pdmodel || test -f "$dest"/*/inference.json || test -f "$dest"/*/inference.yml; \
    }; \
    fetch "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar" "$PWHL/det/ppocrv5" "v5 mobile det"; \
    fetch "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar" "$PWHL/rec/ppocrv5" "v5 mobile rec"; \
    fetch "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv5_mobile_rec_infer.tar" "$PWHL/rec/en" "v5 mobile rec (en)"; \
    fetch "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/korean_PP-OCRv5_mobile_rec_infer.tar" "$PWHL/rec/korean" "v5 mobile rec (korean)"

FROM python:3.11-slim
WORKDIR /app

ENV FLAGS_use_mkldnn=0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HOME=/root

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 libstdc++6 curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=models /root/.paddleocr /root/.paddleocr

COPY main.py .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
