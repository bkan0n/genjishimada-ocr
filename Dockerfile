FROM debian:bookworm-slim AS models
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates curl tar && \
    rm -rf /var/lib/apt/lists/*

ENV PWHL=/root/.paddleocr/whl

RUN set -eux; \
    mkdir -p \
    "$PWHL/det/en" \
    "$PWHL/det/ml" \
    "$PWHL/det/ch" \
    "$PWHL/rec/en" \
    "$PWHL/rec/korean" \
    "$PWHL/rec/japan" \
    "$PWHL/rec/ch" \
    "$PWHL/cls"

RUN set -eux; \
    fetch() { \
    url="$1"; dest="$2"; name="$3"; \
    echo "↓ $name -> $dest"; \
    curl --fail --show-error --location "$url" | tar -x -C "$dest"; \
    test -f "$dest"/*/inference.pdmodel; \
    test -f "$dest"/*/inference.pdiparams; \
    test -f "$dest"/*/inference.pdiparams.info; \
    }; \
    fetch "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar"              "$PWHL/det/en" "en det"; \
    fetch "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar" "$PWHL/det/ml" "ml det"; \
    fetch "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar"                "$PWHL/det/ch" "ch det"; \
    fetch "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar"               "$PWHL/rec/en" "en rec"; \
    fetch "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer.tar"       "$PWHL/rec/korean" "korean rec"; \
    fetch "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_infer.tar"        "$PWHL/rec/japan" "japan rec"; \
    fetch "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar"                "$PWHL/rec/ch" "ch rec"; \
    fetch "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar"         "$PWHL/cls" "cls"

FROM python:3.10-slim
WORKDIR /app

ENV FLAGS_use_mkldnn=1 \
    ONEDNN_MAX_CPU_ISA=AVX2 \
    DNNL_MAX_CPU_ISA=AVX2 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HOME=/root

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 libstdc++6 curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=models /root/.paddleocr /root/.paddleocr

COPY main.py .

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
