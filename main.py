from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import re
import unicodedata
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import AsyncIterator, Literal

import aiohttp
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from paddleocr import PaddleOCR
from PIL import Image, ImageFile
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

# =============================================================================
# Runtime / CPU safety
# IMPORTANT: these env vars MUST be set before importing paddle/paddleocr.
# =============================================================================


def _available_cores() -> int:
    """
    Compute the number of CPU cores available to this process.

    Purpose:
      - Respect container CPU limits and cpuset/affinity when available.
      - Provide a safe fallback for environments where affinity is unavailable.

    Returns:
      - An integer >= 1 representing the usable CPU core count.

    Notes:
      - Prefers `os.sched_getaffinity(0)` (Linux) because it reflects cgroup/cpuset limits.
      - Falls back to `os.cpu_count()` if affinity is not supported or fails.
    """
    # Respect container CPU limits / cpuset when possible
    try:
        if hasattr(os, "sched_getaffinity"):
            return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    return max(1, (os.cpu_count() or 1))


def _env_bool(name: str, default: bool = False) -> bool:
    """
    Read a boolean-like environment variable.

    Purpose:
      - Convert common truthy strings into a Python boolean.
      - Provide a consistent toggle mechanism for runtime flags.

    Args:
      name:
        - Environment variable name to read.
      default:
        - Value returned if the variable is not set.

    Returns:
      - True if the env value is one of: "1", "true", "yes", "on" (case-insensitive).
      - Otherwise False, or `default` if not set.

    Notes:
      - Whitespace is stripped before comparison.
    """
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


AVAILABLE_CORES = _available_cores()

# MKLDNN is ON by default; disable with OCR_MKLDNN=0
OCR_MKLDNN: bool = _env_bool("OCR_MKLDNN", default=True)

# Threads: if OCR_CPU_THREADS is unset/0, auto-pick a safe default.
try:
    _t = int(os.environ.get("OCR_CPU_THREADS", "0") or "0")
except Exception:
    _t = 0

# Auto: use up to 4 threads (or less if CPU-limited)
OCR_CPU_THREADS: int = _t if _t > 0 else max(1, min(4, AVAILABLE_CORES))

# Perf/behavior toggles
OCR_EARLY_EXIT = _env_bool("OCR_EARLY_EXIT", default=True)  # stop once we have everything
OCR_DEBUG_TEXTS = _env_bool("OCR_DEBUG_TEXTS", default=False)  # expensive debug OCR strings
ACCURATE_OCR = _env_bool("ACCURATE_OCR", default=False)  # if True, run banner/top5 even if TL time found
OCR_TRUST_HINTS = _env_bool("OCR_TRUST_HINTS", default=False)  # allow using hints as fallback if OCR fails

# Hint matching: prefer provided hint only when it matches OCR at this threshold.
HINT_MATCH_THRESHOLD = float(os.environ.get("HINT_MATCH_THRESHOLD", "0.90"))
HINT_TIME_ABS_TOL = float(os.environ.get("HINT_TIME_ABS_TOL", "0"))

# Avoid thread oversubscription (OpenCV/OpenBLAS) and configure OMP/oneDNN threads.
os.environ.setdefault("CPU_RUNTIME_CACHE_CAPACITY", "20")
os.environ.setdefault("OPENBLAS_CORETYPE", "NEHALEM")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", str(OCR_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(OCR_CPU_THREADS))
os.environ.setdefault("FLAGS_use_mkldnn", "1" if OCR_MKLDNN else "0")
os.environ.setdefault("OMP_PROC_BIND", "TRUE")
os.environ.setdefault("OMP_PLACES", "cores")

# Optional workaround (only if you explicitly enable it):
# Disable PIR API: set OCR_DISABLE_PIR=1 if you hit oneDNN/PIR regressions.
if _env_bool("OCR_DISABLE_PIR", default=False):
    os.environ.setdefault("FLAGS_enable_pir_api", "0")

ImageFile.LOAD_TRUNCATED_IMAGES = True

# OpenCV: avoid internal threading (we rely on OMP/oneDNN threads)
cv2.setNumThreads(0)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# Limit per-request OCR time and variant count
OCR_TIMEOUT_S = float(os.environ.get("OCR_TIMEOUT_S", "60"))
FAST_OCR = os.environ.get("FAST_OCR", "1") == "1"
MIN_NAME_LEN = int(os.environ.get("MIN_NAME_LEN", "3"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logger = logging.getLogger("genjipk-ocr")
logger.setLevel(LOG_LEVEL)

if not logger.handlers:
    h = logging.StreamHandler()
    h.setLevel(LOG_LEVEL)
    h.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s:%(name)s:%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(h)

logger.propagate = False

_KERNEL_3 = np.ones((3, 3), np.uint8)
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

PADDLE_WHL_DIR = Path.home() / ".paddleocr" / "whl"

# -----------------------------------------------------------------------------
# Request queue (serialize Paddle/oneDNN inference)
# -----------------------------------------------------------------------------
OCR_QUEUE_MAXSIZE = int(os.environ.get("OCR_QUEUE_MAXSIZE", "32"))  # 0 = unlimited
OCR_QUEUE_WAIT_S = float(os.environ.get("OCR_QUEUE_WAIT_S", "180"))  # max wait in queue (sec)

# =============================================================================
# Types / Models
# =============================================================================
LanguageCode = Literal["en", "ch", "korean", "japan"]
RoiLabel = Literal["BL", "BAN", "TR", "TL"]


def to_camel(s: str) -> str:
    """
    Convert a snake_case identifier to camelCase.

    Purpose:
      - Provide a consistent JSON/API naming convention (camelCase) while keeping Python fields snake_case.
      - Used by Pydantic's alias generator for request/response models.

    Args:
      s:
        - Input string in snake_case (e.g., "top_left_white").

    Returns:
      - camelCase string (e.g., "topLeftWhite").

    Notes:
      - The first segment is kept lowercase; subsequent segments are title-cased.
      - Does not attempt to handle acronyms specially; it is a simple transformation.
    """
    # Split into words and keep the first segment lowercase.
    parts = s.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ScriptProfile(BaseModel):
    hangul: float
    kana: float
    han: float
    latin: float


class OcrCandidate(BaseModel):
    text: str
    confidence: float
    language_code: LanguageCode
    roi_label: RoiLabel
    profile: ScriptProfile


class ExtractedTexts(CamelModel):
    top_left: str
    top_left_white: str
    top_left_cyan: str
    banner: str
    top_right: str
    bottom_left: str


class ExtractedResult(CamelModel):
    name: str | None
    time: float | None
    code: str | None
    texts: ExtractedTexts


class ApiResponse(CamelModel):
    extracted: ExtractedResult


# =============================================================================
# ROI (fractional coords)
# =============================================================================
ROI_TOPLEFT = [0.010, 0.020, 0.360, 0.300]
ROI_TOPLEFT_WIDE = [0.005, 0.010, 0.420, 0.340]
ROI_BANNER_TIGHT = [0.240, 0.083, 0.760, 0.557]
ROI_TOPRIGHT = [0.821, 0.077, 0.985, 0.664]
ROI_BOTTOMLEFT = [0.050, 0.825, 0.330, 0.990]

# TOP5 strip inside ROI_TOPRIGHT (to avoid "HOLD ... LEADERBOARD" junk)
ROI_TR_TOP5_STRIP_0 = [0.02, 0.18, 1.00, 0.62]
ROI_TR_TOP5_STRIP_1 = [0.22, 0.50, 1.00, 0.78]
ROI_TR_TOP5_STRIP_2 = [0.16, 0.48, 1.00, 0.82]

# =============================================================================
# Regex / parsing
# =============================================================================
RE_SPACES = re.compile(r"\s+")
RE_DIGITS_LOOSE_CLEANUP1 = re.compile(r"[^\d\.,]")
RE_DIGITS_LOOSE_CLEANUP2 = re.compile(r"(\d{1,5}\.\d{2})")

RE_PARSE_TIME_AGAIN = re.compile(r"(?<![0-9.,])(\d{1,5}[.,]\d{2})\s*SEC", re.IGNORECASE)
RE_PARSE_TOPLEFT_TIME_ANY = re.compile(r"(?<![0-9.,])(\d{1,5}[.,]\d{2})\s*(?:SEC|초)?", re.IGNORECASE)

RE_PARSE_BANNER_TIME_SEARCH_WITH_SEC = re.compile(r"([0-9OQDBZGISL\,\.]{3,12})\s*(?:SEC|초)?")
RE_PARSE_BANNER_TIME_SEARCH_NO_SEC = re.compile(r"([0-9OQDBZGISL\,\.]{3,12})")
RE_PARSE_BANNER_TIME_SEARCH_ONLY_SEC = re.compile(r"(SEC|초)")
RE_CLEAN_BANNER_FRAGMENT = re.compile(r"\s{2,}")

RE_TOP5_SECTION = re.compile(r"TOP\s*5", re.IGNORECASE)
RE_TOP5_TIME_FOR_NAME_ASCII = re.compile(
    r"\b([A-Z][A-Z0-9_]{2,24})\b\s+(\d{1,5}[.,]\d{2})\s*(?:SEC|초)?",
    re.IGNORECASE,
)

RE_CODE_KEYWORD_EXTRACT = re.compile(r"(?:MAP\s+)?C(?:O|0)?DE\s*[:\-]?\s*([A-Z0-9]{4,6})\b", re.IGNORECASE)
RE_CODE_AFTER_COLON = re.compile(r":\s*([A-Z0-9]{4,6})\b", re.IGNORECASE)
RE_MAP_CODE_FIND = re.compile(r"\b[A-Z0-9]{4,6}\b")
RE_BASIC_NORMALIZATION = re.compile(r"[^A-Z0-9]")
RE_MAP_CODE_NORMALIZATION = re.compile(r"MAP\s*[CLO0][O0D]{2}E", flags=re.IGNORECASE)

RE_ASCII_NAME_MATCH = re.compile(r"[A-Z][A-Z0-9_]{2,23}")

# CJK ranges
_HANGUL = r"\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F"
_HIRAKATA = r"\u3040-\u30FF\u31F0-\u31FF\uFF66-\uFF9F"
_HAN = r"\u3400-\u4DBF\u4E00-\u9FFF"
_LATIN = r"A-Za-z"
_CJK_ALL = f"{_HANGUL}{_HIRAKATA}{_HAN}"

RE_CJK_CHAR = re.compile(f"[{_CJK_ALL}]")
RE_CJK_SEQ = re.compile(rf"([{_CJK_ALL}]{{2,40}})")
RE_TOP5_TIME_FOR_NAME_CJK = re.compile(
    rf"([{_CJK_ALL}]{{2,40}})\s+(\d{{1,5}}[.,]\d{{2}})\s*(?:SEC|초)?",
    re.IGNORECASE,
)

# Banner name patterns
RE_BANNER_NAME_ASCII = re.compile(
    r"\b([A-Z][A-Z0-9_]{2,24})\b[\s:!|~.,*_-]*MISSION[\s:!|~.,*_-]*COMPLETE",
    re.IGNORECASE,
)
RE_BANNER_NAME_CJK = re.compile(
    rf"([{_CJK_ALL}]{{2,40}})[\s:!|~.,*_-]*MISSION[\s:!|~.,*_-]*COMPLETE",
    re.IGNORECASE,
)

_GENERIC_ASCII = {
    "MISSION",
    "COMPLETE",
    "TIME",
    "CLEAR",
    "PLAYER",
    "SPLIT",
    "LEVEL",
    "EASY",
    "EXTREME",
    "HARD",
    "MEDIUM",
    "NORMAL",
    "TOP",
    "SEC",
    "HOLD",
    "RESTART",
    "LEADERBOARD",
    "TOGGLE",
    "INVISIBLE",
    "INVINCIBLE",
    "PRACTICE",
    "SHOW",
    "PREVIEW",
    "CHECKPOINT",
}

_ROMAN_PREFIXES = ("VIII", "VII", "VI", "IV", "IX", "V", "III", "II", "I", "X")

# Hangul syllable helpers (U+AC00..U+D7A3).
_HANGUL_BASE = 0xAC00
_HANGUL_END = 0xD7A3
_HANGUL_N_CHO = 19
_HANGUL_N_JUNG = 21
_HANGUL_N_JONG = 28
_HANGUL_CHO = [
    "ㄱ",
    "ㄲ",
    "ㄴ",
    "ㄷ",
    "ㄸ",
    "ㄹ",
    "ㅁ",
    "ㅂ",
    "ㅃ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅉ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
]
_HANGUL_TENSE_MAP = {
    0: 1,  # ㄱ -> ㄲ
    1: 0,  # ㄲ -> ㄱ
    3: 4,  # ㄷ -> ㄸ
    4: 3,  # ㄸ -> ㄷ
    7: 8,  # ㅂ -> ㅃ
    8: 7,  # ㅃ -> ㅂ
    9: 10,  # ㅅ -> ㅆ
    10: 9,  # ㅆ -> ㅅ
    12: 13,  # ㅈ -> ㅉ
    13: 12,  # ㅉ -> ㅈ
}

# =============================================================================
# PaddleOCR engine registry
# =============================================================================
OCR_ENGINES: dict[LanguageCode, PaddleOCR] = {}
OCR_ENGINES_NO_MKLDNN: dict[LanguageCode, PaddleOCR] = {}
SUPPORTED_LANGUAGES: tuple[LanguageCode, ...] = ("en", "ch", "korean", "japan")

# Backwards-compat mapping used by older code paths
LANG_MAP: dict[str, str] = {
    "en": "en",
    "ch": "ch",
    "japan": "japan",
    "korean": "korean",
}


def _pick_existing_dir(*candidates: Path) -> str | None:
    """
    Pick the first existing directory from a list of candidates.

    Purpose:
      - Resolve PaddleOCR model directories across multiple possible locations.
      - Keep priority order deterministic.

    Args:
      *candidates:
        - One or more Path objects, checked in the order given.

    Returns:
      - The first existing path as a string, or None if none exist.

    Notes:
      - Uses `Path.exists()` (works for both files and directories, but candidates are expected to be dirs).
      - Priority is preserved by iterating in-order.
    """
    # Preserve priority by checking in order.
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return None


def _model_dirs_for_language_code(
    language_code: LanguageCode,
) -> tuple[str | None, str | None, str, str | None, str | None]:
    """
    Resolve detection/recognition model directories and names for a language.

    Purpose:
      - Centralize model path selection per language.
      - Prefer PP-OCRv5 "mobile" models to reduce size and cold-start costs.

    Args:
      language_code:
        - One of the supported PaddleOCR language identifiers.

    Returns:
      - (det_dir, rec_dir, ocr_version, det_name, rec_name)
        - det_dir / rec_dir: filesystem paths as strings (or None if missing)
        - ocr_version: OCR version string passed to PaddleOCR
        - det_name / rec_name: model name identifiers (or None if missing)

    Notes:
      - English and Korean use per-language recognition models when available.
      - Chinese and Japanese use the shared mobile recognition model if present.
      - Logs a single summary line per language to aid debugging container mounts.
    """
    normalized_language = language_code.lower()
    ocr_version = "PP-OCRv5"

    v5_mobile_det = PADDLE_WHL_DIR / "det" / "ppocrv5" / "PP-OCRv5_mobile_det_infer"
    v5_mobile_rec = PADDLE_WHL_DIR / "rec" / "ppocrv5" / "PP-OCRv5_mobile_rec_infer"
    v5_en_rec = PADDLE_WHL_DIR / "rec" / "en" / "en_PP-OCRv5_mobile_rec_infer"
    v5_kr_rec = PADDLE_WHL_DIR / "rec" / "korean" / "korean_PP-OCRv5_mobile_rec_infer"

    det_dir = _pick_existing_dir(v5_mobile_det)
    det_name = "PP-OCRv5_mobile_det" if det_dir else None

    if normalized_language == "en":
        rec_dir = _pick_existing_dir(v5_en_rec)
        rec_name = "en_PP-OCRv5_mobile_rec" if rec_dir else None
    elif normalized_language == "korean":
        rec_dir = _pick_existing_dir(v5_kr_rec)
        rec_name = "korean_PP-OCRv5_mobile_rec" if rec_dir else None
    elif normalized_language in ("ch", "japan"):
        rec_dir = _pick_existing_dir(v5_mobile_rec)
        rec_name = "PP-OCRv5_mobile_rec" if rec_dir else None
    else:
        return None, None, ocr_version, None, None

    logger.info(
        f"[models] {language_code} ocr={ocr_version} det_dir={'OK' if det_dir else 'MISS'} "
        f"rec_dir={'OK' if rec_dir else 'MISS'} base={PADDLE_WHL_DIR}"
    )
    return det_dir, rec_dir, ocr_version, det_name, rec_name


def _build_ocr_engine(
    language_code: LanguageCode,
    *,
    enable_mkldnn: bool = OCR_MKLDNN,
    cpu_threads: int = OCR_CPU_THREADS,
) -> PaddleOCR:
    """
    Construct a PaddleOCR engine configured for this service.

    Purpose:
      - Create a CPU-only PaddleOCR instance with consistent thresholds and model paths.
      - Support toggling MKLDNN and CPU thread count for stability/performance.

    Args:
      language_code:
        - PaddleOCR language code used for recognition.
      enable_mkldnn:
        - Whether to enable oneDNN/MKLDNN acceleration (may be unstable on some CPUs/builds).
      cpu_threads:
        - Thread count passed to PaddleOCR inference engine.

    Returns:
      - A configured `PaddleOCR` instance.

    Notes:
      - Model directories are resolved before instantiation; missing dirs may still allow fallback behavior
        depending on PaddleOCR internals, but this service expects models to be present.
      - Detection/recognition thresholds are tuned for HUD text (small/low-contrast) rather than documents.
    """
    # Resolve model directories before instantiation.
    det_dir, rec_dir, ocr_version, det_name, rec_name = _model_dirs_for_language_code(language_code)
    return PaddleOCR(
        ocr_version=ocr_version,
        lang=language_code,
        device="cpu",
        use_textline_orientation=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        enable_mkldnn=enable_mkldnn,
        cpu_threads=cpu_threads,
        mkldnn_cache_capacity=int(os.environ.get("OCR_MKLDNN_CACHE", "20")),
        text_recognition_batch_size=1,
        textline_orientation_batch_size=1,
        text_det_limit_side_len=960,
        text_det_limit_type="max",
        text_det_box_thresh=0.3,
        text_rec_score_thresh=0.0,
        text_detection_model_name=det_name,
        text_detection_model_dir=det_dir,
        text_recognition_model_name=rec_name,
        text_recognition_model_dir=rec_dir,
    )


def warm_ocr_engines(languages: tuple[LanguageCode, ...] = SUPPORTED_LANGUAGES) -> None:
    """
    Preload PaddleOCR models into memory.

    Purpose:
      - Avoid cold-start latency on the first request.
      - Ensure model initialization failures appear at startup rather than mid-request.

    Args:
      languages:
        - Languages to load and register in `OCR_ENGINES`.

    Returns:
      - None.

    Notes:
      - Idempotent: skips languages already present in `OCR_ENGINES`.
      - Uses the global MKLDNN/thread settings at load time.
    """
    # Avoid reloading models that are already initialized.
    for lang in languages:
        if lang in OCR_ENGINES:
            continue
        logger.info(f"📥 Warming PaddleOCR model: {lang} (MKLDNN {'ON' if OCR_MKLDNN else 'OFF'}, threads={OCR_CPU_THREADS})")
        OCR_ENGINES[lang] = _build_ocr_engine(lang)
        logger.info(f"✅ Model ready: {lang}")


def get_ocr_engine(language_code: LanguageCode) -> PaddleOCR:
    """
    Retrieve a warmed PaddleOCR engine for a given language.

    Purpose:
      - Provide a single access point for OCR engines.
      - Fail fast with an HTTP-friendly error if models are not ready.

    Args:
      language_code:
        - Requested language engine key.

    Returns:
      - The warmed `PaddleOCR` instance.

    Raises:
      - HTTPException(503) if the engine is not loaded.

    Notes:
      - Call `warm_ocr_engines()` during startup to populate the registry.
    """
    # Fail fast if models are not ready yet.
    engine = OCR_ENGINES.get(language_code)
    if engine is None:
        raise HTTPException(status_code=503, detail=f"OCR model '{language_code}' not loaded")
    return engine


def log_model_dirs() -> None:
    """
    Log resolved model directories for each supported language.

    Purpose:
      - Make model resolution visible in container logs.
      - Simplify debugging when model files are missing or mounted incorrectly.

    Returns:
      - None.

    Notes:
      - Calls `_model_dirs_for_language_code()` for a fixed set of languages.
      - Intended to run at startup.
    """
    # Emit per-language model paths for troubleshooting.
    for lang in ("en", "korean", "japan", "ch"):
        det_dir, rec_dir, ocr_version, det_name, rec_name = _model_dirs_for_language_code(lang)  # type: ignore[arg-type]
        logger.info(f"[models] {lang}: ocr={ocr_version} det={det_dir} rec={rec_dir} det_name={det_name} rec_name={rec_name}")


# =============================================================================
# Core image utilities
# =============================================================================
def normalize_base64_padding(b64_string: str) -> str:
    """
    Normalize a base64 string and ensure proper '=' padding.

    Purpose:
      - Accept base64 from multiple sources (URL-safe variants, whitespace, missing padding).
      - Produce a decodable base64 payload for downstream decoding.

    Args:
      b64_string:
        - Raw base64 content (may contain whitespace or URL-safe characters).

    Returns:
      - A cleaned base64 string padded to a multiple-of-4 length.

    Notes:
      - Converts '-' -> '+', '_' -> '/'.
      - Removes whitespace and normalizes spaces to '+' (common form encoding).
    """
    # Normalize whitespace and URL-safe characters first.
    cleaned = re.sub(RE_SPACES, "", b64_string).replace("-", "+").replace("_", "/").replace(" ", "+")
    missing = (-len(cleaned)) % 4
    return cleaned + ("=" * missing if missing else "")


def decode_base64_image(image_b64: str) -> np.ndarray:
    """
    Decode a base64 image (optionally a data URL) into an OpenCV BGR image.

    Purpose:
      - Support API clients that upload images as base64 strings.
      - Convert PIL image bytes into an OpenCV-compatible ndarray.

    Args:
      image_b64:
        - Base64 string or full data URL (e.g. "data:image/png;base64,...").

    Returns:
      - OpenCV image in BGR color space (np.ndarray).

    Raises:
      - HTTPException(400) for missing/invalid base64 or invalid image streams.

    Notes:
      - Uses PIL for robust decoding of various formats, then converts RGB->BGR.
      - `validate=False` is used to be tolerant of minor base64 irregularities.
    """
    # Validate and strip data URL header if present.
    if not image_b64:
        raise HTTPException(status_code=400, detail="image_b64 is required")
    if image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    image_b64 = normalize_base64_padding(image_b64)
    try:
        image_bytes = base64.b64decode(image_b64, validate=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid base64: {e}")
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.load()
        pil_image = pil_image.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image stream: {e}")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw image bytes into an OpenCV BGR image.

    Purpose:
      - Decode fetched image content (HTTP response bytes) quickly via OpenCV.

    Args:
      image_bytes:
        - Raw bytes representing an encoded image (PNG/JPG/WebP, etc.).

    Returns:
      - OpenCV image in BGR color space (np.ndarray).

    Raises:
      - HTTPException(400) if bytes are empty or cannot be decoded.

    Notes:
      - OpenCV is faster than PIL here but less forgiving on malformed streams.
    """
    # OpenCV decoding is faster but expects valid image bytes.
    if not image_bytes:
        raise HTTPException(status_code=400, detail="invalid image stream: empty body")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="invalid image stream: cv2.imdecode failed")
    return img


def crop_by_frac_roi(image: np.ndarray, roi_frac: list[float]) -> np.ndarray:
    """
    Crop an image using fractional ROI coordinates.

    Purpose:
      - Define ROIs in a resolution-independent way (fractions of width/height).
      - Extract HUD blocks reliably across different screenshot sizes.

    Args:
      image:
        - Source image (BGR or grayscale).
      roi_frac:
        - Fractional ROI as [x1, y1, x2, y2] in range 0..1.

    Returns:
      - A copied crop (np.ndarray) of the ROI region.

    Notes:
      - Coordinates are clamped to image bounds.
      - Returns a `.copy()` to avoid referencing the original buffer.
    """
    # Convert fractional ROI to pixel coordinates.
    h, w = image.shape[:2]
    x1 = int(w * roi_frac[0])
    y1 = int(h * roi_frac[1])
    x2 = int(w * roi_frac[2])
    y2 = int(h * roi_frac[3])
    return image[max(y1, 0) : min(y2, h), max(x1, 0) : min(x2, w)].copy()


def crop_within(parent_crop: np.ndarray, rel_roi: list[float]) -> np.ndarray:
    """
    Crop a sub-ROI inside an already-cropped image.

    Purpose:
      - Allow nested ROI definitions (e.g., TOP5 strip inside TOPRIGHT ROI).
      - Keep ROI definitions clean and composable.

    Args:
      parent_crop:
        - Image that represents the parent ROI.
      rel_roi:
        - Fractional ROI relative to `parent_crop` coordinates.

    Returns:
      - Cropped sub-image (np.ndarray).

    Notes:
      - Thin wrapper around `crop_by_frac_roi()` for readability.
    """
    # Delegate to the generic fractional cropper.
    return crop_by_frac_roi(parent_crop, rel_roi)


# =============================================================================
# Pre-processing
# =============================================================================
def enhance_contrast_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR -> grayscale and enhance local contrast for OCR.

    Purpose:
      - Improve text readability in low-contrast HUD regions.
      - Reduce noise while preserving edges.

    Args:
      image_bgr:
        - Input BGR image.

    Returns:
      - Preprocessed grayscale image (np.ndarray).

    Notes:
      - Applies CLAHE (adaptive histogram equalization) + mild Gaussian blur.
    """
    g = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    g = _CLAHE.apply(g)
    return cv2.GaussianBlur(g, (3, 3), 0)


def mask_white_regions(image_bgr: np.ndarray) -> np.ndarray:
    """
    Build a binary mask for bright white HUD text/elements.

    Purpose:
      - Isolate white text which often OCRs better on a binary mask.
      - Reduce background clutter in HUD regions.

    Args:
      image_bgr:
        - Input BGR image.

    Returns:
      - Single-channel mask image (uint8, 0 or 255).

    Notes:
      - Uses HSV thresholding tuned for bright/low-saturation whites.
      - Applies median blur + morphological close to fill small gaps.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 190], np.uint8), np.array([179, 70, 255], np.uint8))
    mask = cv2.medianBlur(mask, 3)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL_3, 1)  # type: ignore


def mask_cyan_regions(image_bgr: np.ndarray) -> np.ndarray:
    """
    Build a binary mask for saturated cyan UI accents.

    Purpose:
      - Extract cyan-colored HUD text/edges which appear in some themes.
      - Provide an alternate OCR input when white masks fail.

    Args:
      image_bgr:
        - Input BGR image.

    Returns:
      - Single-channel mask image (uint8, 0 or 255).

    Notes:
      - HSV bounds are tuned for saturated cyan (not pale HUD cyan).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([85, 35, 70], np.uint8), np.array([130, 255, 255], np.uint8))
    mask = cv2.medianBlur(mask, 3)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL_3, 1)  # type: ignore


def mask_hud_cyan_regions(image_bgr: np.ndarray) -> np.ndarray:
    """
    Build a binary mask for pale cyan HUD text.

    Purpose:
      - Target lighter cyan shades commonly used in Overwatch HUD typography.
      - Improve OCR on names/times rendered in cyan.

    Args:
      image_bgr:
        - Input BGR image.

    Returns:
      - Single-channel mask image (uint8, 0 or 255).

    Notes:
      - HSV bounds are wider and allow lower saturation than `mask_cyan_regions()`.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([80, 10, 110], np.uint8), np.array([135, 255, 255], np.uint8))
    mask = cv2.medianBlur(mask, 3)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL_3, 1)  # type: ignore


def unsharp(image_bgr: np.ndarray, amount: float = 1.6, sigma: float = 1.0) -> np.ndarray:
    """
    Apply an unsharp mask to emphasize edges.

    Purpose:
      - Sharpen HUD text strokes that are slightly blurred by compression.
      - Help OCR detect character boundaries.

    Args:
      image_bgr:
        - Input image (typically BGR).
      amount:
        - Sharpening strength (higher = more edge emphasis).
      sigma:
        - Gaussian blur sigma used to create the "unsharp" component.

    Returns:
      - Sharpened image (np.ndarray).

    Notes:
      - Uses linear blending of original and blurred image.
    """
    blur = cv2.GaussianBlur(image_bgr, (0, 0), sigma)
    return cv2.addWeighted(image_bgr, amount, blur, -(amount - 1.0), 0)


def upscale(image_bgr: np.ndarray, scale: float) -> np.ndarray:
    """
    Upscale an image using cubic interpolation.

    Purpose:
      - Increase effective character size for OCR on small ROIs.
      - Improve recognition on small fonts without changing aspect ratio.

    Args:
      image_bgr:
        - Input image.
      scale:
        - Scale factor (e.g., 2.0, 2.8).

    Returns:
      - Upscaled image.

    Notes:
      - Uses INTER_CUBIC for smoother edges.
    """
    return cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def build_map_code_variants(roi_bgr: np.ndarray) -> list[np.ndarray]:
    """
    Build focused OCR variants for top-left code extraction.

    Purpose:
      - Recover short map codes that get degraded inside the full top-left HUD block.
      - Emphasize thin trailing characters via stronger white-mask and threshold variants.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return []

    variants: list[np.ndarray] = []
    base = roi_bgr
    variants.append(base)

    h, w = base.shape[:2]
    scale = 3.6 if min(h, w) < 120 else 3.0
    up = unsharp(upscale(base, scale), amount=1.9, sigma=0.9)
    variants.append(up)

    white = mask_white_regions(base)
    variants.append(white)

    white_up = cv2.resize(white, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    white_up = cv2.dilate(white_up, _KERNEL_3, iterations=1)
    variants.append(white_up)

    gray = enhance_contrast_grayscale(base)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    variants.append(thr)
    variants.append(255 - thr)

    return variants


def build_cjk_variants(roi_bgr: np.ndarray) -> list[np.ndarray]:
    """
    Build a small set of preprocessing variants for CJK OCR.

    Purpose:
      - Provide multiple "views" of the same ROI to increase OCR robustness.
      - Include masks and contrast-enhanced variants that help on different HUD styles.

    Args:
      roi_bgr:
        - ROI image in BGR.

    Returns:
      - A list of images (BGR or single-channel masks) to feed into OCR.

    Notes:
      - Returns an empty list on empty ROI.
      - In FAST_OCR mode, the variant list is intentionally small to reduce CPU load.
      - Variants include: base, upscaled+unsharp, white mask, cyan mask, grayscale, etc.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return []

    variants: list[np.ndarray] = []
    base = roi_bgr
    variants.append(base)

    h, w = base.shape[:2]
    scale = 2.8 if min(h, w) < 160 else 2.0
    up = upscale(base, scale)
    up = unsharp(up, amount=1.7, sigma=1.0)
    variants.append(up)

    wmask = mask_white_regions(base)
    variants.append(wmask)

    cmask = mask_cyan_regions(base)
    variants.append(cmask)

    g = enhance_contrast_grayscale(base)
    if FAST_OCR:
        variants.append(g)
        return variants

    variants.append(255 - wmask)
    variants.append(255 - cmask)

    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
    variants.append(thr)
    variants.append(255 - thr)

    return variants


def build_cjk_name_variants(roi_bgr: np.ndarray) -> list[np.ndarray]:
    """
    Build extra preprocessing variants tuned for HUD player names (CJK).

    Purpose:
      - Expand `build_cjk_variants()` with name-specific transforms.
      - Capture thin strokes and pale cyan text used in name plates.

    Args:
      roi_bgr:
        - ROI image in BGR.

    Returns:
      - A list of variant images suitable for OCR.

    Notes:
      - Adds an aggressive stretch for very small ROIs (improves vertical stroke separation).
      - Adds HUD cyan mask + optional inverted/dilated forms (disabled in FAST_OCR).
    """
    variants = build_cjk_variants(roi_bgr)
    if roi_bgr is None or roi_bgr.size == 0:
        return variants

    h, w = roi_bgr.shape[:2]
    if min(h, w) < 180:
        stretch = cv2.resize(roi_bgr, None, fx=2.4, fy=3.2, interpolation=cv2.INTER_CUBIC)
        stretch = unsharp(stretch, amount=1.7, sigma=1.0)
        variants.append(stretch)

    hud = mask_hud_cyan_regions(roi_bgr)
    variants.append(hud)
    if FAST_OCR:
        return variants

    variants.append(255 - hud)
    variants.append(cv2.dilate(hud, _KERNEL_3, iterations=1))
    return variants


# =============================================================================
# OCR wrapper
# =============================================================================
def _looks_like_onednn_pir_bug(err: Exception) -> bool:
    """
    Heuristically detect oneDNN / PIR-related Paddle runtime failures.

    Purpose:
      - Identify a known class of intermittent MKLDNN/PIR errors.
      - Enable automatic fallback to a MKLDNN-OFF engine when detected.

    Args:
      err:
        - The exception raised by Paddle/PaddleOCR inference.

    Returns:
      - True if the error message contains signatures commonly associated with oneDNN/PIR crashes.

    Notes:
      - This is string-based detection; it is intentionally broad to catch multiple variants.
      - Used only as a guard for retry logic when MKLDNN is enabled.
    """
    s = str(err)
    return (
        "onednn_instruction" in s
        or "onednn_kernel" in s
        or "Tensor holds no memory" in s
        or "holder_ should not be null" in s
        or "ConvertPirAttribute2RuntimeAttribute" in s
        or "pir::ArrayAttribute" in s
        or "Unimplemented" in s
    )


def ocr_lines_cached(
    image: np.ndarray,
    language_code: LanguageCode,
    cache: dict[tuple[str, int, tuple[int, ...]], list[tuple[str, float]]],
) -> list[tuple[str, float]]:
    """
    OCR with a lightweight per-request cache.

    Purpose:
      - Avoid running OCR multiple times on the same ndarray variant within a single request.
      - Reduce CPU load when several parsing stages reuse the same crops/masks.

    Args:
      image:
        - The image/mask to OCR.
      language_code:
        - OCR language code.
      cache:
        - Dict keyed by (language, object-id, shape) storing OCR outputs.

    Returns:
      - List of (text, confidence) tuples.

    Notes:
      - Cache key uses `id(image)` and `image.shape`, assuming the same ndarray object is reused.
      - Cache lifetime is the request; callers pass a fresh dict per pipeline run.
    """
    """Run OCR on an image with caching."""
    if image is None or image.size == 0:
        return []
    key = (str(language_code), id(image), tuple(image.shape))
    hit = cache.get(key)
    if hit is not None:
        return hit
    out = ocr_lines(image, language_code)
    cache[key] = out
    return out


def ocr_lines(image: np.ndarray, language_code: LanguageCode) -> list[tuple[str, float]]:
    """
    Run OCR and normalize PaddleOCR outputs into (text, confidence) lines.

    Purpose:
      - Provide a stable OCR interface across PaddleOCR 2.x and 3.x output formats.
      - Implement automatic fallback when MKLDNN hits known oneDNN/PIR issues.

    Args:
      image:
        - Input image or mask (BGR or single-channel). Empty images return [].
      language_code:
        - OCR language to use for recognition.

    Returns:
      - A list of (text, confidence) tuples, best-effort.

    Notes:
      - If the engine exposes `.predict()`, it is used; otherwise `.ocr()`.
      - When MKLDNN is enabled and the error looks like a oneDNN/PIR crash,
        the function retries once using a MKLDNN-OFF engine cached in `OCR_ENGINES_NO_MKLDNN`.
      - Confidence values are cast to float; missing scores become 0.0.
    """
    # Guard against empty inputs to avoid OCR errors.
    if image is None or image.size == 0:
        return []
    engine = get_ocr_engine(language_code)
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image
    try:
        if hasattr(engine, "predict"):
            result = engine.predict(bgr) or []
        else:
            result = engine.ocr(bgr) or []
    except Exception as e:
        if OCR_MKLDNN and _looks_like_onednn_pir_bug(e):
            logger.warning(f"OCR({language_code}) failed with oneDNN/PIR error under MKLDNN; retrying with MKLDNN OFF: {e}")
            fb = OCR_ENGINES_NO_MKLDNN.get(language_code)
            if fb is None:
                fb = _build_ocr_engine(language_code, enable_mkldnn=False, cpu_threads=OCR_CPU_THREADS)
                OCR_ENGINES_NO_MKLDNN[language_code] = fb
            try:
                if hasattr(fb, "predict"):
                    result = fb.predict(bgr) or []
                else:
                    result = fb.ocr(bgr) or []
            except Exception as e2:
                logger.warning(f"OCR({language_code}) fallback (MKLDNN OFF) failed: {e2}")
                return []
        else:
            logger.warning(f"OCR({language_code}) failed: {e}")
            return []

    out: list[tuple[str, float]] = []

    def _add_text(text: str | None, score: float | None) -> None:
        """
        Internal helper: normalize and append a text line.

        Purpose:
          - Centralize cleaning and confidence parsing.
          - Skip empty/whitespace-only strings.

        Args:
          text:
            - Recognized text (may be None or empty).
          score:
            - Confidence score (may be None or non-float).

        Returns:
          - None (mutates the outer `out` list).
        """
        if not text:
            return
        cleaned = str(text).strip()
        if not cleaned:
            return
        try:
            conf = float(score) if score is not None else 0.0
        except Exception:
            conf = 0.0
        out.append((cleaned, conf))

    def _extract_from_dict(data: dict) -> None:
        """
        Internal helper: extract text/score fields from PaddleOCR 3.x dict-like outputs.

        Purpose:
          - Support multiple nested result layouts produced by PaddleOCR 3.x.
          - Walk through likely fields ("rec_texts", "rec_text", "ocr_res", etc.)

        Args:
          data:
            - A dict potentially containing OCR text/score fields (possibly nested).

        Returns:
          - None (mutates the outer `out` list).

        Notes:
          - This function is defensive: it checks types before iterating.
          - It recurses into known nesting keys where Paddle may embed results.
        """
        if not isinstance(data, dict):
            return

        rec_texts = data.get("rec_texts")
        rec_scores = data.get("rec_scores")
        if isinstance(rec_texts, list):
            if isinstance(rec_scores, list):
                for text, score in zip(rec_texts, rec_scores):
                    _add_text(text, score)
            else:
                for text in rec_texts:
                    _add_text(text, None)

        if isinstance(data.get("rec_text"), str) or data.get("rec_text") is not None:
            _add_text(data.get("rec_text"), data.get("rec_score"))

        if isinstance(data.get("text"), str) or data.get("text") is not None:
            _add_text(data.get("text"), data.get("score") or data.get("rec_score"))

        if isinstance(data.get("res"), dict):
            _extract_from_dict(data["res"])
        if isinstance(data.get("overall_ocr_res"), dict):
            _extract_from_dict(data["overall_ocr_res"])

        ocr_res = data.get("ocr_res")
        if isinstance(ocr_res, list):
            for item in ocr_res:
                if isinstance(item, dict):
                    _extract_from_dict(item)
        elif isinstance(ocr_res, dict):
            _extract_from_dict(ocr_res)

    # Legacy PaddleOCR 2.x output format (list of blocks).
    if isinstance(result, list) and result and isinstance(result[0], list):
        blocks = result[0] if (len(result) > 0 and isinstance(result[0], list)) else result
        for block in blocks or []:
            if not block or len(block) < 2:
                continue
            info = block[1]
            if not isinstance(info, (list, tuple)) or len(info) < 2:
                continue
            text = str(info[0] or "").strip()
            try:
                conf = float(info[1]) if info[1] is not None else 0.0
            except Exception:
                conf = 0.0
            if text:
                out.append((text, conf))
        return out

    # PaddleOCR 3.x output format (Result objects or dicts).
    if not isinstance(result, list) and hasattr(result, "__iter__"):
        try:
            result = list(result)
        except Exception:
            result = [result]
    items = result if isinstance(result, list) else [result]
    for item in items:
        data = None
        if hasattr(item, "json"):
            try:
                data = item.json() if callable(item.json) else item.json
            except Exception:
                data = None
        elif isinstance(item, dict):
            data = item

        if isinstance(data, dict):
            _extract_from_dict(data)

    return out


def join_lines(lines: list[tuple[str, float]]) -> str:
    """
    Join OCR line tuples into a single space-normalized string.

    Purpose:
      - Produce a single text blob for downstream regex parsing.
      - Drop confidence values while preserving original ordering.

    Args:
      lines:
        - List of (text, confidence) tuples.

    Returns:
      - A single string formed by joining texts with spaces and trimming ends.

    Notes:
      - Confidence is intentionally ignored here; selection is done elsewhere.
    """
    return " ".join([t for t, _ in lines]).strip()


# =============================================================================
# Script profiling + scoring
# =============================================================================
def remove_all_whitespace(text: str) -> str:
    """
    Remove all whitespace characters from a string.

    Purpose:
      - Normalize OCR outputs for script counting and similarity.
      - Avoid discrepancies caused by OCR inserting random spaces.

    Args:
      text:
        - Input string (may be empty/None-like).

    Returns:
      - String with all whitespace removed.
    """
    return re.sub(RE_SPACES, "", text or "")


def count_cjk(text: str) -> int:
    """
    Count the number of CJK (Hangul/Kana/Han) characters in text.

    Purpose:
      - Detect whether a string should follow the CJK matching path.
      - Avoid treating mixed strings as ASCII-only.

    Args:
      text:
        - Input string.

    Returns:
      - Number of matched CJK characters.
    """
    return len(RE_CJK_CHAR.findall(text or ""))


def fraction_of_unicode_class(unicode_class_pattern: str, text: str) -> float:
    """
    Compute the fraction of characters matching a given Unicode class.

    Purpose:
      - Build a script profile (Hangul/Kana/Han/Latin) for a string.
      - Compare expected script distributions by language.

    Args:
      unicode_class_pattern:
        - Character class range string (e.g., Hangul range).
      text:
        - Input text to analyze.

    Returns:
      - A float in [0, 1] representing the ratio among non-whitespace chars.

    Notes:
      - Whitespace is removed before computation.
      - Returns 0.0 for empty strings after compaction.
    """
    compact = remove_all_whitespace(text)
    return 0.0 if not compact else len(re.findall(f"[{unicode_class_pattern}]", compact)) / len(compact)


def build_script_profile(text: str) -> ScriptProfile:
    """
    Build a ScriptProfile with Hangul/Kana/Han/Latin ratios.

    Purpose:
      - Provide script-aware scoring and matching for multilingual OCR.
      - Used for candidate scoring and language selection heuristics.

    Args:
      text:
        - Input string.

    Returns:
      - ScriptProfile with per-script fractions.
    """
    return ScriptProfile(
        hangul=fraction_of_unicode_class(_HANGUL, text),
        kana=fraction_of_unicode_class(_HIRAKATA, text),
        han=fraction_of_unicode_class(_HAN, text),
        latin=fraction_of_unicode_class(_LATIN, text),
    )


def expected_script_for_language(language_code: str) -> str:
    """
    Map a language code to its expected dominant script.

    Purpose:
      - Provide a coarse prior for scoring OCR name candidates.

    Args:
      language_code:
        - Language identifier string.

    Returns:
      - One of: "hangul", "kana", "han", "latin".

    Notes:
      - Defaults to "latin" when language is unknown.
    """
    return {"korean": "hangul", "japan": "kana", "ch": "han", "en": "latin"}.get(language_code, "latin")


def roi_label_weight(roi: RoiLabel) -> float:
    """
    Assign a heuristic weight to an ROI label based on reliability.

    Purpose:
      - Bias candidate scoring toward ROIs that are more trustworthy for names.

    Args:
      roi:
        - ROI label ("BL", "TR", "BAN", "TL").

    Returns:
      - Weight value added to candidate scores.

    Notes:
      - BL is the most reliable for names; TL is least.
    """
    return {"BL": 0.35, "TR": 0.25, "BAN": 0.10, "TL": 0.05}.get(roi, 0.0)


def normalize_banner_fragment(fragment_text: str) -> str:
    """
    Normalize banner OCR fragments into a cleaner parseable string.

    Purpose:
      - Reduce OCR noise (extra spaces and punctuation) before regex parsing.
      - Improve banner name/time extraction stability.

    Args:
      fragment_text:
        - Raw OCR text assembled from banner variants.

    Returns:
      - Normalized string with collapsed spaces and trimmed punctuation.
    """
    return re.sub(RE_CLEAN_BANNER_FRAGMENT, " ", (fragment_text or "")).strip(" :|~!.,*_-").strip()


def _cjk_best_substring_min(text: str, min_len: int) -> str | None:
    """
    Extract the longest contiguous CJK substring meeting a minimum length.

    Purpose:
      - Pull the most likely name segment from noisy mixed OCR strings.
      - Avoid short accidental matches that are not real names.

    Args:
      text:
        - Input string possibly containing CJK content.
      min_len:
        - Minimum length required for a returned substring.

    Returns:
      - The longest CJK substring if >= min_len, else None.

    Notes:
      - Uses `RE_CJK_SEQ` to find contiguous runs.
    """
    if not text:
        return None
    best = ""
    for m in RE_CJK_SEQ.finditer(text):
        seq = m.group(1) or ""
        if len(seq) > len(best):
            best = seq
    return best if len(best) >= min_len else None


def _cjk_best_substring(text: str) -> str | None:
    """
    Extract the longest contiguous CJK substring (minimum length = 2).

    Purpose:
      - Convenience wrapper around `_cjk_best_substring_min()` for common use.

    Args:
      text:
        - Input string.

    Returns:
      - Longest CJK substring if length >= 2, else None.
    """
    return _cjk_best_substring_min(text, 2)


# =============================================================================
# Similarity helpers (hints: 90% matching)
# =============================================================================
def _levenshtein_distance(a: str, b: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.

    Purpose:
      - Provide a robust similarity measure for short OCR strings.
      - Used for both ASCII and CJK candidate matching.

    Args:
      a:
        - First string.
      b:
        - Second string.

    Returns:
      - Integer edit distance (0 = identical).

    Notes:
      - Uses a memory-optimized DP row approach.
      - Ensures `a` is the shorter string to reduce memory footprint.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Ensure a is the shorter string for less memory.
    if len(a) > len(b):
        a, b = b, a

    prev = list(range(len(a) + 1))
    for j, bj in enumerate(b, start=1):
        cur = [j]
        for i, ai in enumerate(a, start=1):
            ins = cur[i - 1] + 1
            dele = prev[i] + 1
            sub = prev[i - 1] + (0 if ai == bj else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def _levenshtein_ratio(a: str, b: str) -> float:
    """
    Convert Levenshtein distance to a normalized similarity ratio.

    Purpose:
      - Provide a similarity score in [0, 1] that is length-aware.

    Args:
      a:
        - First string.
      b:
        - Second string.

    Returns:
      - 1.0 if identical, else 1 - (distance / max_len), clipped to [0, 1].

    Notes:
      - Returns 0.0 when either string is empty.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    d = _levenshtein_distance(a, b)
    m = max(len(a), len(b))
    return 0.0 if m <= 0 else max(0.0, 1.0 - (d / m))


def _bigrams(s: str) -> set[str]:
    """
    Generate bigrams (2-character shingles) from a string.

    Purpose:
      - Support a Jaccard similarity metric that is resilient to small edits.
      - Works reasonably across scripts for short tokens.

    Args:
      s:
        - Input string.

    Returns:
      - A set of bigram strings; for length<2 returns {s} or empty.

    Notes:
      - Whitespace is removed before shingling.
    """
    s = remove_all_whitespace(s)
    if len(s) < 2:
        return {s} if s else set()
    return {s[i : i + 2] for i in range(len(s) - 1)}


def _sim(a: str, b: str) -> float:
    """
    Compute Jaccard similarity on bigram sets.

    Purpose:
      - Provide a fast approximate similarity for OCR tokens.
      - Useful for clustering and containment heuristics.

    Args:
      a:
        - First string.
      b:
        - Second string.

    Returns:
      - Jaccard similarity in [0, 1].

    Notes:
      - Returns 0.0 if either bigram set is empty.
    """
    A = _bigrams(a)
    B = _bigrams(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    uni = len(A | B)
    return inter / uni if uni else 0.0


def _norm_cjk(s: str) -> str:
    """
    Normalize CJK strings for robust comparison.

    Purpose:
      - Remove invisible characters and normalize compatibility forms.
      - Improve matching when OCR uses variant code points or spacing.

    Args:
      s:
        - Input string.

    Returns:
      - Normalized string (NFKC, whitespace removed, zero-width chars removed).

    Notes:
      - Strips common zero-width characters: U+200B and BOM (U+FEFF).
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = remove_all_whitespace(s)
    s = s.replace("\u200b", "").replace("\ufeff", "")
    return s


def _normalize_ascii_for_compare(name: str) -> str:
    """
    Normalize ASCII name tokens for fuzzy matching.

    Purpose:
      - Reduce common OCR confusions (0/O, 1/I, 5/S, 8/B, 2/Z).
      - Keep only characters relevant to BattleTag-like identifiers.

    Args:
      name:
        - Raw name string.

    Returns:
      - Uppercased, sanitized string containing A-Z, 0-9, underscore.

    Notes:
      - Intended for matching, not for display.
    """
    s = (name or "").upper()
    s = s.replace("0", "O").replace("1", "I").replace("5", "S").replace("8", "B").replace("2", "Z")
    return re.sub(r"[^A-Z0-9_]", "", s)


def _name_variants_ascii(name: str) -> set[str]:
    """
    Generate ASCII variants for name matching.

    Purpose:
      - Account for roman numeral rank prefixes that may be attached by OCR.
      - Provide a small candidate set for robust equality/containment checks.

    Args:
      name:
        - Raw name string.

    Returns:
      - Set of normalized variants.

    Notes:
      - Produces the base normalized token plus an optional prefix-stripped token.
      - Only strips the first matching roman prefix where the remaining suffix stays length-safe.
    """
    s = _normalize_ascii_for_compare(name)
    out = {s}
    for p in _ROMAN_PREFIXES:
        if s.startswith(p) and (len(s) - len(p)) >= 3:
            out.add(s[len(p) :])
            break
    return {v for v in out if v}


def _toggle_hangul_tense(ch: str) -> str | None:
    """
    Toggle Hangul tense initial consonant (where applicable) for a syllable.

    Purpose:
      - Compensate for OCR confusion between tense and non-tense initials (e.g., ㄱ/ㄲ).
      - Generate plausible name variants for matching.

    Args:
      ch:
        - Single-character string.

    Returns:
      - The toggled Hangul syllable character if applicable, else None.

    Notes:
      - Only works for Hangul syllables in U+AC00..U+D7A3.
      - Keeps Jung/Jong components intact.
    """
    code = ord(ch)
    if code < _HANGUL_BASE or code > _HANGUL_END:
        return None

    offset = code - _HANGUL_BASE
    cho = offset // (_HANGUL_N_JUNG * _HANGUL_N_JONG)
    jung = (offset % (_HANGUL_N_JUNG * _HANGUL_N_JONG)) // _HANGUL_N_JONG
    jong = offset % _HANGUL_N_JONG

    if cho not in _HANGUL_TENSE_MAP:
        return None

    new_cho = _HANGUL_TENSE_MAP[cho]
    new_code = _HANGUL_BASE + ((new_cho * _HANGUL_N_JUNG + jung) * _HANGUL_N_JONG + jong)
    return chr(new_code)


def _hangul_tense_variants(text: str, max_variants: int = 64) -> list[str]:
    """
    Generate Hangul variants by toggling tense initial consonants across positions.

    Purpose:
      - Produce a bounded set of candidate names to tolerate OCR errors in Hangul initials.
      - Improve matching without requiring another OCR pass.

    Args:
      text:
        - Input Hangul-containing string.
      max_variants:
        - Maximum number of variants to generate (hard cap).

    Returns:
      - List of unique variants including the original string.

    Notes:
      - Uses a DFS walk that toggles each eligible character position on/off.
      - Stops generating once `max_variants` is reached.
    """
    positions: list[tuple[int, str]] = []
    for idx, ch in enumerate(text):
        toggled = _toggle_hangul_tense(ch)
        if toggled:
            positions.append((idx, toggled))

    if not positions:
        return [text]

    variants: set[str] = set()

    def _walk(i: int, current: list[str]) -> None:
        """
        Internal DFS generator for tense toggling combinations.

        Purpose:
          - Enumerate combinations of toggles while respecting max_variants.

        Args:
          i:
            - Index into `positions`.
          current:
            - Current mutable character list.

        Returns:
          - None (mutates outer `variants`).
        """
        if len(variants) >= max_variants:
            return
        if i >= len(positions):
            variants.add("".join(current))
            return
        pos, toggled = positions[i]
        _walk(i + 1, current)
        current2 = current.copy()
        current2[pos] = toggled
        _walk(i + 1, current2)

    _walk(0, list(text))
    variants.add(text)
    return list(variants)


def name_similarity(a: str | None, b: str | None) -> float:
    """
    Compute a script-aware similarity score between two names.

    Purpose:
      - Compare OCR-extracted names with hints or other sources robustly.
      - Use CJK-aware normalization and Hangul-specific tolerance when applicable.

    Args:
      a:
        - First name string (may be None).
      b:
        - Second name string (may be None).

    Returns:
      - Similarity score in [0, 1].

    Notes:
      - If either string contains meaningful CJK (>=2 chars), uses CJK path:
        - NFKC normalization + whitespace stripping
        - containment heuristic
        - Levenshtein ratio and bigram Jaccard
        - Hangul tense variants for Korean-heavy strings
      - Otherwise uses ASCII path:
        - normalized variants + containment + Levenshtein ratio
    """
    if not a or not b:
        return 0.0

    # CJK path (either string contains meaningful CJK)
    if count_cjk(a) >= 2 or count_cjk(b) >= 2:
        aa = _norm_cjk(a)
        bb = _norm_cjk(b)
        if not aa or not bb:
            return 0.0
        if aa == bb:
            return 1.0

        best = 0.0

        # Containment heuristic (useful if OCR adds one extra char)
        if aa in bb or bb in aa:
            best = max(best, min(len(aa), len(bb)) / max(len(aa), len(bb)))

        # Compare base + Hangul tense variants when relevant
        candidates = [aa]
        pa = build_script_profile(aa)
        if pa.hangul >= 0.45:
            candidates.extend(_hangul_tense_variants(aa, max_variants=32))

        for cand in set(candidates):
            best = max(best, _levenshtein_ratio(cand, bb), _sim(cand, bb))

        return best

    # ASCII path
    va = _name_variants_ascii(a)
    vb = _name_variants_ascii(b)
    if not va or not vb:
        return 0.0

    best = 0.0
    for x in va:
        for y in vb:
            if x == y:
                return 1.0
            if x in y or y in x:
                best = max(best, min(len(x), len(y)) / max(len(x), len(y)))
            best = max(best, _levenshtein_ratio(x, y))
    return best


def names_match_strict(a: str | None, b: str | None, threshold: float = HINT_MATCH_THRESHOLD) -> bool:
    """
    Strict name match predicate (used for hint validation).

    Purpose:
      - Decide whether an OCR name "matches" a provided hint strongly enough to trust it.

    Args:
      a:
        - Candidate name (e.g., OCR output).
      b:
        - Reference name (e.g., provided hint).
      threshold:
        - Similarity threshold; defaults to HINT_MATCH_THRESHOLD (typically 0.90).

    Returns:
      - True if similarity(a, b) >= threshold, else False.

    Notes:
      - Uses `name_similarity()` under the hood (script-aware).
    """
    return name_similarity(a, b) >= threshold


def _normalize_code_for_compare(code: str) -> str:
    """
    Normalize a map code for fuzzy comparisons.

    Purpose:
      - Make code matching resilient to OCR character confusions.
      - Ensure codes compare in a stable alphanumeric form.

    Args:
      code:
        - Raw map code string.

    Returns:
      - Uppercased, alphanumeric-only string with OCR confusion remaps applied.

    Notes:
      - Applies NFKC normalization and then strips non A-Z/0-9.
      - Remaps letters that OCR confuses with digits and vice-versa (safe for codes).
    """
    s = unicodedata.normalize("NFKC", (code or ""))
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    # Common OCR confusions (safe for codes)
    s = (
        s.replace("O", "0")
        .replace("Q", "0")
        .replace("I", "1")
        .replace("L", "1")
        .replace("S", "5")
        .replace("B", "8")
        .replace("Z", "2")
    )
    return s


def code_similarity(a: str | None, b: str | None) -> float:
    """
    Compute similarity between two map codes.

    Purpose:
      - Validate whether OCR and provided code hint refer to the same map.
      - Allow minor OCR errors while keeping strictness high.

    Args:
      a:
        - First code (may be None).
      b:
        - Second code (may be None).

    Returns:
      - Similarity score in [0, 1].

    Notes:
      - Uses normalized codes and returns max(Levenshtein ratio, bigram Jaccard).
    """
    if not a or not b:
        return 0.0
    aa = _normalize_code_for_compare(a)
    bb = _normalize_code_for_compare(b)
    if not aa or not bb:
        return 0.0
    if aa == bb:
        return 1.0
    return max(_levenshtein_ratio(aa, bb), _sim(aa, bb))


def time_similarity(extracted: float | None, hinted: float | None) -> float:
    """
    Compute similarity between two time values.

    STRICT ABSOLUTE policy:
    - Return 1.0 ONLY if abs(extracted - hinted) <= HINT_TIME_ABS_TOL
    - Otherwise return 0.0

    Rationale:
    - Relative comparisons are too permissive for large times (e.g. 4330 vs 4334.43).
    - We want "time hint" validation to behave like an absolute tolerance in seconds.
    """
    if extracted is None or hinted is None:
        return 0.0
    try:
        a = float(extracted)
        b = float(hinted)
    except Exception:
        return 0.0
    if a <= 0 or b <= 0:
        return 0.0

    diff = abs(a - b)
    return 1.0 if diff <= HINT_TIME_ABS_TOL else 0.0


def is_likely_leading_digit_truncation(shorter: float | None, fuller: float | None) -> bool:
    """
    Detect when OCR likely dropped one leading digit from a time value.

    Purpose:
      - Keep the rule narrow so it only fires on strong suffix evidence.
    """
    if shorter is None or fuller is None:
        return False
    try:
        s = int(round(float(shorter) * 100))
        f = int(round(float(fuller) * 100))
    except Exception:
        return False
    if f <= s:
        return False
    ss = str(s)
    fs = str(f)
    return len(fs) == len(ss) + 1 and fs.endswith(ss)


# =============================================================================
# NAME scoring helpers
# =============================================================================
def _clean_ascii_token(raw: str) -> str:
    """
    Clean and normalize a raw OCR token into a plausible ASCII name candidate.

    Purpose:
      - Reduce OCR noise and normalize diacritics.
      - Remove invalid characters and mitigate common OCR artifacts.

    Args:
      raw:
        - Raw token extracted from OCR text.

    Returns:
      - Uppercased token restricted to [A-Z0-9_] with some heuristics applied.

    Notes:
      - Strips combining marks (NFKD) to remove accents.
      - Removes long trailing digit sequences that look like false OCR tails.
      - If token is "polluted" (many digits), applies roman-prefix removal and digit->letter fixes.
    """
    norm = unicodedata.normalize("NFKD", (raw or ""))
    norm = "".join(ch for ch in norm if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Z0-9_]", "", norm.upper()).strip("_")
    if not s:
        return ""

    m_tail = re.search(r"\d{6,}$", s)
    if m_tail:
        prefix = s[: m_tail.start()]
        if len(prefix) >= 3 and sum(ch.isalpha() for ch in prefix) >= 2:
            s = prefix

    digit_count = sum(ch.isdigit() for ch in s)
    polluted = (len(s) >= 12 and digit_count >= 3)

    if polluted:
        s_for_prefix = s.replace("1", "I")
        for p in _ROMAN_PREFIXES:
            if s_for_prefix.startswith(p) and (len(s_for_prefix) - len(p)) >= 3:
                s = s[len(p) :]
                break

        s = s.replace("0", "O").replace("1", "I").replace("5", "S").replace("8", "B").replace("2", "Z")

    return s


def _strip_rank_prefix_ascii(name: str) -> str:
    """
    Strip a roman numeral prefix from an ASCII name when it is safe.

    Purpose:
      - Remove leaderboard rank prefixes that OCR sometimes merges into the name.
      - Keep the name stable for matching across banner/TOP5.

    Args:
      name:
        - Raw name token.

    Returns:
      - Cleaned token with a safe roman prefix removed if applicable.

    Notes:
      - Only strips if the remaining suffix still matches the ASCII name pattern and is length-safe.
    """
    s = _clean_ascii_token(name)
    if not s:
        return s

    s_for_prefix = s.replace("1", "I")
    for p in _ROMAN_PREFIXES:
        if s_for_prefix.startswith(p):
            suffix = s[len(p) :]
            if len(suffix) >= 3 and re.fullmatch(RE_ASCII_NAME_MATCH, suffix):
                return suffix
            return s
    return s


def _strip_rank_prefix_ascii_with_top5_hint(name: str, top5_text: str | None) -> str:
    """
    Strip rank prefix only if TOP5 text provides evidence for the suffix.

    Purpose:
      - Prevent over-stripping when OCR mistakes a real name prefix for a rank.
      - Use TOP5 presence as a confirmation signal.

    Args:
      name:
        - Raw or partially cleaned ASCII name.
      top5_text:
        - OCR text from TOP5 area (may be empty/None).

    Returns:
      - Possibly prefix-stripped name, otherwise the cleaned original.

    Notes:
      - If stripping changed the name but the stripped form is not found in TOP5,
        returns the non-stripped cleaned token instead.
    """
    base = _strip_rank_prefix_ascii(name)
    if not top5_text:
        return base

    if base == _clean_ascii_token(name):
        return base

    up = (top5_text or "").upper()
    if re.search(rf"\b{re.escape(base)}\b", up):
        return base

    return _clean_ascii_token(name)


def _score_name_candidate(c: OcrCandidate, cleaned_text: str) -> float:
    """
    Score a name candidate using OCR confidence + script/ROI heuristics.

    Purpose:
      - Rank candidates across languages/variants by combining confidence with plausibility.
      - Prefer names that match the expected script for their OCR language.

    Args:
      c:
        - Candidate metadata including OCR confidence, language, ROI label.
      cleaned_text:
        - Cleaned candidate text used for profiling/length checks.

    Returns:
      - A float score; higher is better.

    Notes:
      - Adds script-alignment bonuses (Hangul/Kana/Han).
      - Adds length bonus to prefer non-trivial names.
      - Adds ROI weight to prefer more reliable regions (BL/TR > banner/TL).
    """
    exp = expected_script_for_language(c.language_code)
    prof = build_script_profile(cleaned_text)

    bonus = 0.0
    if exp == "hangul" and prof.hangul >= 0.60:
        bonus += 0.45
    elif exp == "kana" and (prof.kana >= 0.45 or (prof.kana >= 0.25 and prof.han >= 0.25)):
        bonus += 0.35
    elif exp == "han" and prof.han >= 0.60:
        bonus += 0.30

    bonus += 0.40 * prof.hangul + 0.25 * prof.kana + 0.15 * prof.han

    length = len(remove_all_whitespace(cleaned_text))
    bonus += min(0.20, max(0.0, (length - 2) * 0.03))

    bonus += roi_label_weight(c.roi_label)
    return float(c.confidence) + bonus


def _consensus_pick(cands: list[tuple[str, float]]) -> str | None:
    """
    Pick a best name by clustering similar candidates and selecting the strongest cluster.

    Purpose:
      - Stabilize name selection when OCR produces multiple near-duplicates across variants.
      - Prefer candidates with consistent support rather than a single high-confidence outlier.

    Args:
      cands:
        - List of (name, score) candidate pairs.

    Returns:
      - The selected name string, or None if input is empty.

    Notes:
      - Clusters are formed using bigram similarity >= 0.72.
      - Cluster score is the sum of member scores; representative is best-scoring item.
    """
    if not cands:
        return None

    clusters: list[dict[str, object]] = []
    for name, score in cands:
        placed = False
        for cl in clusters:
            rep = cl["rep"]
            if isinstance(rep, str) and _sim(name, rep) >= 0.72:
                cl["items"].append((name, score))  # type: ignore[union-attr]
                cl["total"] = float(cl["total"]) + float(score)  # type: ignore[index]
                best_in = max(cl["items"], key=lambda x: x[1])  # type: ignore[union-attr]
                cl["rep"] = best_in[0]
                placed = True
                break
        if not placed:
            clusters.append({"items": [(name, score)], "rep": name, "total": float(score)})

    best_cluster = max(clusters, key=lambda cl: float(cl["total"]))  # type: ignore[index]
    best_item = max(best_cluster["items"], key=lambda x: x[1])  # type: ignore[index]
    return best_item[0]


# =============================================================================
# NAME: Bottom-left extraction (source of truth when NO name hint provided)
# =============================================================================
def extract_name_from_bottom_left(
    bl_name_roi: np.ndarray,
    bl_alt_roi: np.ndarray,
    *,
    cache: dict,
    name_hint: str | None = None,
) -> str | None:
    """
    Extract the player name from the bottom-left HUD ROI(s).

    Purpose:
      - Primary name extraction path in the normal flow (no explicit names provided).
      - Supports both ASCII and CJK names with multiple preprocessing variants.

    Args:
      bl_name_roi:
        - Bottom-left ROI expected to contain the player name.
      bl_alt_roi:
        - Alternate ROI (currently same in caller; kept for compatibility/experimentation).
      cache:
        - Per-request OCR cache dict used by `ocr_lines_cached`.
      name_hint:
        - Optional name hint to restrict candidates (looser than strict hint matching).

    Returns:
      - Best extracted name string, or None if not found.

    Notes:
      - ASCII path:
        - OCR with EN, tokenize, clean tokens, apply heuristics and optional hint filtering.
      - CJK path:
        - OCR with targeted languages, build variants, score candidates, and consensus-pick.
      - If CJK is picked, additional Hangul tense variants may be applied to improve stability.
    """
    """Extract player name from bottom-left HUD regions."""
    if bl_name_roi is None or bl_name_roi.size == 0:
        return None

    name_rois = [bl_name_roi]

    hint_ascii = None
    hint_cjk = None
    if name_hint:
        if count_cjk(name_hint) >= 2:
            hint_cjk = remove_all_whitespace(name_hint)
        else:
            hint_ascii = _normalize_ascii_for_compare(name_hint)

    # ---- ASCII candidates ----
    ascii_scores: dict[str, float] = {}
    for roi in name_rois:
        if roi is None or roi.size == 0:
            continue
        lines = ocr_lines_cached(roi, "en", cache)
        text = " ".join(t for t, _ in lines).replace("|", " ")
        tokens = [t for t in re.split(RE_SPACES, (text or "")) if t]
        avg_conf = float(np.mean([c for _, c in lines])) if lines else 0.0

        for raw in tokens:
            if not raw or raw.isdigit():
                continue
            tok = _clean_ascii_token(raw)
            if not tok:
                continue

            tok = _strip_rank_prefix_ascii(tok)
            tok = _clean_ascii_token(tok)
            if not tok or tok in _GENERIC_ASCII:
                continue
            if len(tok) < MIN_NAME_LEN:
                continue

            if hint_ascii:
                if _sim(tok, hint_ascii) < 0.55 and hint_ascii not in tok and tok not in hint_ascii:
                    continue

            digit_count = sum(ch.isdigit() for ch in tok)
            letter_count = sum(ch.isalpha() for ch in tok)
            if digit_count > 0 and letter_count < 4:
                continue

            if not re.fullmatch(RE_ASCII_NAME_MATCH, tok):
                continue

            score = avg_conf + 0.10 + max(0.0, (len(tok) - MIN_NAME_LEN) * 0.05)
            ascii_scores[tok] = ascii_scores.get(tok, 0.0) + score

    # ---- CJK candidates ----
    cjk_candidates: list[tuple[str, float]] = []

    def _collect_cjk_for_lang(lang: LanguageCode) -> None:
        """
        Internal helper: collect CJK name candidates for a specific OCR language.

        Purpose:
          - Run OCR on multiple variants for the ROI(s) and score plausible CJK substrings.
          - Apply script-based sanity checks per language.

        Args:
          lang:
            - OCR language to use ("korean", "japan", "ch", etc.)

        Returns:
          - None (appends to outer `cjk_candidates` list).

        Notes:
          - Uses `MIN_NAME_LEN` as minimum substring length.
          - Applies optional hint filtering using bigram similarity / containment.
        """
        for roi in name_rois:
            if roi is None or roi.size == 0:
                continue
            for v in build_cjk_name_variants(roi):
                for text, conf in ocr_lines_cached(v, lang, cache):
                    cjk = _cjk_best_substring_min(text, MIN_NAME_LEN)
                    if not cjk:
                        continue

                    if hint_cjk:
                        cjk_compact = remove_all_whitespace(cjk)
                        if _sim(cjk_compact, hint_cjk) < 0.55 and hint_cjk not in cjk_compact and cjk_compact not in hint_cjk:
                            continue

                    prof = build_script_profile(cjk)
                    if max(prof.hangul, prof.kana, prof.han) < 0.35:
                        continue
                    if lang == "korean" and prof.hangul < 0.35:
                        continue
                    if lang == "japan" and (prof.kana + prof.han) < 0.35:
                        continue
                    if lang == "ch" and prof.han < 0.50:
                        continue

                    fake = OcrCandidate(
                        text=text.strip(),
                        confidence=float(conf or 0.0),
                        language_code=lang,
                        roi_label="BL",
                        profile=build_script_profile(text),
                    )
                    score = _score_name_candidate(fake, cjk)
                    cjk_candidates.append((cjk, score))

    _collect_cjk_for_lang("korean")
    strong_korean = any(build_script_profile(n).hangul >= 0.55 for n, _ in cjk_candidates)
    if not strong_korean and not FAST_OCR:
        _collect_cjk_for_lang("japan")
        _collect_cjk_for_lang("ch")

    picked_cjk = _consensus_pick(cjk_candidates)
    picked_ascii = None
    if ascii_scores:
        picked_ascii = max(ascii_scores.items(), key=lambda kv: (kv[1], len(kv[0])))[0]

    if picked_cjk and count_cjk(picked_cjk) >= MIN_NAME_LEN:
        evidence = [name for name, _ in cjk_candidates]
        # If no evidence, keep the original.
        if evidence:
            # Try Hangul tense variants for small OCR mistakes.
            variants = _hangul_tense_variants(picked_cjk, max_variants=64)
            best = picked_cjk
            best_score = max((_sim(best, e) for e in evidence), default=0.0)
            for v in variants:
                score = max((_sim(v, e) for e in evidence), default=0.0)
                if score > best_score + 0.05:
                    best_score = score
                    best = v
            return best
        return picked_cjk

    if picked_ascii:
        picked_ascii = _clean_ascii_token(picked_ascii)
        picked_ascii = _strip_rank_prefix_ascii(picked_ascii)
        picked_ascii = _clean_ascii_token(picked_ascii)

    return picked_ascii


# =============================================================================
# NAME: Banner extraction (to decide if banner time is valid)
# =============================================================================
def extract_name_from_banner(text_banner: str) -> str | None:
    """
    Extract the player name from a banner text line.

    Purpose:
      - Determine whether a banner time belongs to a specific player.
      - Provide a name string for BL<->banner validation.

    Args:
      text_banner:
        - OCR text assembled from the banner ROI.

    Returns:
      - Extracted player name (ASCII or CJK), or None if not found.

    Notes:
      - CJK extraction uses the CJK banner pattern + best substring selection.
      - ASCII extraction cleans token, strips rank prefix, and validates against generic words.
    """
    """Extract player name from the banner text line."""
    if not text_banner:
        return None

    t = (text_banner or "").strip()

    m_cjk = RE_BANNER_NAME_CJK.search(t)
    if m_cjk:
        name = _cjk_best_substring(m_cjk.group(1) or "")
        if name and count_cjk(name) >= 2:
            return name

    m_ascii = RE_BANNER_NAME_ASCII.search(t.upper())
    if m_ascii:
        name = _clean_ascii_token(m_ascii.group(1) or "")
        name = _strip_rank_prefix_ascii(name)
        if name and name not in _GENERIC_ASCII and re.fullmatch(RE_ASCII_NAME_MATCH, name):
            return name

    return None


def names_match(a: str | None, b: str | None) -> bool:
    """
    Lenient internal match predicate for names (BL <-> banner).

    Purpose:
      - Validate that two names likely refer to the same player without requiring strict hint-level accuracy.
      - Support small OCR errors and Hangul tense confusions.

    Args:
      a:
        - First name.
      b:
        - Second name.

    Returns:
      - True if considered a match, else False.

    Notes:
      - CJK path:
        - Normalizes via `_norm_cjk()`, uses containment and bigram similarity.
        - For Korean-heavy strings, also tries Hangul tense variants.
      - ASCII path:
        - Compares normalized variants and containment.
    """
    """Lenient name matching for internal validations (BL <-> banner)."""
    if not a or not b:
        return False

    # CJK path
    if count_cjk(a) >= 2 or count_cjk(b) >= 2:
        aa = _norm_cjk(a)
        bb = _norm_cjk(b)
        if aa == bb:
            return True

        pa = build_script_profile(aa)
        pb = build_script_profile(bb)
        if pa.hangul >= 0.45 and pb.hangul >= 0.45:
            for v in _hangul_tense_variants(aa, max_variants=32):
                if v == bb or _sim(v, bb) >= 0.74:
                    return True

        if len(aa) >= 3 and (aa in bb or bb in aa):
            return True

        return _sim(aa, bb) >= 0.74

    # ASCII path
    va = _name_variants_ascii(a)
    vb = _name_variants_ascii(b)
    if va & vb:
        return True
    for x in va:
        for y in vb:
            if len(x) >= 4 and (x in y or y in x):
                return True
    return False


# =============================================================================
# TIME parsing helpers
# =============================================================================
def parse_loose_numeric_token(raw_token: str) -> float | None:
    """
    Parse a noisy OCR numeric token into a float.

    Purpose:
      - Convert OCR text that may contain misread characters into a numeric time.
      - Handle common OCR substitutions (O->0, S->5, etc.) and punctuation noise.

    Args:
      raw_token:
        - OCR token that should contain a number like "123.45" (but may be noisy).

    Returns:
      - Parsed float if a plausible value is found, else None.

    Notes:
      - Normalizes common OCR confusions before extracting a `\d{1,5}.\d{2}` pattern.
    """
    if not raw_token:
        return None
    normalized = (
        raw_token.upper()
        .replace("O", "0")
        .replace("Q", "0")
        .replace("D", "0")
        .replace("I", "1")
        .replace("L", "1")
        .replace("S", "5")
        .replace("B", "8")
        .replace("Z", "2")
        .replace("G", "6")
    )
    normalized = re.sub(RE_DIGITS_LOOSE_CLEANUP1, "", normalized).replace(",", ".")
    res = re.search(RE_DIGITS_LOOSE_CLEANUP2, normalized)
    return float(res.group(1)) if res else None


def extract_banner_time_seconds(text: str) -> float | None:
    """
    Extract a time value (seconds) from banner OCR text.

    Purpose:
      - Parse the "TIME ... SEC" segment from the mission complete banner.
      - Handle OCR distortions in keywords and digits.

    Args:
      text:
        - OCR text from the banner ROI.

    Returns:
      - Time in seconds as float, or None if not found.

    Notes:
      - Prefer parsing near the "TIME" keyword window when present.
      - Falls back to scanning all numeric candidates and ranking by proximity to TIME / presence of SEC.
    """
    if not text:
        return None
    text = (
        text.upper()
        .replace("T1ME", "TIME")
        .replace("TLME", "TIME")
        .replace("TI ME", "TIME")
        .replace("5EC", "SEC")
        .replace("SE€", "SEC")
        .replace("SEL", "SEC")
        .replace("SEG", "SEC")
    )
    text = re.sub(RE_SPACES, " ", text).strip()
    # Normalize common noisy numeric forms
    text = re.sub(r"(\d{1,5})\s*[,\s]\s*(\d{1,2})", r"\1.\2", text)

    time_idx = text.find("TIME")
    if time_idx != -1:
        window = text[time_idx : time_idx + 90]
        window = window.replace("O", "0")  # O -> 0 for misread numbers
        window = re.sub(r"([0-9OQDBZGISL]{1,5})\s+([0-9OQDBZGISL]{1}\.\d{2})", r"\1\2", window)
        window = re.sub(r"(\d{1,5})\s*[,\s]\s*(\d{1,2})", r"\1.\2", window)
        m = re.search(RE_PARSE_BANNER_TIME_SEARCH_WITH_SEC, window)
        if m:
            v = parse_loose_numeric_token(m.group(1))
            if v is not None:
                return v

    best: tuple[int, float] | None = None
    for m in re.finditer(RE_PARSE_BANNER_TIME_SEARCH_NO_SEC, text):
        cand = parse_loose_numeric_token(m.group(1))
        if cand is None:
            continue
        score = 0
        if time_idx != -1 and 0 <= (m.start() - time_idx) <= 90:
            score += 2
        if re.search(RE_PARSE_BANNER_TIME_SEARCH_ONLY_SEC, text[m.end() : m.end() + 8]):
            score += 1
        if best is None or score > best[0]:
            best = (score, float(cand))
    return best[1] if best else None


def extract_time_from_top_left(text_top_left: str, text_top_left_white: str) -> float | None:
    """
    Extract a time value from top-left HUD OCR text.

    Purpose:
      - Parse the displayed run time from the top-left HUD panel.
      - Combine plain OCR and white-mask OCR to improve reliability.

    Args:
      text_top_left:
        - OCR text from the top-left ROI.
      text_top_left_white:
        - OCR text from the white mask of the top-left ROI.

    Returns:
      - Parsed time in seconds (float), or None.

    Notes:
      - First tries strict "NNN.NN SEC" parse.
      - Then falls back to collecting all numeric candidates and returning the maximum.
    """
    src = f"{text_top_left_white or ''} {text_top_left or ''}".upper()
    # Normalize common noisy numeric
    src = re.sub(RE_SPACES, " ", src).strip()
    src = re.sub(r"(\d{1,5})\s*[,\s]\s*(\d{1,2})", r"\1.\2", src)
    m = re.search(RE_PARSE_TIME_AGAIN, src)
    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except Exception:
            return None

    values: list[float] = []
    for m2 in re.finditer(RE_PARSE_TOPLEFT_TIME_ANY, src):
        try:
            v = float(m2.group(1).replace(",", "."))
            if v > 0:
                values.append(v)
        except Exception:
            continue
    return max(values) if values else None


def validate_time_seconds(t: float | None) -> float | None:
    """
    Validate and clamp a candidate time value.

    Purpose:
      - Remove implausible or invalid time values before selecting/returning them.
      - Enforce service-level constraints on acceptable run times.

    Args:
      t:
        - Candidate time in seconds.

    Returns:
      - Rounded time (2 decimals) if valid, else None.

    Notes:
      - Current constraints discard values < 30.0 and > 15360 (4h 16m).
      - Keeps rounding consistent with API output and hint comparison.
    """
    if t is None:
        return None
    try:
        v = float(t)
    except Exception:
        return None
    # Discard implausible values
    if v < 30.0:
        return None
    if v > 15360:
        return None
    return round(v, 2)


def _preferred_langs_for_name_hint(name_hint: str) -> list[LanguageCode]:
    """
    Choose a minimal OCR language set likely to recognize the given name hint.

    Purpose:
      - Reduce compute by avoiding unnecessary multi-language OCR passes.
      - Still include EN as a cheap fallback for digits/keywords.

    Args:
      name_hint:
        - Provided name hint string.

    Returns:
      - List of language codes to try in order.

    Notes:
      - ASCII-like hints -> ["en"].
      - CJK hints -> [best_guess_lang, "en"] where best_guess_lang is inferred via script profile.
    """
    """
    Pick the cheapest OCR language set likely to recognize the name_hint.

    Rules:
    - ASCII -> ['en']
    - CJK -> [best_guess_lang, 'en'] (en as a cheap fallback for digits/keywords)
    """
    if not name_hint:
        return ["en"]

    if count_cjk(name_hint) < 2:
        return ["en"]

    prof = build_script_profile(name_hint)
    if prof.hangul >= max(prof.kana, prof.han):
        primary: LanguageCode = "korean"
    elif prof.kana >= prof.han:
        primary = "japan"
    else:
        primary = "ch"

    return [primary, "en"] if primary != "en" else ["en"]


def extract_confirmed_time_from_banner(
    banner_crop: np.ndarray,
    cache: dict,
    name_hint: str,
) -> tuple[float | None, str, str | None]:
    """
    Extract a banner time only if the banner name matches the provided name hint (strict).

    Purpose:
      - Prevent using banner time when the banner belongs to a different player.
      - Use strict name matching (>= HINT_MATCH_THRESHOLD) for hint-based flows.

    Args:
      banner_crop:
        - Banner ROI image.
      cache:
        - Per-request OCR cache dict.
      name_hint:
        - Target player name to verify against banner name.

    Returns:
      - (time_seconds_or_none, banner_text, extracted_banner_name_or_none)

    Notes:
      - Tries a small ordered language set from `_preferred_langs_for_name_hint()`.
      - OCRs both raw banner and its white mask for robustness.
      - Only returns a non-None time when BOTH:
        - a banner name is extracted, AND
        - it matches the hint strictly, AND
        - a valid time can be parsed and validated.
      - When not confirmed, returns (None, last_text_seen, last_name_seen).
    """
    """
    Return (time, banner_text, banner_name) ONLY IF banner_name matches name_hint.

    This uses STRICT matching for name hints (default 90%).
    """
    if banner_crop is None or banner_crop.size == 0:
        return None, "", None
    name_hint = (name_hint or "").strip()
    if not name_hint:
        return None, "", None

    banner_white = mask_white_regions(banner_crop)
    langs = _preferred_langs_for_name_hint(name_hint)

    all_lines: list[tuple[str, float]] = []
    last_text = ""
    last_name: str | None = None

    for lang in langs:
        lines: list[tuple[str, float]] = []
        lines.extend(ocr_lines_cached(banner_crop, lang, cache))
        if banner_white is not None:
            lines.extend(ocr_lines_cached(banner_white, lang, cache))

        all_lines.extend(lines)
        text = normalize_banner_fragment(join_lines(all_lines))
        last_text = text

        nm = extract_name_from_banner(text)
        last_name = nm

        if nm and names_match_strict(nm, name_hint):
            t = validate_time_seconds(extract_banner_time_seconds(text))
            if t is not None:
                return t, text, nm

        # Some maps render a leaderboard list inside the banner ROI rather than a
        # "MISSION COMPLETE" sentence. Reuse the leaderboard parser on banner text.
        t_list = validate_time_seconds(extract_time_from_top5(text, name_hint, min_similarity=HINT_MATCH_THRESHOLD))
        if t_list is not None:
            return t_list, text, name_hint

    return None, last_text, last_name


def extract_top5_text(top_right_crop: np.ndarray, cache: dict) -> tuple[str, str]:
    """
    Extract OCR text from the TOP5 leaderboard area.

    Purpose:
      - Read the TOP5 block (names + times) without contamination from unrelated UI text.
      - Provide a debug OCR line (optional) for troubleshooting.

    Args:
      top_right_crop:
        - Full top-right ROI image.
      cache:
        - Per-request OCR cache dict.

    Returns:
      - (top5_text, dbg_full_text)
        - top5_text: OCR text from TOP5 strips and variants.
        - dbg_full_text: optional OCR of full top-right ROI (only when OCR_DEBUG_TEXTS is enabled).

    Notes:
      - Uses two strip ROIs to avoid "HOLD/LEADERBOARD" junk.
      - Always runs EN OCR on strips; also adds limited Korean variants for CJK robustness.
      - In non-FAST_OCR mode, may add Japan/Chinese variants if little CJK is detected.
    """
    """Extract TOP5 text and a debug OCR line."""
    if top_right_crop is None or top_right_crop.size == 0:
        return "", ""

    tr0 = crop_within(top_right_crop, ROI_TR_TOP5_STRIP_0)
    tr1 = crop_within(top_right_crop, ROI_TR_TOP5_STRIP_1)
    tr2 = crop_within(top_right_crop, ROI_TR_TOP5_STRIP_2)

    top5_lines: list[tuple[str, float]] = []
    for strip in (tr0, tr1, tr2):
        if strip is None or strip.size == 0:
            continue

        top5_lines.extend(ocr_lines_cached(strip, "en", cache))

        # Add a small set of CJK name variants for robustness.
        for v in build_cjk_name_variants(strip):
            top5_lines.extend(ocr_lines_cached(v, "korean", cache))

        text_k = join_lines(top5_lines)
        if count_cjk(text_k) < 2 and not FAST_OCR:
            for v in build_cjk_name_variants(strip):
                top5_lines.extend(ocr_lines_cached(v, "japan", cache))
                top5_lines.extend(ocr_lines_cached(v, "ch", cache))

    top5_text = join_lines(top5_lines)

    dbg_full = ""
    if OCR_DEBUG_TEXTS:
        dbg_lines: list[tuple[str, float]] = []
        dbg_lines.extend(ocr_lines_cached(top_right_crop, "en", cache))
        dbg_full = join_lines(dbg_lines)

    return top5_text, dbg_full


def extract_time_from_top5(top5_text: str, target_name: str | None, *, min_similarity: float = 0.78) -> float | None:
    """
    Extract the target player's time from TOP5 leaderboard OCR text.

    Purpose:
      - Parse TOP5 entries (name + time) and select the time matching `target_name`.
      - Support both ASCII and CJK name formats.

    Args:
      top5_text:
        - OCR text extracted from TOP5 area.
      target_name:
        - Name to match against TOP5 entries.
      min_similarity:
        - Minimum similarity required to accept a match.
        - Use HINT_MATCH_THRESHOLD (e.g., 0.90) for strict hint matching.

    Returns:
      - Time in seconds (float) if a match is found and parsed, else None.

    Notes:
      - Normalizes the TOP5 block starting at the "TOP5" header when present.
      - Parses ASCII entries using an ASCII regex on uppercased text.
      - Parses CJK entries using a CJK regex on the raw text.
      - On ties, prefers earlier position in the text (more stable ordering).
    """
    """
    Extract the player time from the TOP5 leaderboard.

    Args:
      top5_text: OCR text from TOP5 block.
      target_name: Name to match.
      min_similarity: Minimum similarity threshold. Use 0.90 for strict hint matching.

    Returns:
      Time in seconds or None.
    """
    if not top5_text or not target_name:
        return None

    def _to_float(s: str) -> float | None:
        """
        Internal helper: convert a time token to float.

        Purpose:
          - Support comma or dot decimal separators.

        Args:
          s:
            - String containing a numeric token.

        Returns:
          - float value if parseable, else None.
        """
        return parse_loose_numeric_token(remove_all_whitespace(s or ""))

    def _normalize_top5_time_layout(text: str) -> str:
        text = re.sub(r"(?<!\d)(\d{1,2})\s+(\d{3,4}[.,]\d{2})(?!\d)", r"\1\2", text)
        return re.sub(r"(?<!\d)(\d{1,5})\s+(\d{2})(?!\d)", r"\1.\2", text)

    # Normalize and isolate TOP5 block
    upper_full = (top5_text or "").upper()
    m_top5 = RE_TOP5_SECTION.search(upper_full)
    block_upper = upper_full[m_top5.start() :] if m_top5 else upper_full
    block_upper = _normalize_top5_time_layout(block_upper)

    block_raw = top5_text
    m2 = RE_TOP5_SECTION.search(block_raw)
    block_raw = block_raw[m2.start() :] if m2 else block_raw
    block_raw = _normalize_top5_time_layout(block_raw)

    entries: list[tuple[int, str, float]] = []
    re_top5_time_for_name_ascii = re.compile(
        r"\b([A-Z][A-Z0-9_]{2,24})\b(?:\s*[-:|]\s*|\s+)((?:\d{1,2}\s+)?\d{1,5}[.,]\d{2})\s*(?:SEC|ì´ˆ)?",
        re.IGNORECASE,
    )
    re_top5_time_for_name_cjk = re.compile(
        rf"([{_CJK_ALL}]{{2,40}})(?:\s*[-:|]\s*|\s+)((?:\d{{1,2}}\s+)?\d{{1,5}}[.,]\d{{2}})\s*(?:SEC|ì´ˆ)?",
        re.IGNORECASE,
    )

    def _append_ascii_entry(pos: int, raw_name: str, t: float) -> None:
        """
        Internal helper: normalize/validate an ASCII name candidate before adding it.
        """
        nm = _strip_rank_prefix_ascii(raw_name)
        if not nm or nm in _GENERIC_ASCII:
            return
        if not any(ch.isalpha() for ch in nm):
            return
        if not re.fullmatch(RE_ASCII_NAME_MATCH, nm):
            return
        entries.append((pos, nm, t))

    # ASCII entries
    for m in re.finditer(re_top5_time_for_name_ascii, block_upper):
        t = _to_float(m.group(2))
        if t is None:
            continue
        _append_ascii_entry(m.start(), (m.group(1) or "").upper(), t)

    # Fallback for split ASCII names around time tokens, e.g. "BRAT1 SHKA7 833.79 SEC".
    # We build short joined tails from the text right before each time and let
    # similarity scoring decide the best candidate.
    re_time_any = re.compile(r"(?<![0-9.,])(\d{1,5}[.,]\d{2})\s*(?:SEC|ì´ˆ)?", re.IGNORECASE)
    re_time_any = re.compile(r"(?<![0-9.,])((?:\d{1,2}\s+)?\d{1,5}[.,]\d{2})\s*(?:SEC|ÃƒÂ¬Ã‚Â´Ã‹â€ |Ã¬Â´Ë†)?", re.IGNORECASE)
    re_ascii_chunk = re.compile(r"[A-Z0-9_]{2,24}")
    noise_tokens = _GENERIC_ASCII | {
        "TOP5",
        "CTRL",
        "CONTROL",
        "PERSON",
        "CLASS",
        "SERVER",
        "CAMERA",
        "PLAYTEST",
        "QUICK",
        "RESET",
        "SPECTATE",
        "HCTRLE",
        "TRLE",
    }
    for mt in re.finditer(re_time_any, block_upper):
        t = _to_float(mt.group(1))
        if t is None:
            continue
        before = block_upper[max(0, mt.start() - 52) : mt.start()]
        chunks = [tok for tok in re.findall(re_ascii_chunk, before) if any(ch.isalpha() for ch in tok)]
        if not chunks:
            continue
        filtered = [tok for tok in chunks if tok not in noise_tokens]
        if not filtered:
            continue
        tail = filtered[-3:]
        for n in range(1, len(tail) + 1):
            joined = "".join(tail[-n:])
            _append_ascii_entry(mt.start(), joined, t)

    # CJK entries
    for m in re.finditer(re_top5_time_for_name_cjk, block_raw):
        nm = _cjk_best_substring(m.group(1) or "")
        if not nm or count_cjk(nm) < 2:
            continue
        t = _to_float(m.group(2))
        if t is None:
            continue
        entries.append((m.start(), nm, t))

    if not entries:
        return None

    # Find best match by similarity
    best: tuple[float, int, float] | None = None  # (sim, pos, time)
    for pos, nm, t in entries:
        sim = name_similarity(nm, target_name)
        if sim < min_similarity:
            continue
        if best is None:
            best = (sim, pos, t)
            continue
        # Prefer higher similarity; on tie prefer earlier position (more stable)
        if sim > best[0] + 1e-6 or (abs(sim - best[0]) <= 1e-6 and pos < best[1]):
            best = (sim, pos, t)

    return best[2] if best else None


def extract_confirmed_time_from_top5(
    top_right_crop: np.ndarray,
    cache: dict,
    name_hint: str,
) -> tuple[float | None, str, str]:
    """
    Extract a TOP5 time only if TOP5 contains the provided name hint (strict).

    Purpose:
      - Support hint-driven flow where we must not return another player's time.
      - Use strict name matching threshold (>= HINT_MATCH_THRESHOLD) for validation.

    Args:
      top_right_crop:
        - Top-right ROI image.
      cache:
        - Per-request OCR cache dict.
      name_hint:
        - Target player name hint.

    Returns:
      - (time_seconds_or_none, top5_text, dbg_full_text)

    Notes:
      - ASCII hints:
        - Cheapest path: EN OCR over strip ROIs + white masks (+ optional grayscale in non-FAST_OCR).
      - CJK hints:
        - Uses `_preferred_langs_for_name_hint()` to select a primary language.
        - Runs only a capped number of variants to limit CPU load.
      - Debug full text is only returned when OCR_DEBUG_TEXTS is enabled.
    """
    """
    Return (time, top5_text, dbg_full_text) ONLY IF TOP5 contains name_hint and a time.

    This uses STRICT matching for name hints (default 90%).
    """
    if top_right_crop is None or top_right_crop.size == 0:
        return None, "", ""

    name_hint = (name_hint or "").strip()
    if not name_hint:
        return None, "", ""

    tr0 = crop_within(top_right_crop, ROI_TR_TOP5_STRIP_0)
    tr1 = crop_within(top_right_crop, ROI_TR_TOP5_STRIP_1)
    tr2 = crop_within(top_right_crop, ROI_TR_TOP5_STRIP_2)

    strips = (tr0, tr1, tr2)

    lines: list[tuple[str, float]] = []

    def _try_now() -> float | None:
        """
        Internal helper: attempt to parse a validated time for the current accumulated `lines`.

        Purpose:
          - Re-run parsing as we append more OCR lines to incrementally find a match.

        Returns:
          - Validated time (float) if found, else None.

        Notes:
          - Uses strict similarity threshold (HINT_MATCH_THRESHOLD) when calling `extract_time_from_top5`.
        """
        txt = join_lines(lines)
        t = extract_time_from_top5(txt, name_hint, min_similarity=HINT_MATCH_THRESHOLD)
        return validate_time_seconds(t)

    # ASCII: cheapest path (EN only, optional masks)
    if count_cjk(name_hint) < 2:
        for strip in strips:
            if strip is None or strip.size == 0:
                continue

            lines.extend(ocr_lines_cached(strip, "en", cache))

            w = mask_white_regions(strip)
            if w is not None:
                lines.extend(ocr_lines_cached(w, "en", cache))

            if not FAST_OCR:
                g = enhance_contrast_grayscale(strip)
                lines.extend(ocr_lines_cached(g, "en", cache))

            t = _try_now()
            if t is not None:
                dbg_full = ""
                if OCR_DEBUG_TEXTS:
                    dbg_full = join_lines(ocr_lines_cached(top_right_crop, "en", cache))
                return t, join_lines(lines), dbg_full

        # Fallback: OCR the full top-right ROI (still safe because parsing starts at TOP5)
        lines.extend(ocr_lines_cached(top_right_crop, "en", cache))
        if not FAST_OCR:
            g = enhance_contrast_grayscale(top_right_crop)
            lines.extend(ocr_lines_cached(g, "en", cache))

        t = _try_now()
        dbg_full = ""
        if OCR_DEBUG_TEXTS:
            dbg_full = join_lines(ocr_lines_cached(top_right_crop, "en", cache))
        return t, join_lines(lines), dbg_full

    # CJK: targeted language, limited variants
    langs = _preferred_langs_for_name_hint(name_hint)
    primary = langs[0] if langs else "korean"

    for strip in strips:
        if strip is None or strip.size == 0:
            continue

        variants = build_cjk_name_variants(strip)
        max_v = 4 if FAST_OCR else 6  # hard cap to reduce load
        for v in variants[:max_v]:
            lines.extend(ocr_lines_cached(v, primary, cache))

        t = _try_now()
        if t is not None:
            dbg_full = ""
            if OCR_DEBUG_TEXTS:
                dbg_full = join_lines(ocr_lines_cached(top_right_crop, "en", cache))
            return t, join_lines(lines), dbg_full

    # Optional EN fallback if primary didn't find it
    if "en" in langs and primary != "en":
        for strip in strips:
            if strip is None or strip.size == 0:
                continue
            lines.extend(ocr_lines_cached(strip, "en", cache))
            t = _try_now()
            if t is not None:
                dbg_full = ""
                if OCR_DEBUG_TEXTS:
                    dbg_full = join_lines(ocr_lines_cached(top_right_crop, "en", cache))
                return t, join_lines(lines), dbg_full

    # Fallback: OCR the full top-right ROI using primary language
    variants_full = build_cjk_name_variants(top_right_crop)
    max_v_full = 4 if FAST_OCR else 6
    for v in variants_full[:max_v_full]:
        lines.extend(ocr_lines_cached(v, primary, cache))

    t = _try_now()
    dbg_full = ""
    if OCR_DEBUG_TEXTS:
        dbg_full = join_lines(ocr_lines_cached(top_right_crop, "en", cache))
    return t, join_lines(lines), dbg_full


# =============================================================================
# TIME: final selection (decision tree)
# =============================================================================
def pick_final_time(
    bl_name: str | None,
    banner_name: str | None,
    banner_time: float | None,
    top5_time: float | None,
    top_left_time: float | None,
) -> float | None:
    """
    Select the final time output using a simple decision tree.

    Purpose:
      - Combine multiple time candidates (banner, TOP5, top-left) and pick the most reliable.
      - Prefer banner time only when the banner name matches the bottom-left name.

    Args:
      bl_name:
        - Name extracted from bottom-left ROI (or None).
      banner_name:
        - Name extracted from banner OCR (or None).
      banner_time:
        - Time extracted from banner OCR (or None).
      top5_time:
        - Time extracted from TOP5 OCR (or None).
      top_left_time:
        - Time extracted from top-left OCR (or None).

    Returns:
      - Selected time (float seconds) or None.

    Notes:
      - Validates each candidate via `validate_time_seconds()` before selection.
      - Priority:
        1) banner time if names match and time valid
        2) TOP5 time if valid
        3) top-left time if valid
    """
    """Pick the best time candidate using the decision tree."""
    bt = validate_time_seconds(banner_time)
    t5 = validate_time_seconds(top5_time)
    tl = validate_time_seconds(top_left_time)

    if bl_name and banner_name and names_match(bl_name, banner_name) and bt is not None:
        return bt
    if t5 is not None:
        return t5
    if tl is not None:
        return tl
    return None


# =============================================================================
# CODE extraction (top-left)
# =============================================================================
def normalize_map_code(raw_code_text: str | None, require_digit: bool = True) -> str | None:
    """
    Normalize and validate a candidate map code.

    Purpose:
      - Convert OCR output into a standardized code format.
      - Reject common false positives (keywords, times, junk tokens).

    Args:
      raw_code_text:
        - Raw candidate token that may contain a map code.
      require_digit:
        - Whether at least one digit must appear in the normalized code.

    Returns:
      - Normalized code string (uppercase, O->0) if valid, else None.

    Notes:
      - Rejects known generic words and suspicious tokens.
      - Enforces code length between 4 and 6.
      - Uses `RE_BASIC_NORMALIZATION` to keep only A-Z0-9.
    """
    """Normalize and validate a candidate map code string."""
    if not raw_code_text:
        return None

    GENERIC_BAD = {
        "MADE",
        "BY",
        "TIME",
        "SEC",
        "SPLIT",
        "LEVEL",
        "TOP",
        "PLAYTEST",
        "CODE",
        "C0DE",
        "BH0P",
        "BHOP",
        "AUTO",
        "MANTA",
        "KUMA",
        "WUHZI",
        "MOISTY",
    }

    raw_up = (raw_code_text or "").upper()
    if raw_up.endswith("SEC"):
        return None

    raw_clean = re.sub(RE_BASIC_NORMALIZATION, "", raw_up)
    if raw_clean in GENERIC_BAD:
        return None

    if not (4 <= len(raw_clean) <= 6):
        return None

    normalized = raw_clean.replace("O", "0")
    if require_digit and not any(ch.isdigit() for ch in normalized):
        return None
    return normalized


def extract_code(top_left_text: str, top_left_white_text: str, top_left_cyan_text: str) -> str | None:
    """
    Extract the map code from top-left OCR text using heuristic passes.

    Purpose:
      - Identify the map code in the "MAP CODE" HUD region.
      - Handle formats with keywords, colons, and loose code-like tokens.

    Args:
      top_left_text:
        - OCR text from the top-left ROI.
      top_left_white_text:
        - OCR text from the top-left white mask.
      top_left_cyan_text:
        - OCR text from the top-left cyan mask.

    Returns:
      - Best candidate map code string, or None.

    Notes:
      - Pass order:
        1) Keyword extraction (MAP CODE: XXXX)
        2) Colon-based candidates (score by frequency and sanity)
        3) Generic token scan (score by context window and letter/digit mix)
      - Uses `normalize_map_code()` for validation.
    """
    """Extract the map code using heuristic passes."""
    source_texts: list[tuple[str, float]] = [
        ((top_left_text or "").upper(), 1.0),
        ((top_left_white_text or "").upper(), 1.1),
        ((top_left_cyan_text or "").upper(), 1.5),
    ]
    normalized_sources: list[tuple[str, float]] = []
    for source_text, weight in source_texts:
        if not source_text.strip():
            continue
        normalized_sources.append((re.sub(RE_MAP_CODE_NORMALIZATION, "MAP CODE", source_text), weight))

    normalized = " ".join(text for text, _ in normalized_sources)

    keyword_candidates: dict[str, float] = defaultdict(float)
    for source_text, weight in normalized_sources:
        for m_keyword in re.finditer(RE_CODE_KEYWORD_EXTRACT, source_text):
            cand = normalize_map_code(m_keyword.group(1) or "", require_digit=False)
            if not cand:
                continue
            keyword_candidates[cand] += weight

    if keyword_candidates:
        best, _ = max(
            keyword_candidates.items(),
            key=lambda kv: (
                kv[1],
                int(any(ch.isalpha() for ch in kv[0]) and any(ch.isdigit() for ch in kv[0])),
                int(any(ch.isdigit() for ch in kv[0])),
                sum(ch.isdigit() for ch in kv[0]),
                len(kv[0]),
            ),
        )
        return best

    colon_candidates: dict[str, float] = defaultdict(float)
    for source_text, weight in normalized_sources:
        for m in re.finditer(RE_CODE_AFTER_COLON, source_text):
            token = m.group(1) or ""
            cand = normalize_map_code(token, require_digit=False)
            if not cand:
                continue

            letters = sum(ch.isalpha() for ch in cand)
            digits = sum(ch.isdigit() for ch in cand)
            if letters == 0:
                continue

            score = float(weight)
            if 5 <= len(cand) <= 6:
                score += 0.30
            if letters >= digits:
                score += 0.30
            if digits == 0:
                score += 0.15
            if digits >= 3:
                score -= 0.20

            before = source_text[max(0, m.start() - 20) : m.start()]
            after = source_text[m.end() : m.end() + 20]
            if "MAP" in before[-15:] or "MAP" in after[:15]:
                score += 1.00
            if "CODE" in before[-15:] or "CODE" in after[:15]:
                score += 1.00
            if "TIME" in before[-15:] or "SEC" in after[:10]:
                score -= 1.00

            colon_candidates[cand] += score

    if colon_candidates:
        best, _ = max(
            colon_candidates.items(),
            key=lambda kv: (
                kv[1],
                len(kv[0]),
                sum(ch.isalpha() for ch in kv[0]),
                int(any(ch.isdigit() for ch in kv[0])),
                -sum(ch.isdigit() for ch in kv[0]),
            ),
        )
        return best

    scores_all: dict[str, float] = defaultdict(float)

    for m in re.finditer(RE_MAP_CODE_FIND, normalized):
        token = m.group(0)
        if token in {
            "MADE",
            "BY",
            "TIME",
            "SEC",
            "SPLIT",
            "LEVEL",
            "TOP",
            "PLAYTEST",
            "CODE",
            "C0DE",
            "AUTO",
            "AUT0",
            "MANTA",
            "KUMA",
            "WUHZI",
        }:
            continue

        cand = normalize_map_code(token, require_digit=True)
        if not cand:
            continue

        has_letter = any(c.isalpha() for c in cand)
        has_digit = any(c.isdigit() for c in cand)

        score = 1.0
        if has_letter and has_digit:
            score += 1.0
        elif has_digit:
            score += 0.7

        before = normalized[max(0, m.start() - 20) : m.start()]
        after = normalized[m.end() : m.end() + 20]
        if ":" in before[-3:]:
            score += 0.8
        if "MAP" in before[-15:] or "MAP" in after[:15]:
            score += 1.0
        if "CODE" in before[-15:] or "CODE" in after[:15]:
            score += 1.0
        if "TIME" in before[-15:] or "SEC" in after[:10]:
            score -= 1.0

        scores_all[cand] += score

    if not scores_all:
        return None

    best_code, _ = max(
        scores_all.items(),
        key=lambda kv: (
            kv[1],
            int(any(c.isalpha() for c in kv[0]) and any(c.isdigit() for c in kv[0])),
            int(any(c.isalpha() for c in kv[0])),
            -sum(c.isdigit() for c in kv[0]),
            len(kv[0]),
        ),
    )
    return best_code


def extract_code_from_code_variants(
    top_left_bgr: np.ndarray,
    cache: dict[tuple[str, int, tuple[int, ...]], list[tuple[str, float]]],
) -> str | None:
    """
    OCR the top-left HUD with extra preprocessing variants and extract the code.

    Purpose:
      - Add stronger preprocessing when the standard raw/white/cyan passes disagree.
      - Recover final characters that disappear in the initial OCR blob.
    """
    variants = build_map_code_variants(top_left_bgr)
    if not variants:
        return None

    primary_texts: list[str] = []
    white_texts: list[str] = []
    aux_texts: list[str] = []

    for idx, variant in enumerate(variants):
        text = join_lines(ocr_lines_cached(variant, "en", cache))
        if not text:
            continue
        if idx < 2:
            primary_texts.append(text)
        elif idx < 4:
            white_texts.append(text)
        else:
            aux_texts.append(text)

    return extract_code(" ".join(primary_texts), " ".join(white_texts), " ".join(aux_texts))


# =============================================================================
# Hint preference helpers (compare OCR vs provided)
# =============================================================================
def prefer_provided_code(extracted_code: str | None, provided_code: str | None) -> str | None:
    """
    Prefer the client's provided code only if it matches OCR strongly.

    Purpose:
      - Keep OCR as the source of truth, but allow hint override when it agrees.
      - Optionally allow using hints as fallback when OCR fails (OCR_TRUST_HINTS).

    Args:
      extracted_code:
        - Code extracted from OCR (may be None).
      provided_code:
        - Code provided by the client (may be empty/None).

    Returns:
      - The chosen code:
        - provided_code if it matches extracted_code at >= HINT_MATCH_THRESHOLD
        - otherwise extracted_code
        - (optional) provided_code if OCR_TRUST_HINTS is enabled and extracted_code is missing

    Notes:
      - Normalizes provided_code using `normalize_map_code(require_digit=False)` for comparison.
      - Similarity uses `code_similarity()` and the global threshold.
    """
    """
    If provided_code matches extracted_code at >= 90%, return provided_code.
    Otherwise return extracted_code (or provided_code only if OCR_TRUST_HINTS fallback is enabled).
    """
    provided_raw = (provided_code or "").strip()
    if not provided_raw:
        # No hint provided -> normal behavior
        return extracted_code

    provided = normalize_map_code(provided_raw, require_digit=False)
    if not provided:
        return None

    # Require OCR extraction + strong match.
    if extracted_code:
        if code_similarity(provided, extracted_code) >= HINT_MATCH_THRESHOLD:
            return provided

    return None


def prefer_provided_time(extracted_time: float | None, provided_time: float | None) -> float | None:
    """
    Prefer the client's provided time only if it matches OCR strongly.

    Purpose:
      - Avoid trusting client-provided times unless OCR corroborates them.
      - Optionally allow using provided time as a fallback when OCR fails (OCR_TRUST_HINTS).

    Args:
      extracted_time:
        - Time extracted from OCR (may be None).
      provided_time:
        - Time provided by the client (may be 0/None).

    Returns:
      - The chosen time:
        - provided_time (rounded) if it matches extracted_time at >= HINT_MATCH_THRESHOLD
        - otherwise extracted_time
        - (optional) provided_time if OCR_TRUST_HINTS enabled and extracted_time is missing

    Notes:
      - Validates returned hint fallback via `validate_time_seconds()` to avoid nonsense.
      - Similarity uses `time_similarity()` and the global threshold.
    """
    """
    If provided_time matches extracted_time at >= 90%, return provided_time.
    Otherwise return extracted_time (or provided_time only if OCR_TRUST_HINTS fallback is enabled).
    """
    hint = None
    try:
        if provided_time and float(provided_time) > 0:
            hint = float(provided_time)
    except Exception:
        hint = None

    if hint is None:
        return extracted_time

    # Round the hint the same way as the output.
    hint_rounded = round(hint, 2)

    if extracted_time is not None and time_similarity(extracted_time, hint_rounded) >= HINT_MATCH_THRESHOLD:
        return hint_rounded

    if OCR_TRUST_HINTS and extracted_time is None:
        # Still validate to avoid returning nonsense.
        return validate_time_seconds(hint_rounded)

    return extracted_time


# =============================================================================
# OCR pipeline (sync)
# =============================================================================
def run_ocr_pipeline(
    img: np.ndarray,
    *,
    code: str,
    time: float | None,
    names: list[str],
) -> dict:
    """
    Run the full OCR pipeline on a decoded screenshot.

    Hint behavior:
    - If code/time are provided, we STILL extract from OCR.
    - If the OCR output matches the provided hint at >= 90%, we return the PROVIDED hint in the response.
    - If name(s) are provided, we do NOT use bottom-left. We try:
        1) Banner: if banner name matches (>=90%), return banner time
        2) TOP5: if any entry name matches (>=90%), return that entry time

      IMPORTANT (pair validation):
      - If a time hint is provided (time > 0) together with names, the returned time is ONLY non-null
        when BOTH the name is confirmed AND the OCR time matches the provided time (>= threshold).
      - If (name confirmed) but time does NOT match the provided time -> return time = None.
    """
    t0 = perf_counter()
    cache: dict = {}

    def _nonempty(s: str | None) -> str:
        return (s or "").strip()

    def _build_texts_payload(
        *,
        include_bottom_left: bool,
        banner_text: str,
        top5_text_in: str,
        tr_full_in: str,
    ) -> dict:
        out: dict[str, object] = {}

        tl_block: dict[str, str] = {}
        if _nonempty(text_top_left_en):
            tl_block["en"] = _nonempty(text_top_left_en)
        if _nonempty(text_top_left_white_en):
            tl_block["whiteMaskEn"] = _nonempty(text_top_left_white_en)
        if _nonempty(text_top_left_cyan_en):
            tl_block["cyanMaskEn"] = _nonempty(text_top_left_cyan_en)
        if tl_block:
            out["topLeft"] = tl_block

        if include_bottom_left:
            bl_txt = join_lines(ocr_lines_cached(bottom_left, "en", cache))
            if _nonempty(bl_txt):
                out["bottomLeft"] = {"en": _nonempty(bl_txt)}

        if _nonempty(banner_text):
            out["banner"] = {"text": _nonempty(banner_text)}

        tr_block: dict[str, str] = {}
        if _nonempty(top5_text_in):
            tr_block["top5"] = _nonempty(top5_text_in)
        if _nonempty(tr_full_in):
            tr_block["fullEn"] = _nonempty(tr_full_in)
        if tr_block:
            out["topRight"] = tr_block

        return out

    def _normalize_code_hint(raw: str) -> str | None:
        s = (raw or "").strip()
        return normalize_map_code(s, require_digit=False) if s else None

    def _normalize_time_hint(raw: float | None) -> float | None:
        try:
            if raw and float(raw) > 0:
                return validate_time_seconds(round(float(raw), 2))
        except Exception:
            return None
        return None

    # Normalize name hints (keep order, drop empties, de-dup)
    names_clean: list[str] = []
    seen: set[str] = set()
    for n in (names or []):
        s = (n or "").strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        names_clean.append(s)

    use_names_flow = bool(names_clean)
    primary_name_hint = names_clean[0] if names_clean else None

    # ---- crops ----
    top_left_wide = crop_by_frac_roi(img, ROI_TOPLEFT_WIDE)
    banner = crop_by_frac_roi(img, ROI_BANNER_TIGHT)
    top_right = crop_by_frac_roi(img, ROI_TOPRIGHT)
    bottom_left = crop_by_frac_roi(img, ROI_BOTTOMLEFT)
    bottom_left_name = bottom_left  # same ROI

    # -------------------------------------------------------------------------
    # TOP LEFT OCR (code only, minimal)
    # -------------------------------------------------------------------------
    text_top_left_en = ""
    text_top_left_white_en = ""
    text_top_left_cyan_en = ""
    top_left_time: float | None = None

    tl_lines: list[tuple[str, float]] = []
    tl_lines.extend(ocr_lines_cached(top_left_wide, "en", cache))
    text_top_left_en = join_lines(tl_lines)

    code_hint_norm = _normalize_code_hint(code)

    # Quick code attempt WITHOUT masks (cheap).
    map_code = extract_code(text_top_left_en, "", "")
    need_mask_pass = map_code is None
    if (not need_mask_pass) and code_hint_norm:
        # If a hint is present but the quick pass disagrees, run masked OCR before
        # deciding this is a mismatch.
        need_mask_pass = code_similarity(code_hint_norm, map_code) < HINT_MATCH_THRESHOLD

    if need_mask_pass:
        # Only do masks if needed for code refinement
        tl_white_mask = mask_white_regions(top_left_wide)
        tl_cyan_mask = mask_cyan_regions(top_left_wide)
        text_top_left_white_en = join_lines(ocr_lines_cached(tl_white_mask, "en", cache)) if tl_white_mask is not None else ""
        text_top_left_cyan_en = join_lines(ocr_lines_cached(tl_cyan_mask, "en", cache)) if tl_cyan_mask is not None else ""
        refined_map_code = extract_code(text_top_left_en, text_top_left_white_en, text_top_left_cyan_en)
        variant_map_code = extract_code_from_code_variants(top_left_wide, cache)
        if variant_map_code:
            if refined_map_code is None:
                refined_map_code = variant_map_code
            elif code_hint_norm:
                if code_similarity(code_hint_norm, variant_map_code) >= code_similarity(code_hint_norm, refined_map_code):
                    refined_map_code = variant_map_code
            elif len(variant_map_code) > len(refined_map_code):
                refined_map_code = variant_map_code
        if refined_map_code:
            if map_code is None:
                map_code = refined_map_code
            elif code_hint_norm:
                if code_similarity(code_hint_norm, refined_map_code) >= code_similarity(code_hint_norm, map_code):
                    map_code = refined_map_code
            elif any(ch.isdigit() for ch in refined_map_code) and not any(ch.isdigit() for ch in map_code):
                map_code = refined_map_code

    # Apply code hint preference (compare OCR vs provided)
    map_code_final = prefer_provided_code(map_code, code)

    code_source = None
    code_verified_by = None

    if (code_hint_norm or "").strip():
        # Hint was provided: we only return code if it matched OCR
        if map_code_final is not None:
            code_source = "hint"
            code_verified_by = "topLeft"
        else:
            # hint present but not confirmed by OCR
            # (either OCR couldn't extract, or mismatch)
            code_source = "hintMismatch" if map_code is not None else "hintUnconfirmed"
    else:
        # No hint provided: normal OCR behavior
        if map_code_final:
            code_source = "topLeft"

    # -------------------------------------------------------------------------
    # FLOW A: name(s) provided -> Banner (strict) then TOP5 (strict)
    #   + If time hint is provided, time must match ABSOLUTELY (via time_similarity/HINT_TIME_ABS_TOL)
    #   + If name is confirmed but time doesn't match -> return time = None (but keep the name)
    # -------------------------------------------------------------------------
    if use_names_flow:
        names_to_try = names_clean[:5]

        # Parse + validate provided time hint (if any)
        time_hint_valid = _normalize_time_hint(time)

        picked_name: str | None = None
        seconds: float | None = None

        # Track first confirmed name (even if time mismatch), so we can return name with time=None.
        first_confirmed_name: str | None = None

        text_banner = ""
        banner_name = None

        top5_text = ""
        tr_debug_full = ""

        time_source = None
        time_verified_by = None

        def _time_hint_matches(ocr_time: float | None) -> bool:
            """
            Return True if provided time hint matches the OCR time with strict ABS tolerance.
            """
            if time_hint_valid is None:
                return True  # no hint -> not a constraint
            if ocr_time is None:
                return False
            return time_similarity(ocr_time, time_hint_valid) >= HINT_MATCH_THRESHOLD

        def _get_top_left_time_for_validation() -> float | None:
            nonlocal top_left_time, text_top_left_white_en
            if top_left_time is not None:
                return top_left_time
            if not text_top_left_white_en:
                tl_white_mask = mask_white_regions(top_left_wide)
                text_top_left_white_en = (
                    join_lines(ocr_lines_cached(tl_white_mask, "en", cache)) if tl_white_mask is not None else ""
                )
            top_left_time = extract_time_from_top_left(text_top_left_en, text_top_left_white_en)
            return top_left_time

        for name_hint in names_to_try:
            # 1) Banner-confirmed time first (name is STRICTLY confirmed inside)
            banner_time_confirmed, text_banner, banner_name = extract_confirmed_time_from_banner(
                banner,
                cache,
                name_hint=name_hint,
            )
            if banner_time_confirmed is not None:
                # Name is confirmed here.
                if first_confirmed_name is None:
                    first_confirmed_name = name_hint

                # If a time hint is provided, it MUST match. Otherwise we keep searching other names.
                if _time_hint_matches(banner_time_confirmed):
                    picked_name = name_hint
                    if time_hint_valid is not None:
                        seconds = time_hint_valid
                        time_source = "hint"
                        time_verified_by = "banner"
                    else:
                        seconds = validate_time_seconds(banner_time_confirmed)
                        time_source = "banner"
                    break
                # else: mismatch -> keep searching for another (name,time) pair

            # 2) TOP5-confirmed time (name is STRICTLY matched inside parsing)
            top5_time_confirmed, top5_text, tr_debug_full = extract_confirmed_time_from_top5(
                top_right,
                cache,
                name_hint=name_hint,
            )
            if top5_time_confirmed is not None:
                if first_confirmed_name is None:
                    first_confirmed_name = name_hint

                # TOP5 is prone to dropping the leading thousands digit; if the
                # top-left HUD shows the same suffix with one extra leading digit,
                # don't let TOP5 validate the provided hint.
                if time_hint_valid is not None:
                    tl_for_validation = _get_top_left_time_for_validation()
                    if is_likely_leading_digit_truncation(top5_time_confirmed, tl_for_validation):
                        continue

                if _time_hint_matches(top5_time_confirmed):
                    picked_name = name_hint
                    if time_hint_valid is not None:
                        seconds = time_hint_valid
                        time_source = "hint"
                        time_verified_by = "topRight.top5"
                    else:
                        seconds = validate_time_seconds(top5_time_confirmed)
                        time_source = "topRight.top5"
                    break

        # If banner/top5 confirmed the name but not the hinted time, allow the
        # top-left timer to corroborate the same hint before returning unconfirmed.
        if time_hint_valid is not None and seconds is None:
            if not text_top_left_white_en:
                tl_white_mask = mask_white_regions(top_left_wide)
                text_top_left_white_en = (
                    join_lines(ocr_lines_cached(tl_white_mask, "en", cache)) if tl_white_mask is not None else ""
                )
            top_left_time = extract_time_from_top_left(text_top_left_en, text_top_left_white_en)

            if first_confirmed_name and _time_hint_matches(top_left_time):
                picked_name = first_confirmed_name
                seconds = time_hint_valid
                time_source = "hint"
                time_verified_by = "topLeft"
            else:
                picked_name = first_confirmed_name
                seconds = None
                time_source = "unconfirmed"

        texts_payload = _build_texts_payload(
            include_bottom_left=False,
            banner_text=text_banner,
            top5_text_in=top5_text,
            tr_full_in=(tr_debug_full if OCR_DEBUG_TEXTS else ""),
        )

        sources: dict[str, object] = {
            "name": "hint",
            "code": code_source,
            "time": time_source,
        }
        if code_verified_by:
            sources["codeVerifiedBy"] = code_verified_by
        if time_verified_by:
            sources["timeVerifiedBy"] = time_verified_by

        elapsed = perf_counter() - t0
        logger.info(f"[ocr] names-flow t={elapsed:.2f}s fast={FAST_OCR}")
        return {
            "extracted": {
                "name": picked_name,
                "time": seconds,
                "code": map_code_final,
                "sources": sources,
                "texts": texts_payload,
            }
        }

    # -------------------------------------------------------------------------
    # FLOW B: normal flow (bottom-left name, then TL/top5/banner decision tree)
    # -------------------------------------------------------------------------

    # ---- TL time ----
    if not text_top_left_white_en:
        tl_white_mask = mask_white_regions(top_left_wide)
        text_top_left_white_en = join_lines(ocr_lines_cached(tl_white_mask, "en", cache)) if tl_white_mask is not None else ""
    top_left_time = extract_time_from_top_left(text_top_left_en, text_top_left_white_en)

    # ---- NAME (BL) ----
    bl_name = extract_name_from_bottom_left(
        bottom_left_name,
        bottom_left,
        cache=cache,
        name_hint=primary_name_hint,
    )

    # ---- EARLY EXIT (fast path) ----
    if OCR_EARLY_EXIT and not ACCURATE_OCR and bl_name and map_code_final and top_left_time is not None:
        seconds_ocr = pick_final_time(
            bl_name=bl_name,
            banner_name=None,
            banner_time=None,
            top5_time=None,
            top_left_time=top_left_time,
        )
        seconds_ocr = validate_time_seconds(seconds_ocr)
        seconds = prefer_provided_time(seconds_ocr, time)

        time_source = "topLeft" if seconds_ocr is not None else None
        time_hint_valid = _normalize_time_hint(time)
        time_verified_by = None
        if time_hint_valid is not None:
            if seconds_ocr is None and OCR_TRUST_HINTS and seconds == time_hint_valid:
                time_source = "hint"
            elif seconds_ocr is not None and time_similarity(seconds_ocr, time_hint_valid) >= HINT_MATCH_THRESHOLD:
                time_source = "hint"
                time_verified_by = "topLeft"

        texts_payload = _build_texts_payload(
            include_bottom_left=True,
            banner_text="",
            top5_text_in="",
            tr_full_in="",
        )

        sources: dict[str, object] = {
            "name": "bottomLeft",
            "code": code_source,
            "time": time_source,
        }
        if code_verified_by:
            sources["codeVerifiedBy"] = code_verified_by
        if time_verified_by:
            sources["timeVerifiedBy"] = time_verified_by

        elapsed = perf_counter() - t0
        logger.info(f"[ocr] fast-exit t={elapsed:.2f}s fast={FAST_OCR}")
        return {
            "extracted": {
                "name": bl_name,
                "time": seconds,
                "code": map_code_final,
                "sources": sources,
                "texts": texts_payload,
            }
        }

    # ---- TOP5 (only if time is missing or accurate mode) ----
    top5_text = ""
    tr_debug_full = ""
    top5_time = None
    if (top_left_time is None or ACCURATE_OCR) and bl_name:
        top5_text, tr_debug_full = extract_top5_text(top_right, cache)
        if bl_name and count_cjk(bl_name) == 0:
            bl_name = _strip_rank_prefix_ascii_with_top5_hint(bl_name, top5_text)
        top5_time = extract_time_from_top5(top5_text, bl_name, min_similarity=0.78)

        if OCR_EARLY_EXIT and not ACCURATE_OCR and map_code_final and bl_name and top5_time is not None:
            seconds_ocr = pick_final_time(
                bl_name=bl_name,
                banner_name=None,
                banner_time=None,
                top5_time=top5_time,
                top_left_time=top_left_time,
            )
            seconds_ocr = validate_time_seconds(seconds_ocr)
            seconds = prefer_provided_time(seconds_ocr, time)

            time_source = "topRight.top5" if seconds_ocr is not None else None
            time_hint_valid = _normalize_time_hint(time)
            time_verified_by = None
            if time_hint_valid is not None:
                if seconds_ocr is None and OCR_TRUST_HINTS and seconds == time_hint_valid:
                    time_source = "hint"
                elif seconds_ocr is not None and time_similarity(seconds_ocr, time_hint_valid) >= HINT_MATCH_THRESHOLD:
                    time_source = "hint"
                    time_verified_by = "topRight.top5"

            texts_payload = _build_texts_payload(
                include_bottom_left=True,
                banner_text="",
                top5_text_in=top5_text,
                tr_full_in=(tr_debug_full if OCR_DEBUG_TEXTS else ""),
            )

            sources: dict[str, object] = {
                "name": "bottomLeft",
                "code": code_source,
                "time": time_source,
            }
            if code_verified_by:
                sources["codeVerifiedBy"] = code_verified_by
            if time_verified_by:
                sources["timeVerifiedBy"] = time_verified_by

            elapsed = perf_counter() - t0
            logger.info(f"[ocr] top5-exit t={elapsed:.2f}s fast={FAST_OCR}")
            return {
                "extracted": {
                    "name": bl_name,
                    "time": seconds,
                    "code": map_code_final,
                    "sources": sources,
                    "texts": texts_payload,
                }
            }

    # ---- BANNER (only if still missing time OR accurate mode) ----
    text_banner = ""
    banner_name = None
    banner_time = None

    if (top_left_time is None and top5_time is None) or ACCURATE_OCR:
        banner_lines: list[tuple[str, float]] = []
        banner_lines.extend(ocr_lines_cached(banner, "en", cache))

        banner_white = mask_white_regions(banner)
        banner_lines.extend(ocr_lines_cached(banner_white, "en", cache))

        text_banner = normalize_banner_fragment(join_lines(banner_lines))
        banner_name = extract_name_from_banner(text_banner)
        banner_time = extract_banner_time_seconds(text_banner)
        if bl_name and (banner_name is None or banner_time is None):
            banner_time_list = validate_time_seconds(extract_time_from_top5(text_banner, bl_name, min_similarity=0.78))
            if banner_time_list is not None:
                banner_name = bl_name
                banner_time = banner_time_list

        if (banner_time is None or (bl_name and banner_name and not names_match(bl_name, banner_name))) and not FAST_OCR:
            banner_gray = enhance_contrast_grayscale(banner)
            banner_binary = cv2.adaptiveThreshold(
                banner_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9
            )

            extra: list[tuple[str, float]] = []
            extra.extend(ocr_lines_cached(banner_binary, "en", cache))
            extra.extend(ocr_lines_cached(banner, "korean", cache))
            extra.extend(ocr_lines_cached(banner_white, "korean", cache))
            extra.extend(ocr_lines_cached(banner_binary, "korean", cache))

            if count_cjk(bl_name or "") >= 2:
                extra.extend(ocr_lines_cached(banner, "japan", cache))
                extra.extend(ocr_lines_cached(banner, "ch", cache))

            text_banner = normalize_banner_fragment(join_lines(banner_lines + extra))
            banner_name = extract_name_from_banner(text_banner)
            banner_time = extract_banner_time_seconds(text_banner)
            if bl_name and (banner_name is None or banner_time is None):
                banner_time_list = validate_time_seconds(extract_time_from_top5(text_banner, bl_name, min_similarity=0.78))
                if banner_time_list is not None:
                    banner_name = bl_name
                    banner_time = banner_time_list

    # ---- Final time decision tree ----
    seconds_ocr = pick_final_time(
        bl_name=bl_name,
        banner_name=banner_name,
        banner_time=banner_time,
        top5_time=top5_time,
        top_left_time=top_left_time,
    )
    seconds_ocr = validate_time_seconds(seconds_ocr)
    seconds = prefer_provided_time(seconds_ocr, time)

    time_source = None
    if seconds_ocr is not None:
        bt = validate_time_seconds(banner_time)
        t5 = validate_time_seconds(top5_time)
        tl = validate_time_seconds(top_left_time)

        if bl_name and banner_name and names_match(bl_name, banner_name) and bt is not None and seconds_ocr == bt:
            time_source = "banner"
        elif t5 is not None and seconds_ocr == t5:
            time_source = "topRight.top5"
        elif tl is not None and seconds_ocr == tl:
            time_source = "topLeft"

    time_hint_valid = _normalize_time_hint(time)
    time_verified_by = None
    if time_hint_valid is not None:
        if seconds_ocr is None and OCR_TRUST_HINTS and seconds == time_hint_valid:
            time_source = "hint"
        elif seconds_ocr is not None and time_similarity(seconds_ocr, time_hint_valid) >= HINT_MATCH_THRESHOLD:
            time_verified_by = time_source
            time_source = "hint"

    texts_payload = _build_texts_payload(
        include_bottom_left=True,
        banner_text=text_banner,
        top5_text_in=top5_text,
        tr_full_in=(tr_debug_full if OCR_DEBUG_TEXTS else ""),
    )

    sources: dict[str, object] = {
        "name": "bottomLeft" if bl_name else None,
        "code": code_source,
        "time": time_source,
    }
    if code_verified_by:
        sources["codeVerifiedBy"] = code_verified_by
    if time_verified_by:
        sources["timeVerifiedBy"] = time_verified_by

    elapsed = perf_counter() - t0
    logger.info(f"[ocr] done t={elapsed:.2f}s fast={FAST_OCR}")
    return {
        "extracted": {
            "name": bl_name,
            "time": seconds,
            "code": map_code_final,
            "sources": sources,
            "texts": texts_payload,
        }
    }


# =============================================================================
# FastAPI
# =============================================================================
@dataclass
class _OcrJob:
    img: np.ndarray
    url: str
    fut: "asyncio.Future[ApiResponse]"
    enqueued_at: float
    code: str
    time: float | None
    names: list[str]


async def _ocr_worker(app: FastAPI) -> None:
    """
    Process OCR jobs sequentially from a FIFO queue.

    Purpose:
      - Serialize PaddleOCR inference to avoid oneDNN/MKLDNN concurrency crashes.
      - Provide backpressure behavior via queue size + wait timeouts.

    Args:
      app:
        - FastAPI application instance containing `state.ocr_queue`.

    Returns:
      - None (runs forever until cancelled).

    Notes:
      - Each job runs `run_ocr_pipeline()` inside `asyncio.to_thread()` to keep the event loop responsive.
      - Even though inference runs in a worker thread, jobs are strictly sequential due to single worker.
      - Sets result/exception on the job future unless cancelled.
    """
    """
    Single FIFO worker that runs PaddleOCR inference sequentially.
    This avoids oneDNN/MKLDNN concurrency crashes.
    """
    q: asyncio.Queue[_OcrJob] = app.state.ocr_queue

    while True:
        job = await q.get()
        try:
            # If the client disconnected / request cancelled, skip work
            if job.fut.cancelled():
                continue

            wait_s = perf_counter() - job.enqueued_at
            t0 = perf_counter()

            # Run the heavy OCR pipeline in a thread (still sequential thanks to the queue)
            res = await asyncio.wait_for(
                asyncio.to_thread(
                    run_ocr_pipeline,
                    job.img,
                    code=job.code,
                    time=job.time,
                    names=job.names,
                ),
                timeout=OCR_TIMEOUT_S,
            )

            run_s = perf_counter() - t0
            logger.info(f"[ocr] wait={wait_s:.2f}s run={run_s:.2f}s q={q.qsize()} url={job.url}")

            if not job.fut.cancelled():
                job.fut.set_result(res)

        except asyncio.TimeoutError as e:
            if not job.fut.cancelled():
                job.fut.set_exception(e)
        except Exception as e:
            if not job.fut.cancelled():
                job.fut.set_exception(e)
        finally:
            q.task_done()


async def _enqueue_ocr(
    app: FastAPI,
    img: np.ndarray,
    url: str,
    *,
    code: str,
    time: float,
    names: list[str],
) -> ApiResponse:
    """
    Enqueue an OCR job and await its completion.

    Purpose:
      - Provide a single async entrypoint for the API handler to schedule OCR work.
      - Apply queue capacity constraints and overall wait+run timeouts.

    Args:
      app:
        - FastAPI application instance with `state.ocr_queue`.
      img:
        - Decoded screenshot as OpenCV BGR ndarray.
      url:
        - Source image URL (for logging/observability).
      code:
        - Optional code hint.
      time:
        - Optional time hint.
      names:
        - Optional name hints list.

    Returns:
      - ApiResponse from `run_ocr_pipeline()`.

    Raises:
      - HTTPException(429) if queue is full (when OCR_QUEUE_MAXSIZE > 0).
      - asyncio.TimeoutError if the job exceeds (queue wait + OCR run) deadline.

    Notes:
      - Creates a Future bound to the current event loop and hands it to the worker via _OcrJob.
      - Cancels the future on wait timeout to allow the worker to skip completed work if possible.
    """
    """Enqueue an OCR job and wait for the result."""
    q: asyncio.Queue[_OcrJob] = app.state.ocr_queue
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[ApiResponse] = loop.create_future()

    if OCR_QUEUE_MAXSIZE > 0 and q.full():
        raise HTTPException(status_code=429, detail="OCR busy (queue full). Try again later.")

    await q.put(
        _OcrJob(
            img=img,
            url=url,
            fut=fut,
            enqueued_at=perf_counter(),
            code=code,
            time=time,
            names=names,
        )
    )

    try:
        return await asyncio.wait_for(fut, timeout=OCR_QUEUE_WAIT_S + OCR_TIMEOUT_S)
    except asyncio.TimeoutError:
        fut.cancel()
        raise


class ImageURLPayload(BaseModel):
    image_url: HttpUrl
    code: str = ""
    time: float | None = None
    names: list[str] = Field(default_factory=list)


@asynccontextmanager
async def warm_models_on_startup(app: FastAPI) -> AsyncIterator:
    """
    FastAPI lifespan handler: warm models and prepare shared resources.

    Purpose:
      - Load PaddleOCR models at startup.
      - Initialize a shared aiohttp session for image fetching.
      - Create the OCR FIFO queue and start the single worker task.

    Args:
      app:
        - FastAPI app instance.

    Yields:
      - Control back to FastAPI after startup is complete.

    Notes:
      - On shutdown:
        - Cancels the worker task and awaits it safely.
        - Closes the aiohttp session.
      - Queue maxsize uses OCR_QUEUE_MAXSIZE (0 means unlimited per asyncio semantics here).
    """
    """Warm models and create the HTTP session on startup."""
    log_model_dirs()
    warm_ocr_engines()

    timeout = aiohttp.ClientTimeout(total=20, connect=5, sock_read=15)
    app.state.http_session = aiohttp.ClientSession(timeout=timeout)

    # FIFO queue + single worker to serialize PaddleOCR calls
    app.state.ocr_queue = asyncio.Queue(maxsize=OCR_QUEUE_MAXSIZE if OCR_QUEUE_MAXSIZE > 0 else 0)
    app.state.ocr_worker_task = asyncio.create_task(_ocr_worker(app))

    yield

    # Shutdown
    app.state.ocr_worker_task.cancel()
    try:
        await app.state.ocr_worker_task
    except asyncio.CancelledError:
        pass

    await app.state.http_session.close()


app = FastAPI(title="GenjiPK OCR", lifespan=warm_models_on_startup)


@app.get("/ping")
def ping() -> dict:
    """
    Health-check endpoint.

    Purpose:
      - Verify the service is running and models are loaded.
      - Provide a quick view of warmed language engines.

    Returns:
      - Dict with:
        - ok: True
        - models: sorted list of loaded language keys

    Notes:
      - This endpoint does not perform OCR; it only reports readiness.
    """
    """Health check endpoint for warmed models."""
    return {"ok": True, "models": sorted(OCR_ENGINES.keys())}


@app.post("/extract", response_model=ApiResponse)
async def extract_ocr_data(payload: ImageURLPayload, request: Request):
    """
    Main API endpoint: fetch an image by URL and run OCR extraction.

    Purpose:
      - Download the screenshot from `payload.image_url`.
      - Decode into an OpenCV image.
      - Enqueue the OCR job and return the extracted results.

    Args:
      payload:
        - Request body containing:
          - image_url: URL to fetch
          - code/time/names: optional hints
      request:
        - FastAPI request object (used to access app state).

    Returns:
      - ApiResponse containing extracted fields.

    Raises:
      - HTTPException(503) if models are not ready.
      - HTTPException(400/408/500) for fetch/decode failures.
      - HTTPException(504) if OCR exceeds queue+run timeout.

    Notes:
      - Uses a shared aiohttp session stored in app.state for connection pooling.
      - Image fetch logs include byte size and elapsed time for observability.
      - OCR work is serialized by the internal queue/worker to avoid concurrency crashes.
    """
    """Extract name, time, and code from an image URL payload."""
    if "en" not in OCR_ENGINES:
        raise HTTPException(status_code=503, detail="OCR models not ready yet")

    try:
        session = request.app.state.http_session
        image_url = str(payload.image_url)
        t0 = perf_counter()
        async with session.get(image_url) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=400, detail=f"failed to fetch image: HTTP {resp.status}")
            image_bytes = await resp.read()
        elapsed = perf_counter() - t0
        logger.info(f"[fetch] {image_url} bytes={len(image_bytes)} t={elapsed:.2f}s")
        img = decode_image_bytes(image_bytes)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="timeout fetching image")
    except aiohttp.ClientError as e:
        raise HTTPException(status_code=400, detail=f"error fetching image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"unexpected error: {e}")

    try:
        res = await _enqueue_ocr(
            request.app,
            img,
            image_url,
            code=payload.code,
            time=payload.time,
            names=payload.names,
        )
        from fastapi.responses import JSONResponse

        return JSONResponse(content=res)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="ocr timeout (queue + run)")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
