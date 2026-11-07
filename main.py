from __future__ import annotations

import asyncio
import base64
import dataclasses
import hashlib
import io
import logging
import os
import re
import unicodedata
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, AsyncIterator, Literal

import aiohttp
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from PIL import Image, ImageFile
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

import evidence
import fallback_ocr

if TYPE_CHECKING:
    from paddleocr import PaddleOCR, TextRecognition

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
        affinity = getattr(os, "sched_getaffinity", None)
        if affinity is not None:
            return max(1, len(affinity(0)))
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
OCR_HANGUL_TOLERANT_FALLBACK = _env_bool("OCR_HANGUL_TOLERANT_FALLBACK", default=True)
ACCURATE_OCR = _env_bool("ACCURATE_OCR", default=False)  # if True, run banner/top5 even if TL time found
OCR_TRUST_HINTS = _env_bool("OCR_TRUST_HINTS", default=False)  # legacy setting; cannot enable unsupported hints

# Name matching uses a conservative threshold; map codes always require literal equality.
HINT_MATCH_THRESHOLD = float(os.environ.get("HINT_MATCH_THRESHOLD", "0.90"))
HINT_TIME_ABS_TOL = float(os.environ.get("HINT_TIME_ABS_TOL", "0"))
MIN_VALID_TIME_SECONDS = float(os.environ.get("MIN_VALID_TIME_SECONDS", "10.0"))

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
    top_left: dict[str, str] = Field(default_factory=dict)
    banner: dict[str, str] = Field(default_factory=dict)
    top_right: dict[str, str] = Field(default_factory=dict)
    bottom_left: dict[str, str] = Field(default_factory=dict)


class ExtractedResult(CamelModel):
    name: str | None
    time: float | None
    code: str | None
    texts: ExtractedTexts
    sources: dict[str, object] = Field(default_factory=dict)
    diagnostics: dict[str, object] = Field(default_factory=dict)


class ApiResponse(CamelModel):
    extracted: ExtractedResult


# =============================================================================
# ROI (fractional coords)
# =============================================================================
ROI_TOPLEFT = [0.010, 0.020, 0.360, 0.300]
ROI_TOPLEFT_WIDE = [0.005, 0.010, 0.420, 0.340]
ROI_BANNER_TIGHT = [0.100, 0.120, 0.860, 0.400]
ROI_TOPRIGHT = [0.821, 0.077, 0.990, 0.730]
ROI_BOTTOMLEFT = [0.050, 0.825, 0.330, 0.990]

# TOP5 strip inside ROI_TOPRIGHT (to avoid "HOLD ... LEADERBOARD" junk)
ROI_TR_TOP5_STRIP_0 = [0.02, 0.18, 1.00, 0.62]
ROI_TR_TOP5_STRIP_1 = [0.22, 0.50, 1.00, 0.80]
ROI_TR_TOP5_STRIP_2 = [0.16, 0.58, 1.00, 0.98]

# =============================================================================
# Regex / parsing
# =============================================================================
RE_SPACES = re.compile(r"\s+")
RE_DIGITS_LOOSE_CLEANUP1 = re.compile(r"[^\d\.,]")
RE_DIGITS_LOOSE_CLEANUP2 = re.compile(r"(\d{1,5}\.\d{2})")

RE_TIME_TOKEN_PATTERN = r"[0-9OQDBZGISL]{1,5}[.,:][0-9OQDBZGISL]{2}"
RE_PARSE_TIME_AGAIN = re.compile(rf"(?<![0-9.,])({RE_TIME_TOKEN_PATTERN})\s*SEC", re.IGNORECASE)
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

RE_CODE_KEYWORD_EXTRACT = re.compile(
    r"(?:MAP\s+)?C(?:O|0)?DE\s*[:\-]?\s*"
    r"([A-Z0-9]{4,6}|[A-Z0-9]{1,4}\s+[A-Z0-9]{1,4})\b",
    re.IGNORECASE,
)
RE_CODE_AFTER_COLON = re.compile(r":\s*([A-Z0-9]{4,6})\b", re.IGNORECASE)
RE_NON_CODE_FIELD_BEFORE_COLON = re.compile(r"(?:MADE\s+BY|DISCORD|TIME|SPLIT)\s*$", re.IGNORECASE)
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

RE_TOP5_TIME_FOR_NAME_ASCII = re.compile(
    rf"\b([A-Z][A-Z0-9_]{{2,24}})\b\s+({RE_TIME_TOKEN_PATTERN})\s*(?:SEC)?",
    re.IGNORECASE,
)
RE_TOP5_TIME_FOR_NAME_CJK = re.compile(
    rf"([{_CJK_ALL}]{{2,40}})\s+({RE_TIME_TOKEN_PATTERN})\s*(?:SEC)?",
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
_HANGUL_TENSE_MAP = evidence.HANGUL_INITIAL_CONFUSIONS

# =============================================================================
# PaddleOCR engine registry
# =============================================================================
OCR_ENGINES: dict[LanguageCode, PaddleOCR] = {}
OCR_ENGINES_NO_MKLDNN: dict[LanguageCode, PaddleOCR] = {}
OCR_NAME_RECOGNIZERS: dict[str, TextRecognition] = {}
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
    det_name = "PP-OCRv5_mobile_det"

    if normalized_language == "en":
        rec_dir = _pick_existing_dir(v5_en_rec)
        rec_name = "en_PP-OCRv5_mobile_rec"
    elif normalized_language == "korean":
        rec_dir = _pick_existing_dir(v5_kr_rec)
        rec_name = "korean_PP-OCRv5_mobile_rec"
    elif normalized_language in ("ch", "japan"):
        rec_dir = _pick_existing_dir(v5_mobile_rec)
        rec_name = "PP-OCRv5_mobile_rec"
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
    from paddleocr import PaddleOCR  # noqa: PLC0415 - initialize CPU flags before importing Paddle

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
        logger.info(
            f"📥 Warming PaddleOCR model: {lang} (MKLDNN {'ON' if OCR_MKLDNN else 'OFF'}, threads={OCR_CPU_THREADS})"
        )
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
        logger.info(
            f"[models] {lang}: ocr={ocr_version} det={det_dir} rec={rec_dir} det_name={det_name} rec_name={rec_name}"
        )


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
def enhance_contrast_grayscale(image_bgr: np.ndarray, *, blur: bool = True) -> np.ndarray:
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
      - Applies CLAHE (adaptive histogram equalization) and, by default, a mild Gaussian blur.
    """
    g = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    g = _CLAHE.apply(g)
    if blur:
        return cv2.GaussianBlur(g, (3, 3), 0)
    return g


def laplacian_variance(image: np.ndarray) -> float:
    """
    Estimate edge energy via the variance of the Laplacian.

    Purpose:
      - Provide a cheap blur/definition heuristic for HUD ROIs.
      - Trigger stronger OCR preprocessing only when the crop looks soft.
    """
    if image is None or image.size == 0:
        return 0.0
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    try:
        return float(cv2.Laplacian(g, cv2.CV_32F).var())
    except Exception:
        return 0.0


def use_aggressive_latin_variants(roi_bgr: np.ndarray) -> bool:
    """
    Decide whether a Latin HUD ROI deserves stronger OCR preprocessing.

    Purpose:
      - Keep the normal path cheap on clean captures.
      - Spend extra work on tiny or blurry text regions.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return False
    h, w = roi_bgr.shape[:2]
    min_side = min(h, w)
    edge_energy = laplacian_variance(roi_bgr)
    if min_side <= 70:
        return True
    if min_side <= 110:
        return edge_energy < 95.0
    return edge_energy < 70.0


def ocr_lines_have_signal(lines: list[tuple[str, float]], *, min_chars: int = 4) -> bool:
    """
    Decide whether a first OCR pass already contains enough usable signal.

    Purpose:
      - Short-circuit expensive fallback variants on clean captures.
      - Keep the enhanced OCR path focused on weak/empty first-pass results.
    """
    if not lines:
        return False

    texts = [t.strip() for t, _ in lines if (t or "").strip()]
    if not texts:
        return False

    compact = re.sub(RE_SPACES, "", " ".join(texts))
    total_chars = len(compact)
    confs = [float(c or 0.0) for _, c in lines]
    max_conf = max(confs) if confs else 0.0
    avg_conf = float(np.mean(confs)) if confs else 0.0

    if total_chars >= 12:
        return True
    if total_chars >= min_chars and max_conf >= 0.72:
        return True
    if total_chars >= max(6, min_chars) and avg_conf >= 0.58:
        return True
    return False


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


def build_latin_hud_variants(roi_bgr: np.ndarray) -> list[np.ndarray]:
    """
    Build OCR variants for small Latin HUD text (names/times/TOP5).

    Purpose:
      - Recover thin ASCII text from blurry or compressed screenshots.
      - Reuse the same targeted preprocess stack across top-left, bottom-left, and TOP5.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return []

    variants: list[np.ndarray] = []
    base = roi_bgr
    variants.append(base)

    h, w = base.shape[:2]
    aggressive = use_aggressive_latin_variants(base)
    scale = 3.0 if min(h, w) < 110 else (2.6 if aggressive else 2.0)

    up = upscale(base, scale)
    up = unsharp(up, amount=1.9 if aggressive else 1.6, sigma=0.9 if aggressive else 1.0)
    variants.append(up)

    white = mask_white_regions(base)
    variants.append(white)

    hud = mask_hud_cyan_regions(base)
    variants.append(hud)

    gray = enhance_contrast_grayscale(base, blur=not aggressive)
    if FAST_OCR:
        if aggressive:
            variants.append(gray)
            white_up = cv2.resize(white, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            variants.append(cv2.dilate(white_up, _KERNEL_3, iterations=1))
        return variants

    variants.append(gray)

    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    variants.append(thr)
    variants.append(255 - thr)

    if aggressive:
        white_up = cv2.resize(white, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        variants.append(cv2.dilate(white_up, _KERNEL_3, iterations=1))

        hud_up = cv2.resize(hud, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        variants.append(cv2.dilate(hud_up, _KERNEL_3, iterations=1))

        gray_up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.append(gray_up)

    return variants


def ocr_lines_latin_hud(
    roi_bgr: np.ndarray,
    cache: dict[tuple[str, int, tuple[int, ...]], list[tuple[str, float]]],
) -> list[tuple[str, float]]:
    """
    OCR a Latin HUD ROI using focused preprocessing variants.

    Purpose:
      - Centralize the enhanced EN OCR path for small HUD text.
      - Keep callers simple while reusing the same variant set.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return []

    base_lines = ocr_lines_cached(roi_bgr, "en", cache)
    aggressive = use_aggressive_latin_variants(roi_bgr)
    if ocr_lines_have_signal(base_lines, min_chars=4) and not aggressive:
        return base_lines

    lines: list[tuple[str, float]] = list(base_lines)
    variants = build_latin_hud_variants(roi_bgr)
    for variant in variants[1:]:
        lines.extend(ocr_lines_cached(variant, "en", cache))
        if not aggressive and ocr_lines_have_signal(lines, min_chars=4):
            break
    return lines


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


def ocr_lines_cached(image: np.ndarray, language_code: LanguageCode, cache: dict) -> list[tuple[str, float]]:
    """Cache by image content; temporary ndarray identities can be recycled within a request."""
    if image.size == 0:
        return []
    key = (language_code, image.shape, hashlib.blake2b(image.tobytes(), digest_size=16).digest())
    if key not in cache:
        cache[key] = ocr_lines(image, language_code)
    return cache[key]


def ocr_detections(image: np.ndarray, language_code: LanguageCode) -> list[evidence.Detection]:
    """Preserve Paddle geometry and scores, retrying recognized oneDNN failures once."""
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
            logger.warning(
                f"OCR({language_code}) failed with oneDNN/PIR error under MKLDNN; retrying with MKLDNN OFF: {e}"
            )
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

    return evidence.parse_ocr_result(result, language=language_code)


def ocr_lines(image: np.ndarray, language_code: LanguageCode) -> list[tuple[str, float]]:
    """Expose lightweight text/confidence diagnostics without changing structured inference."""
    return [(d.text, d.confidence) for d in ocr_detections(image, language_code)]


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
    """Compare observed identities without deleting Unicode or inventing CJK variants."""
    return evidence.name_score(a, b)


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
    return bool(a and b) and name_similarity(a, b) >= max(0.90, threshold)


def text_contains_name_hint(text: str | None, name_hint: str | None) -> bool:
    """Check complete observed name tokens; this alone never confirms a run time."""
    if not text or not name_hint:
        return False
    return any(names_match_strict(token, name_hint) for token in re.findall(r"[^\W]+(?:[-'][^\W]+)*", text))


def extract_time_after_name_hint_from_text(text: str | None, name_hint: str | None) -> float | None:
    """Read an adjacent name/time pair on one explicit line, without look-ahead searching."""
    return extract_time_from_top5(text or "", name_hint, min_similarity=HINT_MATCH_THRESHOLD)


def text_contains_time_hint(text: str | None, time_hint: float | None) -> bool:
    """
    Return True when OCR text contains a time token matching the provided hint.

    This checks every visible time-like token instead of trusting the first OCR
    hit, which can be wrong in masks while the raw OCR line is correct.
    """
    hinted = validate_time_seconds(time_hint)
    if not text or hinted is None:
        return False

    normalized = normalize_split_time_layout((text or "").upper())

    for m in re.finditer(RE_TIME_TOKEN_PATTERN, normalized, re.IGNORECASE):
        candidate = validate_time_seconds(parse_loose_numeric_token(m.group(0)))
        if time_similarity(candidate, hinted) >= HINT_MATCH_THRESHOLD:
            return True

    return False


def names_match_cjk_aux(a: str | None, b: str | None, threshold: float = 0.90) -> bool:
    """Require independent identity evidence; a matching time cannot relax a CJK name match."""
    return count_cjk(a or "") >= 2 and count_cjk(b or "") >= 2 and names_match_strict(a, b)


def names_match_cjk_time_backed(a: str | None, b: str | None) -> bool:
    """Require independent identity evidence; a matching time cannot relax a CJK name match."""
    return count_cjk(a or "") >= 2 and count_cjk(b or "") >= 2 and names_match_strict(a, b)


def _normalize_code_for_compare(code: str) -> str:
    """Normalize case and compatibility forms without substituting code characters."""
    return unicodedata.normalize("NFKC", code or "").upper().strip()


def code_similarity(a: str | None, b: str | None) -> float:
    """Return exact literal code agreement; different Workshop codes remain distinct."""
    aa, bb = evidence.normalize_code(a), evidence.normalize_code(b)
    return 1.0 if aa is not None and aa == bb else 0.0


def codes_match_for_hint(extracted_code: str | None, provided_code: str | None) -> bool:
    """Require the complete literal code to be observed by OCR."""
    return code_similarity(extracted_code, provided_code) == 1.0


def time_similarity(extracted: float | None, hinted: float | None) -> float:
    """Use strict absolute comparison in integer centiseconds."""
    return float(evidence.times_equal(extracted, hinted, HINT_TIME_ABS_TOL))


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
    polluted = len(s) >= 12 and digit_count >= 3

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
    """Recognize Unicode names per spatial row, preserving their observed spelling."""
    langs: list[LanguageCode] = (
        _preferred_langs_for_name_hint(name_hint) if name_hint else ["en", "korean", "japan", "ch"]
    )
    for lang in langs:
        for idx, variant in enumerate((bl_name_roi, upscale(bl_name_roi, 3.0))):
            detections = read_roi_evidence(
                variant, lang, roi="bottomLeft", variant=f"name{idx}", original_shape=bl_name_roi.shape, cache=cache
            )
            for row in evidence.group_rows(detections):
                name = evidence.clean_name(row.text)
                if name and row.confidence >= 0.65 and (not name_hint or names_match_strict(name, name_hint)):
                    return name
    return None


# =============================================================================
# NAME: Banner extraction (to decide if banner time is valid)
# =============================================================================
def extract_name_from_banner(text_banner: str) -> str | None:
    """Read the displayed Unicode name preceding localized completion terminology."""
    for line in text_banner.splitlines():
        match = evidence.COMPLETION.search(line)
        if match:
            name = evidence.completion_name(line[: match.start()])
            if name:
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
def normalize_split_time_layout(text: str) -> str:
    """Normalize OCR-split time tokens without consuming digits from names.

    Leaderboard OCR can render ``1107.23`` as ``1 107.23`` or ``425.01`` as
    ``425 01``. The alphanumeric boundaries are important: with a digit-suffixed
    player name, a digit-only boundary would turn ``PLAYER44 425.01`` into
    ``PLAYER44425.01`` and make both the name and time impossible to confirm.
    """

    def _merge_leading_fragment(match: re.Match) -> str:
        suffix = match.group(2) or ""
        integer_part = re.split(r"[.,:]", suffix, maxsplit=1)[0]
        if len(re.sub(r"\D", "", integer_part)) >= 4:
            # An isolated digit before an already four-digit time is commonly a
            # leaderboard rank, not a dropped leading time digit.
            return match.group(0)
        return f"{match.group(1)}{suffix}"

    normalized = re.sub(
        r"(?<!\w)(\d{1,2})\s+(\d{3,4}[.,:]\d{2})(?!\w)",
        _merge_leading_fragment,
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"(?<!\w)(\d{1,5})\s+(\d{2})(?!\w)",
        r"\1.\2",
        normalized,
        flags=re.IGNORECASE,
    )


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
      - Normalizes common OCR confusions before extracting a `\\d{1,5}.\\d{2}` pattern.
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
    normalized = re.sub(RE_DIGITS_LOOSE_CLEANUP1, "", normalized.replace(":", ".")).replace(",", ".")
    res = re.search(RE_DIGITS_LOOSE_CLEANUP2, normalized)
    return float(res.group(1)) if res else None


def extract_banner_time_seconds(text: str) -> float | None:
    """Parse a time field for diagnostics; completion verification also requires a name pair."""
    values = evidence.time_values(text, MIN_VALID_TIME_SECONDS)
    return values[0][1] / 100 if len(values) == 1 else None


def extract_time_from_top_left(text_top_left: str, text_top_left_white: str) -> float | None:
    """Read a labelled live timer for diagnostics, excluding checkpoint splits and unrelated numbers."""
    for text in (text_top_left, text_top_left_white):
        for line in text.splitlines():
            if evidence.HUD_NOISE.search(line):
                continue
            if re.search(evidence.TIME_LABEL, line, re.IGNORECASE):
                values = evidence.time_values(line, MIN_VALID_TIME_SECONDS)
                if len(values) == 1:
                    return values[0][1] / 100
    return None


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
      - Current constraints discard values below MIN_VALID_TIME_SECONDS and > 15360 (4h 16m).
      - Keeps rounding consistent with API output and hint comparison.
    """
    if t is None:
        return None
    try:
        v = float(t)
    except Exception:
        return None
    # Discard implausible values
    if not np.isfinite(v) or v < MIN_VALID_TIME_SECONDS:
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

    if count_cjk(name_hint) == 0:
        return ["en"]

    prof = build_script_profile(name_hint)
    if prof.hangul >= max(prof.kana, prof.han):
        return ["korean", "en"]
    if prof.han > 0 and prof.kana == 0 and prof.hangul == 0:
        # Pure Han hints are ambiguous between Chinese and Japanese names.
        return ["japan", "ch", "en"]
    if prof.kana >= prof.han:
        return ["japan", "en"]
    return ["ch", "en"]


def extract_confirmed_time_from_banner(
    banner_crop: np.ndarray,
    cache: dict,
    name_hint: str,
) -> tuple[float | None, str, str | None]:
    """Verify a name/time relationship inside a single localized completion banner pass."""
    texts = []
    observed_name = None
    for lang in _preferred_langs_for_name_hint(name_hint):
        for idx, variant in enumerate((banner_crop, upscale(banner_crop, 2.0))):
            detections = read_roi_evidence(
                variant, lang, roi="banner", variant=f"banner{idx}", original_shape=banner_crop.shape, cache=cache
            )
            texts.extend(row.text for row in evidence.group_rows(detections))
            pairs = evidence.paired_evidence(detections, "banner", MIN_VALID_TIME_SECONDS)
            matches = [p for p in pairs if p.confidence >= 0.65 and names_match_strict(p.name, name_hint)]
            if matches:
                values = {p.centiseconds for p in matches}
                if len(values) == 1:
                    return matches[0].centiseconds / 100, "\n".join(texts), matches[0].name
            if pairs:
                observed_name = pairs[0].name
    return None, "\n".join(texts), observed_name


def extract_top5_names_for_time(top5_text: str, target_time: float | None) -> list[str]:
    """Return only names paired with the exact time in explicit text rows."""
    return [
        name
        for name, cs in evidence.text_pairs(top5_text, minimum=MIN_VALID_TIME_SECONDS)
        if time_similarity(cs / 100, target_time) == 1.0
    ]


def extract_top5_text(top_right_crop: np.ndarray, cache: dict) -> tuple[str, str]:
    """Expose independent spatial leaderboard rows as newline-separated diagnostic text."""
    detections = read_roi_evidence(top_right_crop, "en", roi="topRight", cache=cache)
    text = "\n".join(row.text for row in evidence.group_rows(detections))
    return text, text if OCR_DEBUG_TEXTS else ""


def extract_time_from_top5(top5_text: str, target_name: str | None, *, min_similarity: float = 0.90) -> float | None:
    """Read a target's adjacent leaderboard time without crossing explicit row boundaries."""
    if not target_name:
        return None
    values = {
        cs
        for name, cs in evidence.text_pairs(top5_text, minimum=MIN_VALID_TIME_SECONDS)
        if name_similarity(name, target_name) >= max(HINT_MATCH_THRESHOLD, min_similarity)
    }
    return next(iter(values)) / 100 if len(values) == 1 else None


def extract_confirmed_time_from_top5(
    top_right_crop: np.ndarray,
    cache: dict,
    name_hint: str,
) -> tuple[float | None, str, str]:
    """Verify same-row or directly stacked leaderboard entries within one OCR pass."""
    texts = []
    for lang in _preferred_langs_for_name_hint(name_hint):
        for idx, variant in enumerate((top_right_crop, upscale(top_right_crop, 2.0))):
            detections = read_roi_evidence(
                variant, lang, roi="topRight", variant=f"top5{idx}", original_shape=top_right_crop.shape, cache=cache
            )
            texts.extend(row.text for row in evidence.group_rows(detections))
            pairs = evidence.paired_evidence(detections, "topRight.top5", MIN_VALID_TIME_SECONDS)
            matches = [p for p in pairs if p.confidence >= 0.65 and names_match_strict(p.name, name_hint)]
            values = {p.centiseconds for p in matches}
            if len(values) == 1:
                return matches[0].centiseconds / 100, "\n".join(texts), ""
    return None, "\n".join(texts), ""


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
    """Prefer a matching completion banner, then a paired leaderboard; live timers are diagnostic."""
    if bl_name and banner_name and names_match_strict(bl_name, banner_name):
        candidate = validate_time_seconds(banner_time)
        if candidate is not None:
            return candidate
    return validate_time_seconds(top5_time)


# =============================================================================
# CODE extraction (top-left)
# =============================================================================
def normalize_map_code(raw_code_text: str | None, require_digit: bool = False) -> str | None:
    """Validate a literal 4-6 character code, optionally requiring a digit for legacy callers."""
    code = evidence.normalize_code(raw_code_text)
    return code if code and (not require_digit or any(c.isdigit() for c in code)) else None


def extract_code(top_left_text: str, top_left_white_text: str, top_left_cyan_text: str) -> str | None:
    """Extract labelled codes per pass, keeping duplicate variants from adding votes."""
    values = {
        code for text in (top_left_text, top_left_white_text, top_left_cyan_text) for code in evidence.code_values(text)
    }
    return next(iter(values)) if len(values) == 1 else None


def extract_provided_code_from_texts(texts: list[str], provided_code: str | None) -> str | None:
    """Confirm complete codes only in labelled fields, never in username substrings."""
    provided = evidence.normalize_code(provided_code)
    return provided if provided and any(provided in evidence.code_values(text) for text in texts) else None


def extract_code_from_code_variants(
    top_left_bgr: np.ndarray,
    cache: dict,
) -> str | None:
    """Recognize each targeted code variant independently, rejecting unresolved disagreement."""
    values = set()
    for variant in build_map_code_variants(top_left_bgr):
        values.update(evidence.code_values("\n".join(t for t, _ in ocr_lines_cached(variant, "en", cache))))
    return next(iter(values)) if len(values) == 1 else None


# =============================================================================
# Hint preference helpers (compare OCR vs provided)
# =============================================================================
def prefer_provided_code(extracted_code: str | None, provided_code: str | None) -> str | None:
    """Return a supplied code only when its complete literal identity was independently recognized."""
    if not (provided_code or "").strip():
        return extracted_code
    provided = evidence.normalize_code(provided_code)
    return provided if provided and codes_match_for_hint(extracted_code, provided) else None


def prefer_provided_time(extracted_time: float | None, provided_time: float | None) -> float | None:
    """Return a supplied result only when evidence agrees; never use hints as missing evidence."""
    if provided_time is None or provided_time == 0:
        return validate_time_seconds(extracted_time)
    hint = validate_time_seconds(provided_time)
    return hint if hint is not None and time_similarity(extracted_time, provided_time) == 1.0 else None


# =============================================================================
# OCR pipeline (sync)
# =============================================================================
def read_roi_evidence(  # noqa: PLR0913 - explicit OCR provenance and coordinate transform
    image: np.ndarray,
    language: LanguageCode,
    *,
    roi: str,
    offset: evidence.Point = (0.0, 0.0),
    variant: str = "raw",
    original_shape: tuple[int, ...] | None = None,
    cache: dict,
) -> list[evidence.Detection]:
    """Cache one inference and project its geometry back into the original screenshot."""
    if image.size == 0:
        return []
    key = ("structured", language, image.shape, hashlib.blake2b(image.tobytes(), digest_size=16).digest())
    if key not in cache:
        cache[key] = ocr_detections(image, language)
    base_shape = original_shape or image.shape
    sx, sy = image.shape[1] / base_shape[1], image.shape[0] / base_shape[0]
    model = {
        "en": "en_PP-OCRv5_mobile_rec",
        "korean": "korean_PP-OCRv5_mobile_rec",
        "japan": "PP-OCRv5_mobile_rec",
        "ch": "PP-OCRv5_mobile_rec",
    }[language]
    pass_id = f"{roi}:{language}:{variant}:{offset}:{base_shape}"
    return [
        replace(
            d,
            roi=roi,
            language=language,
            model=model,
            variant=variant,
            pass_id=pass_id,
            polygon=tuple((x / sx + offset[0], y / sy + offset[1]) for x, y in d.polygon) if d.polygon else None,
        )
        for d in cache[key]
    ]


def recognize_precise_name(image: np.ndarray) -> tuple[str, float] | None:
    """Read one bounded CJK name with a lazy, cached high-capacity recognizer."""
    model = "PP-OCRv5_server_rec"
    engine = OCR_NAME_RECOGNIZERS.get(model)
    if engine is None:
        from paddleocr import TextRecognition  # noqa: PLC0415 - lazy targeted fallback only

        model_dir = _pick_existing_dir(PADDLE_WHL_DIR / "rec" / "ppocrv5" / "PP-OCRv5_server_rec_infer")
        engine = TextRecognition(
            model_name=model, model_dir=model_dir, device="cpu", enable_mkldnn=False, cpu_threads=OCR_CPU_THREADS
        )
        OCR_NAME_RECOGNIZERS[model] = engine
    detections = evidence.parse_ocr_result(engine.predict(image))
    if len(detections) != 1 or detections[0].confidence < 0.85:
        return None
    name = evidence.clean_name(detections[0].text)
    return (name, detections[0].confidence) if name else None


def _needs_precise_hangul_name(name: str) -> bool:
    """Identify short Hangul readings whose thin medial stroke is ambiguous in this HUD font."""
    value = evidence.normalize_name(name)
    return (
        1 <= len(value) <= 2
        and all("\uac00" <= char <= "\ud7a3" for char in value)
        and any(((ord(char) - 0xAC00) // 28) % 21 == 0 for char in value)
    )


def run_ocr_pipeline(
    img: np.ndarray,
    *,
    code: str,
    time: float | None,
    names: list[str],
) -> dict:
    """Verify literal codes and paired completion/leaderboard results from independent OCR passes."""
    started = perf_counter()
    cache: dict = {}
    aliases = list(dict.fromkeys(n.strip() for n in names if n.strip()))
    code_hint = evidence.normalize_code(code)
    time_hint = validate_time_seconds(time) if time is not None and time != 0 else None
    invalid_time_hint = time is not None and time != 0 and time_hint is None
    height, width = img.shape[:2]
    regions = {
        "topLeft": ROI_TOPLEFT_WIDE,
        "banner": ROI_BANNER_TIGHT,
        "topRight": ROI_TOPRIGHT,
        "bottomLeft": ROI_BOTTOMLEFT,
    }
    crops = {label: crop_by_frac_roi(img, frac) for label, frac in regions.items()}
    offsets = {label: (float(int(frac[0] * width)), float(int(frac[1] * height))) for label, frac in regions.items()}
    observations: list[evidence.Detection] = []
    precise_entries: list[tuple[evidence.Pair, evidence.Pair]] = []
    passes = 0

    def scan(
        label: str,
        language: LanguageCode,
        variant: str = "raw",
        image: np.ndarray | None = None,
        geometry: tuple[evidence.Point, tuple[int, ...]] | None = None,
    ) -> None:
        nonlocal passes
        if perf_counter() - started >= OCR_TIMEOUT_S:
            return
        crop = crops[label] if image is None else image
        if crop.size == 0:
            return
        observations.extend(
            read_roi_evidence(
                crop,
                language,
                roi=label,
                offset=geometry[0] if geometry else offsets[label],
                variant=variant,
                original_shape=geometry[1] if geometry else crops[label].shape,
                cache=cache,
            )
        )
        passes += 1

    def same_entry(left: evidence.Pair, right: evidence.Pair) -> bool:
        return evidence.same_pair_region(left, right)

    def pairs_now() -> list[evidence.Pair]:
        native = [
            p
            for label, source in (("banner", "banner"), ("topRight", "topRight.top5"))
            for p in evidence.paired_evidence(
                [d for d in observations if d.roi == label], source, MIN_VALID_TIME_SECONDS
            )
            if p.confidence >= 0.65
        ]
        return [p for p in native if not any(same_entry(p, parent) for parent, _ in precise_entries)] + [
            replacement for _, replacement in precise_entries
        ]

    def bottom_left_names() -> list[tuple[str, evidence.Row]]:
        rows = evidence.group_rows([d for d in observations if d.roi == "bottomLeft" and d.confidence >= 0.65])
        return [(name, row) for row in rows if (name := evidence.clean_name(row.text))]

    def relevant_pairs() -> list[evidence.Pair]:
        supported_pairs = [
            pair
            for pair in pairs_now()
            if not _needs_precise_hangul_name(pair.name) or pair.rows[0].detections[0].model == "tesseract-best-kor"
        ]
        if aliases:
            return [p for p in supported_pairs if any(names_match_strict(p.name, a) for a in aliases)]
        bl = bottom_left_names()
        # With no submitted identity, a unique named leaderboard entry at the
        # submitted time is direct result evidence. Its own row supplies BOTH
        # fields; unrelated names and live timers never contribute to this.
        all_pairs = supported_pairs
        headers = [
            d.bounds
            for d in observations
            if d.roi == "topRight"
            and d.confidence >= 0.65
            and d.bounds
            and re.search(r"\bTOP\s*5\b|\ud0d1\s*5|\uc21c\uc704\s*5|\u524d\s*5", d.text, re.IGNORECASE)
        ]
        matching = [
            p
            for p in all_pairs
            if time_hint is not None
            and time_similarity(p.centiseconds / 100, time_hint)
            and p.rows[0].bounds
            and any(p.rows[0].bounds[1] >= box[3] for box in headers)
        ]
        identities_at_time = {evidence.normalize_name(p.name) for p in matching}
        unique = matching if len(identities_at_time) == 1 else []
        banner_names = [
            name
            for row in evidence.group_rows([d for d in observations if d.roi == "banner"])
            if row.confidence >= 0.65 and (name := extract_name_from_banner(row.text))
        ]
        identities = [name for name, _ in bl] + banner_names
        return [
            p
            for p in all_pairs
            if p.source == "banner" or p in unique or any(names_match_strict(p.name, name) for name in identities)
        ]

    def code_candidates() -> list[tuple[str, evidence.Row]]:
        detections = [d for d in observations if d.roi == "topLeft"]
        values = evidence.labelled_codes([d for d in detections if not d.variant.startswith("codeToken")])
        # These observations come from an explicitly located code field, not
        # arbitrary HUD tokens. Require a complete literal reading of the crop.
        for row in evidence.group_rows([d for d in detections if d.variant.startswith("codeToken")]):
            if row.confidence < 0.85:
                continue
            pieces = sorted(row.detections, key=lambda d: d.bounds[0] if d.bounds else 0)
            adjacent = True
            for left_piece, right_piece in pairwise(pieces):
                a, b = left_piece.bounds, right_piece.bounds
                if not a or not b or b[0] - a[2] > min(a[3] - a[1], b[3] - b[1]) * 0.65:
                    adjacent = False
            text = row.text.strip(" :\uff1a")
            bilingual = re.fullmatch(r"([A-Z0-9]{4,6})\s*[-:]\s*(?:代码|代碼|코드|CODE)", text, re.IGNORECASE)
            value = evidence.normalize_code(bilingual.group(1) if bilingual else re.sub(r"\s+", "", text))
            if adjacent and value:
                values.append((value, row))
        return values

    def verified_code() -> str | None:
        # Resolve the observed field before consulting the submitted code. A
        # matching hint must not stop the pass which reveals a clipped suffix.
        focused = [(value, row) for value, row in code_candidates() if row.detections[0].variant != "raw"]
        # This HUD font renders O and 0 ambiguously: several recognizers can
        # confidently agree on O for a real zero. Agreement alone therefore
        # cannot establish that glyph's identity. Abstain, without rewriting it
        # or consulting the hint. Unambiguous zero readings remain literal.
        tokens = [(value, row) for value, row in focused if row.detections[0].variant.startswith("codeToken")]
        pool = tokens or focused
        if (not tokens and not label_rows) or any("O" in value and row.confidence >= 0.85 for value, row in pool):
            return None
        values = {value for value, _ in pool}
        # An independently read longer field supersedes its truncated prefix.
        values = {
            value
            for value in values
            if not any(other.startswith(value) and len(other) > len(value) for other in values)
        }
        if len(values) != 1:
            return None
        value = next(iter(values))
        supporting = [row for observed, row in pool if observed == value]
        languages = {row.detections[0].language for row in supporting}
        if len(languages) >= 2:
            return value
        if tokens and max(row.confidence for row in supporting) >= 0.95:
            # A clean token may exclude the last label glyph while another
            # model still includes it. Require corroboration of the SAME full
            # literal value from that other model's labelled metadata field.
            for observed, row in code_candidates():
                if observed != value or row.confidence < 0.80 or row.detections[0].language in languages:
                    continue
                b = row.bounds
                for token_row in supporting:
                    a = token_row.bounds
                    if not a or not b or token_row.confidence < 0.95:
                        continue
                    glyph_height = min(a[3] - a[1], b[3] - b[1])
                    overlap = min(a[3], b[3]) - max(a[1], b[1])
                    if glyph_height > 0 and overlap >= glyph_height * 0.65 and b[0] <= (a[0] + a[2]) / 2 <= b[2]:
                        return value
        # A high-confidence complete targeted line can recover a missing final
        # glyph even when the other model cannot read that small character.
        shorter = any(value.startswith(other) and len(other) < len(value) for other, _ in focused)
        if not tokens and shorter and max(row.confidence for row in supporting) >= 0.95:
            return value
        return None

    def code_is_confirmed() -> bool:
        value = verified_code()
        # Four characters may be a clipped five-character code. Always inspect
        # the enlarged/token field before treating this as a complete reading.
        return value is not None and (len(value) > 4 or code_field_checked)

    def pair_is_confirmed() -> bool:
        pairs = relevant_pairs()
        banner = [pair for pair in pairs if pair.source == "banner"]
        pool = banner or pairs
        for alias in aliases or list(dict.fromkeys(pair.name for pair in pool)):
            values = {pair.centiseconds for pair in pool if names_match_strict(pair.name, alias)}
            if len(values) == 1 and (time_hint is None or time_similarity(next(iter(values)) / 100, time_hint)):
                return True
        return False

    # Latin code recognition is independent of the player's language.
    code_field_checked = False
    scan("topLeft", "en")
    candidates = code_candidates()
    # Locate observed label/code baselines and retain horizontal context. Crop to
    # one actual line where possible so tiny final characters are not lost in HUD text.
    label_rows = [r for r in evidence.group_rows(observations) if evidence.CODE_LABEL.search(r.text) and r.bounds]
    if label_rows:
        box = label_rows[0].bounds
        assert box is not None
        padding = max(5, int((box[3] - box[1]) * 0.25))
        x1 = max(0, int(box[0]) - padding)
        y1 = max(0, int(box[1]) - padding)
        x2 = min(width, int(box[2] + max(48, (box[3] - box[1]) * 2)))
        y2 = min(height, int(box[3]) + padding)
        focused = img[y1:y2, x1:x2]
        focus_offset = (float(x1), float(y1))
    else:
        focused = crops["topLeft"][: max(1, int(crops["topLeft"].shape[0] * 0.40))]
        focus_offset = offsets["topLeft"]
    enlarged_code = upscale(focused, 3.0)
    # Preserve native glyphs first: detector max-side resizing can undo
    # enlargement and erase a thin trailing character in high-res HUDs.
    scan("topLeft", "en", "codeNative", focused, (focus_offset, focused.shape))
    if not code_is_confirmed():
        scan("topLeft", "ch", "codeNative", focused, (focus_offset, focused.shape))
    if not code_is_confirmed():
        scan("topLeft", "en", "codeUpscale", enlarged_code, (focus_offset, focused.shape))
    if not code_is_confirmed():
        scan("topLeft", "ch", "codeLocalized", enlarged_code, (focus_offset, focused.shape))
    if not label_rows:
        scan("topLeft", "korean", "codeLocalized", focused, (focus_offset, focused.shape))
        scan("topLeft", "ch")
        label_rows = [
            row
            for row in evidence.group_rows(observations)
            if row.detections[0].roi == "topLeft" and evidence.CODE_LABEL.search(row.text) and row.bounds
        ]
    if label_rows:
        # Isolate the actual token after the observed label. Removing the
        # English label reduces word-context O/0 and I/1 recognition bias.
        row = label_rows[0]
        label = evidence.CODE_LABEL.search(row.text)
        if label and row.bounds:
            box = row.bounds
            prefix = row.text[: label.end()]
            advance = sum(0.45 if ch.isspace() or ch in ":-" else 1.0 for ch in row.text)
            prefix_advance = sum(0.45 if ch.isspace() or ch in ":-" else 1.0 for ch in prefix)
            left = max(0, int(box[0] + (box[2] - box[0]) * prefix_advance / max(1, advance) - (box[3] - box[1]) * 0.2))
            top = max(0, int(box[1]) - 5)
            right = min(width, int(box[2] + (box[3] - box[1])))
            bottom = min(height, int(box[3]) + 5)
            token = img[top:bottom, left:right]
            geometry = ((float(left), float(top)), token.shape)
            for language in ("en", "ch"):
                scan("topLeft", language, "codeToken", upscale(token, 4.0), geometry)
    if not code_is_confirmed():
        scan("topLeft", "en", "cyan", upscale(mask_hud_cyan_regions(focused), 3.0), (focus_offset, focused.shape))
    if not code_is_confirmed():
        for region in evidence.metadata_code_regions([d for d in observations if d.roi == "topLeft"]):
            row = region.row
            field = evidence.NOISY_CODE_FIELD.fullmatch(row.text)
            box = row.bounds
            if not field or not box:
                continue
            advance = sum(0.45 if ch.isspace() or ch in ":-" else 1.0 for ch in row.text)
            prefix = row.text[: field.start(1)]
            prefix_advance = sum(0.45 if ch.isspace() or ch in ":-" else 1.0 for ch in prefix)
            left = max(0, int(box[0] + (box[2] - box[0]) * prefix_advance / max(1, advance) - (box[3] - box[1]) * 0.4))
            top = max(0, int(region.creator_bounds[3]) + 1)
            right = min(width, int(box[2] + (box[3] - box[1]) * 0.5))
            bottom = min(height, int(region.timer_bounds[1]) - 1)
            if bottom <= top or right <= left:
                continue
            token = img[top:bottom, left:right]
            geometry = ((float(left), float(top)), token.shape)
            for language in ("en", "ch"):
                scan("topLeft", language, "codeTokenMetadata", token, geometry)
            if code_is_confirmed():
                break
    code_field_checked = True
    candidates = code_candidates()

    preferred: list[LanguageCode] = []
    for alias in aliases:
        for language in _preferred_langs_for_name_hint(alias):
            if language not in preferred:
                preferred.append(language)
    if aliases and all(count_cjk(alias) > 0 for alias in aliases):
        # An English-only pass cannot recognize an all-CJK allowed identity,
        # so it cannot produce a valid independent name/time pair here.
        preferred = [language for language in preferred if language != "en"]
    if "japan" in preferred and "ch" in preferred:
        # Both languages use the same explicit mobile recognizer and detector.
        preferred.remove("ch" if preferred.index("japan") < preferred.index("ch") else "japan")
    if not preferred:
        # The shared model reads Japanese, Han and Latin. Starting with English
        # can hallucinate a consistent Latin spelling for a Japanese HUD name
        # and then suppress the model that actually supports those characters.
        preferred = ["ch", "en", "korean"]

    # A live TL timer cannot trigger early success. Check both completion and
    # leaderboard evidence before selecting a value, even in FAST_OCR mode.
    primary = preferred[0]
    for label in ("banner", "topRight", "bottomLeft"):
        scan(label, primary)
    for language in preferred[1:]:
        if pair_is_confirmed():
            break
        scan("banner", language)
        scan("topRight", language)
        scan("bottomLeft", language)

    # A larger ROI pass may be needed to detect the entry at all. Establish
    # that real name/time relationship before the focused fallbacks consume it.
    seeded_upscale = False
    if not pair_is_confirmed() and not any(
        time_hint is None or time_similarity(pair.centiseconds / 100, time_hint) for pair in pairs_now()
    ):
        seeded_upscale = True
        for language in preferred[:2]:
            for label in ("banner", "topRight"):
                if max(crops[label].shape[:2]) < 960:
                    scan(label, language, "upscale", upscale(crops[label], 2.0))
            if pair_is_confirmed():
                break

    # Re-read only an observed name cell, retaining its existing same-entry
    # time relationship and each recognizer's separate provenance.
    for entry in pairs_now():
        voiced = any(ch in "\u3099\u309a" for ch in unicodedata.normalize("NFD", entry.name))
        korean = all("\uac00" <= ch <= "\ud7a3" for ch in entry.name)
        ambiguous_korean = _needs_precise_hangul_name(entry.name)
        if len(entry.rows) != 2 or (pair_is_confirmed() and not voiced and not ambiguous_korean):
            continue
        if time_hint is not None and not time_similarity(entry.centiseconds / 100, time_hint):
            continue
        if not korean and not re.search(r"[\u3040-\u30ff\u3400-\u9fff]", entry.name):
            continue
        box = entry.rows[0].bounds
        if not box or perf_counter() - started >= OCR_TIMEOUT_S:
            continue
        if korean:
            # Preserve native screenshot pixels in the observed name cell.
            # Whole-HUD scaling can erase the extra Hangul vowel stroke.
            left, top = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
            right, bottom = min(width, int(np.ceil(box[2]))), min(height, int(np.ceil(box[3])))
            crop = img[top:bottom, left:right]
            remaining = max(0.1, min(2.0, (OCR_TIMEOUT_S - (perf_counter() - started)) / 2))
            time_box = entry.rows[1].bounds
            model_dir = PADDLE_WHL_DIR / "rec" / "tesseract"
            word = (
                fallback_ocr.recognize_paired_korean_word(img, box, time_box, model_dir, timeout=remaining)
                if time_box and entry.source == "topRight.top5"
                else None
            )
            word_origin = (0, 0)
            if word is None:
                word = fallback_ocr.recognize_korean_word(crop, model_dir, timeout=remaining)
                word_origin = (left, top)
            result = (word.text, word.confidence) if word else None
            model = "tesseract-best-kor"
        else:
            padding = max(6, int((box[3] - box[1]) * 0.2))
            left, top = max(0, int(box[0]) - padding), max(0, int(box[1]) - padding)
            right, bottom = min(width, int(box[2]) + padding), min(height, int(box[3]) + padding)
            crop = img[top:bottom, left:right]
            try:
                result = recognize_precise_name(crop)
            except Exception:
                logger.warning("[ocr] focused name recognizer unavailable")
                result = None
            model = "PP-OCRv5_server_rec"
        passes += 1
        if result is None:
            continue
        value, confidence = result
        if not aliases and not korean:

            def unvoiced(name: str) -> str:
                return "".join(c for c in unicodedata.normalize("NFD", name) if c not in "\u3099\u309a")

            if unvoiced(value) != unvoiced(entry.name):
                continue
        parent = entry.rows[0].detections[0]
        if korean and word is not None:
            detections = []
            for piece in word.pieces:
                x1, y1, x2, y2 = piece.bounds
                original = tuple(
                    (word_origin[0] + x, word_origin[1] + y) for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
                )
                local = tuple(
                    ((x - word.origin[0]) * word.scale, (y - word.origin[1]) * word.scale)
                    for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
                )
                if piece.local_bounds is not None:
                    lx1, ly1, lx2, ly2 = piece.local_bounds
                    local = ((lx1, ly1), (lx2, ly1), (lx2, ly2), (lx1, ly2))
                detections.append(
                    replace(
                        parent,
                        text=piece.text,
                        confidence=piece.confidence,
                        polygon=original,
                        local_polygon=local,
                        model=model,
                        variant="preciseName",
                        pass_id=parent.pass_id + ":preciseName",
                    )
                )
        else:
            detections = [
                replace(
                    parent,
                    text=value,
                    confidence=confidence,
                    polygon=((left, top), (right, top), (right, bottom), (left, bottom)),
                    local_polygon=((0, 0), (right - left, 0), (right - left, bottom - top), (0, bottom - top)),
                    language="japan" if re.search(r"[\u3040-\u30ff]", value) else "ch",
                    model=model,
                    variant="preciseName",
                    pass_id=parent.pass_id + ":preciseName",
                )
            ]
        observations.extend(detections)
        name_row = evidence.Row(tuple(detections))
        pair = evidence.Pair(
            value,
            entry.centiseconds,
            entry.source,
            min(confidence, entry.rows[1].confidence),
            (name_row, entry.rows[1]),
        )
        precise_entries.append((entry, pair))
        if pair_is_confirmed():
            break

    if not pair_is_confirmed():
        targets = [
            p
            for p in pairs_now()
            if p.source == "topRight.top5" and (time_hint is None or time_similarity(p.centiseconds / 100, time_hint))
        ]
        seen_bounds = set()
        for entry in sorted(targets, key=lambda p: p.confidence, reverse=True):
            boxes = [row.bounds for row in entry.rows if row.bounds]
            if not boxes:
                continue
            left = max(0, int(min(b[0] for b in boxes)) - 8)
            top = max(0, int(min(b[1] for b in boxes)) - 5)
            right = min(width, int(max(b[2] for b in boxes)) + 8)
            bottom = min(height, int(max(b[3] for b in boxes)) + 5)
            bounds = (left, top, right, bottom)
            if bounds in seen_bounds:
                continue
            seen_bounds.add(bounds)
            target = img[top:bottom, left:right]
            geometry = ((float(left), float(top)), target.shape)
            language = preferred[0]
            scan("topRight", language, "entryCrop", upscale(target, 4.0), geometry)
            if not pair_is_confirmed():
                gray = enhance_contrast_grayscale(target, blur=False)
                scan("topRight", language, "entryGray", upscale(gray, 4.0), geometry)
            if pair_is_confirmed() or len(seen_bounds) >= 2:
                break

    if not pair_is_confirmed() and "korean" in preferred:
        # Some recognizers need the neighbouring leaderboard context to retain
        # a thin Hangul vowel stroke. Select at most two existing HUD strips
        # containing an already observed target entry, rather than trying every
        # language and preprocessing combination over the entire screenshot.
        targets = [
            pair
            for pair in pairs_now()
            if pair.source == "topRight.top5"
            and any("\uac00" <= char <= "\ud7a3" for char in pair.name)
            and (time_hint is None or time_similarity(pair.centiseconds / 100, time_hint))
        ]
        context_regions = []
        tr_height, tr_width = crops["topRight"].shape[:2]
        ox, oy = offsets["topRight"]
        for frac in (ROI_TR_TOP5_STRIP_0, ROI_TR_TOP5_STRIP_1, ROI_TR_TOP5_STRIP_2):
            left, top = ox + int(frac[0] * tr_width), oy + int(frac[1] * tr_height)
            right, bottom = ox + int(frac[2] * tr_width), oy + int(frac[3] * tr_height)
            distances = []
            for entry in targets:
                boxes = [row.bounds for row in entry.rows]
                if all(
                    box and left <= box[0] and top <= box[1] and right >= box[2] and bottom >= box[3] for box in boxes
                ):
                    centres = [(box[1] + box[3]) / 2 for box in boxes if box]
                    distances.append(abs((top + bottom) / 2 - sum(centres) / len(centres)))
            if distances:
                context_regions.append((min(distances), frac, (left, top)))
        for _, frac, origin in sorted(context_regions)[:2]:
            target = crop_within(crops["topRight"], frac)
            scale = 2.8 if min(target.shape[:2]) < 160 else 2.0
            processed = unsharp(upscale(target, scale), amount=1.7, sigma=1.0)
            scan("topRight", "korean", "nameContext", processed, (origin, target.shape))
            if pair_is_confirmed():
                break

    def retry_hangul_identity() -> tuple[evidence.HangulFallbackMatch | None, int]:
        target = evidence.to_centiseconds(time_hint)
        eligible = (
            OCR_HANGUL_TOLERANT_FALLBACK
            and aliases
            and target is not None
            and code_hint
            and verified_code() == code_hint
            and not pair_is_confirmed()
        )
        if not eligible or target is None:
            return None, 0
        headers = [
            d.bounds
            for d in observations
            if d.roi == "topRight"
            and d.confidence >= 0.65
            and d.bounds
            and re.search(r"\bTOP\s*5\b|\ud0d1\s*5|\uc21c\uc704\s*5|\u524d\s*5", d.text, re.IGNORECASE)
        ]

        def anchored_pairs() -> list[evidence.Pair]:
            return [
                pair
                for pair in pairs_now()
                if pair.source == "banner"
                or (pair.rows[0].bounds and any(pair.rows[0].bounds[1] >= header[3] for header in headers))
            ]

        match = evidence.select_hangul_fallback(anchored_pairs(), aliases, target)
        # A usable independent word reading is conclusive; a near-match hint
        # must not replace that observed identity, even if it differs by one glyph.
        if match is None or any(pair.centiseconds == target for _, pair in precise_entries):
            return None, 0
        boxes = [row.bounds for row in match.pair.rows]
        if not all(boxes):
            return None, 0
        left = max(0, int(min(box[0] for box in boxes if box)) - 8)
        top = max(0, int(min(box[1] for box in boxes if box)) - 5)
        right = min(width, int(max(box[2] for box in boxes if box)) + 8)
        bottom = min(height, int(max(box[3] for box in boxes if box)) + 5)
        crop = img[top:bottom, left:right]
        label = "banner" if match.pair.source == "banner" else "topRight"
        geometry = ((float(left), float(top)), crop.shape)
        before = passes
        scan(label, "korean", "hangulRetry", crop, geometry)
        if not pair_is_confirmed():
            scan(label, "korean", "hangulRetryGray", upscale(enhance_contrast_grayscale(crop, blur=False), 2), geometry)
        count = passes - before
        if not count or pair_is_confirmed():
            return None, count
        reread_pairs = [
            pair
            for pair in pairs_now()
            if pair.rows[0].detections[0].variant.startswith("hangulRetry") and pair.confidence >= 0.85
        ]
        if any(
            all(evidence.hangul_visual_changes(pair.name, alias) is None for alias in aliases) for pair in reread_pairs
        ):
            return None, count
        return evidence.select_hangul_fallback(anchored_pairs(), aliases, target), count

    tolerant_match, hangul_retry_passes = retry_hangul_identity()
    if not pair_is_confirmed() and not seeded_upscale and tolerant_match is None:
        # One enlarged fallback per relevant ROI/language, rather than a Cartesian
        # product of every language, mask and strip. Respect the request deadline.
        for language in preferred[:2]:
            for label in ("banner", "topRight"):
                if max(crops[label].shape[:2]) < 960:
                    scan(label, language, "upscale", upscale(crops[label], 2.0))
            if pair_is_confirmed():
                break

    if tolerant_match is None and not hangul_retry_passes:
        tolerant_match, hangul_retry_passes = retry_hangul_identity()
    all_pairs = pairs_now()
    relevant = relevant_pairs()
    banner_pairs = [p for p in relevant if p.source == "banner"]
    pool = banner_pairs or relevant
    selected: evidence.Pair | None = None
    # Hints may select an allowed identity supported by a complete row, but
    # cannot resolve contradictory times for that identity or supply digits.
    selections = []
    for alias in aliases or list(dict.fromkeys(p.name for p in pool)):
        matches = [p for p in pool if names_match_strict(p.name, alias)]
        if len({p.centiseconds for p in matches}) == 1:
            selections.append(max(matches, key=lambda p: p.confidence))
    supported = [p for p in selections if time_hint is not None and time_similarity(p.centiseconds / 100, time_hint)]
    if supported or selections:
        selected = (supported or selections)[0]
    if tolerant_match is not None:
        selected = tolerant_match.pair

    bl_names = bottom_left_names()
    observed_name = selected.name if selected else None
    name_source = selected.source if selected else None
    if observed_name is None:
        name_options = [(p.name, p.source) for p in relevant]
        target_has_owner = time_hint is not None and any(
            time_similarity(pair.centiseconds / 100, time_hint) for pair in all_pairs
        )
        # A local HUD name cannot override the independently named owner of the
        # submitted result, particularly when the screenshot is spectated.
        if not target_has_owner:
            name_options.extend((name, "bottomLeft") for name, _ in bl_names if not _needs_precise_hangul_name(name))
        for observed, origin in name_options:
            if not aliases or any(names_match_strict(observed, alias) for alias in aliases):
                observed_name, name_source = observed, origin
                break
    matched_alias = (
        tolerant_match.alias
        if tolerant_match
        else next((a for a in aliases if names_match_strict(observed_name, a)), None)
    )
    result_name = matched_alias if aliases else observed_name
    seconds_ocr = selected.centiseconds / 100 if selected else None
    seconds = None if invalid_time_hint else prefer_provided_time(seconds_ocr, time)
    if aliases and matched_alias is None:
        seconds = None

    codes = list(dict.fromkeys(value for value, _ in candidates))
    observed_code = verified_code()
    result_code = observed_code if not code or observed_code == code_hint else None
    sources: dict[str, object] = {
        "name": "hint" if matched_alias else name_source,
        "code": "hint"
        if code and result_code
        else ("hintMismatch" if code and codes else "hintUnconfirmed" if code else "topLeft" if result_code else None),
        "time": "hint"
        if time_hint is not None and seconds is not None
        else selected.source
        if selected and seconds is not None
        else "unconfirmed",
    }
    if result_code:
        sources["codeVerifiedBy"] = "topLeft"
    if matched_alias:
        sources["nameVerifiedBy"] = name_source
    if tolerant_match:
        sources["nameMatchMode"] = "hangul_tolerant_fallback"
    if seconds is not None and selected:
        sources["timeVerifiedBy"] = selected.source

    reasons = []
    if not result_name:
        reasons.append("name_mismatch" if bl_names or all_pairs else "name_not_detected")
        if any(
            _needs_precise_hangul_name(pair.name)
            and (time_hint is None or time_similarity(pair.centiseconds / 100, time_hint))
            for pair in all_pairs
        ):
            reasons.append("name_glyph_ambiguous")
    if not result_code:
        reasons.append("code_mismatch" if codes else "code_not_detected")
        ambiguity_pool = [
            (value, row) for value, row in candidates if row.detections[0].variant.startswith("codeToken")
        ]
        if not ambiguity_pool:
            ambiguity_pool = [(value, row) for value, row in candidates if row.detections[0].variant != "raw"]
        if any("O" in value and row.confidence >= 0.85 for value, row in ambiguity_pool):
            reasons.append("code_glyph_ambiguous")
    if seconds is None:
        reasons.append(
            "time_mismatch"
            if invalid_time_hint or (selected and time_hint is not None)
            else "name_time_pair_unconfirmed"
            if result_name
            else "completion_not_confirmed"
        )

    # Diagnostics remain grouped by ROI/pass. Never use these text blobs for validation.
    texts: dict[str, object] = {}
    for label in regions:
        grouped: dict[str, list[evidence.Detection]] = defaultdict(list)
        for d in observations:
            if d.roi == label:
                key = d.language if d.variant == "raw" else f"{d.variant}:{d.language}"
                grouped[key].append(d)
        block = {key: "\n".join(r.text for r in evidence.group_rows(ds)) for key, ds in grouped.items()}
        if label == "topLeft":
            block["whiteMaskEn"] = block.get("white:en", "")
            block["cyanMaskEn"] = block.get("cyan:en", "")
        if label == "banner":
            block["text"] = "\n".join(block.values())
        if label == "topRight":
            block["top5"] = "\n".join(block.values())
        texts[label] = block

    diagnostics: dict[str, object] = {
        "reasons": reasons,
        "observedName": observed_name,
        "codeCandidates": codes,
        "codeEvidence": [
            {
                "value": value,
                "confidence": row.confidence,
                "bounds": row.bounds,
                "roi": row.detections[0].roi,
                "language": row.detections[0].language,
                "variant": row.detections[0].variant,
            }
            for value, row in candidates
        ],
        "timeCandidates": [
            {
                "name": p.name,
                "time": p.centiseconds / 100,
                "source": p.source,
                "confidence": p.confidence,
                "roi": p.rows[0].detections[0].roi,
                "language": p.rows[0].detections[0].language,
                "variant": p.rows[0].detections[0].variant,
                "model": p.rows[0].detections[0].model,
                "bounds": [r.bounds for r in p.rows],
            }
            for p in all_pairs
        ],
        "ocrPasses": passes,
    }
    if hangul_retry_passes:
        diagnostics["hangulRereadPasses"] = hangul_retry_passes
    if tolerant_match:
        diagnostics["nameTolerance"] = {
            "observed": tolerant_match.pair.name,
            "matchedAlias": tolerant_match.alias,
            "changes": [dataclasses.asdict(change) for change in tolerant_match.changes],
            "observedConfidence": tolerant_match.pair.confidence,
            "rereadAttempted": True,
            "reason": "targeted_reread_inconclusive",
        }
    if OCR_DEBUG_TEXTS:
        diagnostics["evidence"] = [dataclasses.asdict(d) for d in evidence.deduplicate(observations)]
    logger.info("[ocr] evidence t=%.2fs passes=%d", perf_counter() - started, passes)
    return {
        "extracted": {
            "name": result_name,
            "time": seconds,
            "code": result_code,
            "sources": sources,
            "texts": texts,
            "diagnostics": diagnostics,
        }
    }


# =============================================================================
# FastAPI
# =============================================================================
@dataclass
class _OcrJob:
    img: np.ndarray
    url: str
    fut: "asyncio.Future[dict]"
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
            inference = asyncio.create_task(
                asyncio.to_thread(
                    run_ocr_pipeline,
                    job.img,
                    code=job.code,
                    time=job.time,
                    names=job.names,
                )
            )
            try:
                res = await asyncio.wait_for(asyncio.shield(inference), timeout=OCR_TIMEOUT_S)
            except asyncio.TimeoutError as error:
                if not job.fut.done():
                    job.fut.set_exception(error)
                # Cancelling to_thread does not stop native inference. Hold the
                # queue slot until it exits so the next job cannot overlap it.
                await inference
                continue
            except asyncio.CancelledError:
                await asyncio.shield(inference)
                raise

            run_s = perf_counter() - t0
            logger.info("[ocr] wait=%.2fs run=%.2fs q=%d", wait_s, run_s, q.qsize())

            if not job.fut.done():
                job.fut.set_result(res)

        except asyncio.TimeoutError as e:
            if not job.fut.done():
                job.fut.set_exception(e)
        except Exception as e:
            if not job.fut.done():
                job.fut.set_exception(e)
        finally:
            q.task_done()


async def _enqueue_ocr(
    app: FastAPI,
    img: np.ndarray,
    url: str,
    *,
    code: str,
    time: float | None,
    names: list[str],
) -> dict:
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
    fut: asyncio.Future[dict] = loop.create_future()

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
        logger.info("[fetch] bytes=%d t=%.2fs", len(image_bytes), elapsed)
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

        return JSONResponse(content=res, media_type="application/json; charset=utf-8")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="ocr timeout (queue + run)")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
