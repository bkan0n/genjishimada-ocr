from __future__ import annotations

import base64
import io
import logging
import os
import re
import asyncio
import unicodedata
from collections import defaultdict
from contextlib import asynccontextmanager
from time import perf_counter
from pathlib import Path
from typing import AsyncIterator, Literal, get_args

import aiohttp
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from paddleocr import PaddleOCR
from PIL import Image, ImageFile
from pydantic import BaseModel, ConfigDict, HttpUrl

# =============================================================================
# Runtime / CPU safety
# =============================================================================
os.environ.setdefault("CPU_RUNTIME_CACHE_CAPACITY", "20")
os.environ.setdefault("OPENBLAS_CORETYPE", "NEHALEM")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["FLAGS_use_mkldnn"] = "0"

# Limit per-request OCR time and variant count
OCR_TIMEOUT_S = float(os.environ.get("OCR_TIMEOUT_S", "60"))
FAST_OCR = os.environ.get("FAST_OCR", "1") == "1"
MIN_NAME_LEN = int(os.environ.get("MIN_NAME_LEN", "3"))

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    cv2.setNumThreads(0)
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("genjipk-ocr")

_KERNEL_3 = np.ones((3, 3), np.uint8)
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

PADDLE_WHL_DIR = Path.home() / ".paddleocr" / "whl"

# =============================================================================
# Types / Models
# =============================================================================
LanguageCode = Literal["en", "ch", "korean", "japan"]
RoiLabel = Literal["BL", "BAN", "TR", "TL"]


def to_camel(s: str) -> str:
    """Convert snake_case identifiers to camelCase.

    Args:
      s: Input string in snake_case.

    Returns:
      The camelCase version of the string.
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
ROI_BANNER_TIGHT = [0.240, 0.168, 0.760, 0.380]
ROI_TOPRIGHT = [0.821, 0.077, 0.985, 0.565]
ROI_BOTTOMLEFT = [0.050, 0.825, 0.330, 0.990]

# TOP5 strip inside ROI_TOPRIGHT (to avoid "HOLD ... LEADERBOARD" junk)
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
SUPPORTED_LANGUAGES: tuple[LanguageCode, ...] = ("en", "ch", "korean", "japan")


def _pick_existing_dir(*candidates: Path) -> str | None:
    """Return the first existing path among the candidates.

    Args:
      candidates: Paths to check in priority order.

    Returns:
      The first existing path as a string, or None.
    """
    # Preserve priority by checking in order.
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return None


def _model_dirs_for_language_code(
    language_code: LanguageCode,
) -> tuple[str | None, str | None, str, str | None, str | None]:
    """Resolve model directories and model names for a given language.

    Rules:
    - use PP-OCRv5 mobile models for minimum size
    - keep per-language recognition models when available

    Args:
      language_code: Language identifier.

    Returns:
      Tuple of (det_dir, rec_dir, ocr_version, det_name, rec_name).
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


def _build_ocr_engine(language_code: LanguageCode) -> PaddleOCR:
    """Create and configure a PaddleOCR engine for the language.

    Args:
      language_code: Language identifier.

    Returns:
      Configured PaddleOCR engine.
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
        enable_mkldnn=False,
        cpu_threads=1,
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
    """Warm OCR engines so requests do not pay model load costs.

    Args:
      languages: Languages to warm.

    Returns:
      None.
    """
    # Avoid reloading models that are already initialized.
    for lang in languages:
        if lang in OCR_ENGINES:
            continue
        logger.info(f"📥 Warming PaddleOCR model: {lang} (MKLDNN OFF)")
        OCR_ENGINES[lang] = _build_ocr_engine(lang)
        logger.info(f"✅ Model ready: {lang}")


def get_ocr_engine(language_code: LanguageCode) -> PaddleOCR:
    """Return a warmed OCR engine or raise if unavailable.

    Args:
      language_code: Language identifier.

    Returns:
      Warmed PaddleOCR engine.
    """
    # Fail fast if models are not ready yet.
    engine = OCR_ENGINES.get(language_code)
    if engine is None:
        raise HTTPException(status_code=503, detail=f"OCR model '{language_code}' not loaded")
    return engine


def log_model_dirs() -> None:
    """Log resolved model directories for debugging.

    Args:
      None.

    Returns:
      None.
    """
    # Emit per-language model paths for troubleshooting.
    for lang in ("en", "korean", "japan", "ch"):
        det_dir, rec_dir, ocr_version, det_name, rec_name = _model_dirs_for_language_code(lang)  # type: ignore[arg-type]
        logger.info(
            f"[models] {lang}: ocr={ocr_version} det={det_dir} rec={rec_dir} "
            f"det_name={det_name} rec_name={rec_name}"
        )


# =============================================================================
# Core image utilities
# =============================================================================
def normalize_base64_padding(b64_string: str) -> str:
    """Normalize and pad a base64 string.

    Rules:
    - normalize URL-safe characters
    - remove whitespace
    - pad to a multiple of four

    Args:
      b64_string: Raw base64 string.

    Returns:
      Padded base64 string.
    """
    # Normalize whitespace and URL-safe characters first.
    cleaned = re.sub(RE_SPACES, "", b64_string).replace("-", "+").replace("_", "/").replace(" ", "+")
    missing = (-len(cleaned)) % 4
    return cleaned + ("=" * missing if missing else "")


def decode_base64_image(image_b64: str) -> np.ndarray:
    """Decode a base64 data URL into a BGR OpenCV image.

    Args:
      image_b64: Base64 string or data URL.

    Returns:
      OpenCV BGR image.
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
        # Load with Pillow to support more formats.
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.load()
        pil_image = pil_image.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image stream: {e}")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR OpenCV image.

    Args:
      image_bytes: Raw image bytes.

    Returns:
      OpenCV BGR image.
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
    """Crop an image by fractional ROI coordinates.

    Args:
      image: Source image.
      roi_frac: Fractional ROI [x1, y1, x2, y2].

    Returns:
      Cropped image.
    """
    # Convert fractional ROI to pixel coordinates.
    h, w = image.shape[:2]
    x1 = int(w * roi_frac[0])
    y1 = int(h * roi_frac[1])
    x2 = int(w * roi_frac[2])
    y2 = int(h * roi_frac[3])
    return image[max(y1, 0) : min(y2, h), max(x1, 0) : min(x2, w)].copy()


def crop_within(parent_crop: np.ndarray, rel_roi: list[float]) -> np.ndarray:
    """Crop a sub-ROI within an already cropped image.

    Args:
      parent_crop: Parent crop image.
      rel_roi: Fractional ROI relative to the parent crop.

    Returns:
      Cropped image.
    """
    # Delegate to the generic fractional cropper.
    return crop_by_frac_roi(parent_crop, rel_roi)


# =============================================================================
# Pre-processing
# =============================================================================
def enhance_contrast_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Convert to grayscale and enhance local contrast.

    Args:
      image_bgr: Input BGR image.

    Returns:
      Enhanced grayscale image.
    """
    # CLAHE improves local contrast before thresholding.
    g = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    g = _CLAHE.apply(g)
    return cv2.GaussianBlur(g, (3, 3), 0)


def mask_white_regions(image_bgr: np.ndarray) -> np.ndarray:
    """Create a mask for bright white HUD elements.

    Args:
      image_bgr: Input BGR image.

    Returns:
      Binary mask image.
    """
    # HSV thresholding isolates near-white text.
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 190], np.uint8), np.array([179, 70, 255], np.uint8))
    mask = cv2.medianBlur(mask, 3)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL_3, 1)  # type: ignore


def mask_cyan_regions(image_bgr: np.ndarray) -> np.ndarray:
    """Create a mask for saturated cyan UI accents.

    Args:
      image_bgr: Input BGR image.

    Returns:
      Binary mask image.
    """
    # Hue range covers bright cyan overlays.
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([85, 35, 70], np.uint8),
        np.array([130, 255, 255], np.uint8),
    )
    mask = cv2.medianBlur(mask, 3)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL_3, 1)  # type: ignore


def mask_hud_cyan_regions(image_bgr: np.ndarray) -> np.ndarray:
    """Create a mask for pale cyan HUD text.

    Args:
      image_bgr: Input BGR image.

    Returns:
      Binary mask image.
    """
    # HUD name text is often pale cyan with low saturation; use a softer S cutoff.
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([80, 10, 110], np.uint8),
        np.array([135, 255, 255], np.uint8),
    )
    mask = cv2.medianBlur(mask, 3)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL_3, 1)  # type: ignore


def unsharp(image_bgr: np.ndarray, amount: float = 1.6, sigma: float = 1.0) -> np.ndarray:
    """Apply a simple unsharp mask for edge emphasis.

    Args:
      image_bgr: Input BGR image.
      amount: Sharpening amount.
      sigma: Blur sigma for the mask.

    Returns:
      Sharpened image.
    """
    # Sharpen by subtracting a blurred version.
    blur = cv2.GaussianBlur(image_bgr, (0, 0), sigma)
    return cv2.addWeighted(image_bgr, amount, blur, -(amount - 1.0), 0)


def upscale(image_bgr: np.ndarray, scale: float) -> np.ndarray:
    """Upscale an image with cubic interpolation.

    Args:
      image_bgr: Input BGR image.
      scale: Upscale factor.

    Returns:
      Upscaled image.
    """
    # Cubic interpolation preserves edges for OCR.
    return cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def build_cjk_variants(roi_bgr: np.ndarray) -> list[np.ndarray]:
    """Generate multiple preprocessing variants for CJK OCR.

    Rules:
    - when FAST_OCR is enabled, return a smaller variant set

    Args:
      roi_bgr: ROI image in BGR.

    Returns:
      List of variant images.
    """
    # Return early for empty inputs.
    if roi_bgr is None or roi_bgr.size == 0:
        return []

    variants: list[np.ndarray] = []
    base = roi_bgr

    # Baseline variant.
    variants.append(base)

    h, w = base.shape[:2]
    scale = 2.8 if min(h, w) < 160 else 2.0
    up = upscale(base, scale)
    up = unsharp(up, amount=1.7, sigma=1.0)
    # Upscale + sharpen helps distinguish tight strokes.
    variants.append(up)

    wmask = mask_white_regions(base)
    # White masks capture bright overlay text.
    variants.append(wmask)

    cmask = mask_cyan_regions(base)
    # Cyan masks capture UI text in blue hues.
    variants.append(cmask)

    g = enhance_contrast_grayscale(base)
    if FAST_OCR:
        variants.append(g)
        return variants

    variants.append(255 - wmask)
    variants.append(255 - cmask)

    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
    # Thresholded variants improve high-contrast strokes.
    variants.append(thr)
    variants.append(255 - thr)

    return variants


def build_cjk_name_variants(roi_bgr: np.ndarray) -> list[np.ndarray]:
    """Generate extra CJK variants tuned for HUD names.

    Rules:
    - when FAST_OCR is enabled, skip inverted masks and dilations

    Args:
      roi_bgr: ROI image in BGR.

    Returns:
      List of variant images.
    """
    # Start from the generic CJK variants.
    variants = build_cjk_variants(roi_bgr)
    if roi_bgr is None or roi_bgr.size == 0:
        return variants

    h, w = roi_bgr.shape[:2]
    if min(h, w) < 180:
        # Stretch vertically to better separate stacked strokes.
        stretch = cv2.resize(roi_bgr, None, fx=2.4, fy=3.2, interpolation=cv2.INTER_CUBIC)
        stretch = unsharp(stretch, amount=1.7, sigma=1.0)
        variants.append(stretch)

    hud = mask_hud_cyan_regions(roi_bgr)
    # Add HUD-specific cyan masks.
    variants.append(hud)
    if FAST_OCR:
        return variants

    variants.append(255 - hud)
    variants.append(cv2.dilate(hud, _KERNEL_3, iterations=1))
    return variants


# =============================================================================
# OCR wrapper
# =============================================================================
def ocr_lines(image: np.ndarray, language_code: LanguageCode) -> list[tuple[str, float]]:
    """Run OCR on an image and return text/confidence lines.

    Args:
      image: Input image.
      language_code: OCR language code.

    Returns:
      List of (text, confidence) tuples.
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
        logger.warning(f"OCR({language_code}) failed: {e}")
        return []

    out: list[tuple[str, float]] = []

    def _add_text(text: str | None, score: float | None) -> None:
        """Append a normalized text + score pair if valid."""
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
        """Extract OCR lines from a PaddleOCR 3.x result dict."""
        if not isinstance(data, dict):
            return

        # Direct rec_texts / rec_scores arrays.
        rec_texts = data.get("rec_texts")
        rec_scores = data.get("rec_scores")
        if isinstance(rec_texts, list):
            if isinstance(rec_scores, list):
                for text, score in zip(rec_texts, rec_scores):
                    _add_text(text, score)
            else:
                for text in rec_texts:
                    _add_text(text, None)

        # Direct single record.
        if isinstance(data.get("rec_text"), str) or data.get("rec_text") is not None:
            _add_text(data.get("rec_text"), data.get("rec_score"))

        # Some pipelines return a flat "text"/"score".
        if isinstance(data.get("text"), str) or data.get("text") is not None:
            _add_text(data.get("text"), data.get("score") or data.get("rec_score"))

        # Nested structures from OCR pipelines.
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
    """Join OCR lines into a single normalized string.

    Args:
      lines: List of (text, confidence) tuples.

    Returns:
      Joined string.
    """
    # Drop confidences and keep order.
    return " ".join([t for t, _ in lines]).strip()


# =============================================================================
# Script profiling + scoring
# =============================================================================
def remove_all_whitespace(text: str) -> str:
    """Remove all whitespace from a string.

    Args:
      text: Input string.

    Returns:
      String without whitespace.
    """
    # Normalize spacing for comparisons.
    return re.sub(RE_SPACES, "", text or "")


def count_cjk(text: str) -> int:
    """Count CJK characters in a string.

    Args:
      text: Input string.

    Returns:
      Number of CJK characters.
    """
    # Use the precompiled CJK regex for speed.
    return len(RE_CJK_CHAR.findall(text or ""))


def fraction_of_unicode_class(unicode_class_pattern: str, text: str) -> float:
    """Return the fraction of characters matching a Unicode class.

    Args:
      unicode_class_pattern: Character class regex.
      text: Input string.

    Returns:
      Fraction of matching characters.
    """
    # Compare against a whitespace-stripped view of the string.
    compact = remove_all_whitespace(text)
    return 0.0 if not compact else len(re.findall(f"[{unicode_class_pattern}]", compact)) / len(compact)


def build_script_profile(text: str) -> ScriptProfile:
    """Compute a script profile for Hangul/Kana/Han/Latin ratios.

    Args:
      text: Input string.

    Returns:
      ScriptProfile with ratios.
    """
    # Ratios are used for script-aware scoring.
    return ScriptProfile(
        hangul=fraction_of_unicode_class(_HANGUL, text),
        kana=fraction_of_unicode_class(_HIRAKATA, text),
        han=fraction_of_unicode_class(_HAN, text),
        latin=fraction_of_unicode_class(_LATIN, text),
    )


def expected_script_for_language(language_code: str) -> str:
    """Return the expected script name for a language.

    Args:
      language_code: Language identifier.

    Returns:
      Script name string.
    """
    # Fall back to Latin for unknown language codes.
    return {"korean": "hangul", "japan": "kana", "ch": "han", "en": "latin"}.get(language_code, "latin")


def roi_label_weight(roi: RoiLabel) -> float:
    """Assign a weight based on ROI reliability.

    Args:
      roi: ROI label.

    Returns:
      Weight as a float.
    """
    # Bottom-left is most reliable for names.
    return {"BL": 0.35, "TR": 0.25, "BAN": 0.10, "TL": 0.05}.get(roi, 0.0)


def normalize_banner_fragment(fragment_text: str) -> str:
    """Normalize banner OCR fragments for parsing.

    Args:
      fragment_text: Raw banner fragment.

    Returns:
      Normalized banner text.
    """
    # Collapse noisy separators and trim punctuation.
    return re.sub(RE_CLEAN_BANNER_FRAGMENT, " ", (fragment_text or "")).strip(" :|~!.,*_-").strip()


def ocr_with_labels(image: np.ndarray, language_code: LanguageCode, roi_label: RoiLabel) -> list["OcrCandidate"]:
    """Run OCR and attach language/ROI metadata to each candidate.

    Args:
      image: Input image.
      language_code: OCR language code.
      roi_label: ROI label.

    Returns:
      List of OcrCandidate.
    """
    # Build structured candidates for scoring.
    out: list[OcrCandidate] = []
    for text, conf in ocr_lines(image, language_code):
        out.append(
            OcrCandidate(
                text=text.strip(),
                confidence=float(conf or 0.0),
                language_code=language_code,
                roi_label=roi_label,
                profile=build_script_profile(text),
            )
        )
    return out


def _cjk_best_substring_min(text: str, min_len: int) -> str | None:
    """Extract the longest contiguous CJK substring with a minimum length.

    Args:
      text: Input string.
      min_len: Minimum accepted length.

    Returns:
      Longest CJK substring or None.
    """
    # Prefer the longest CJK span as the candidate.
    if not text:
        return None
    best = ""
    for m in RE_CJK_SEQ.finditer(text):
        seq = m.group(1) or ""
        if len(seq) > len(best):
            best = seq
    return best if len(best) >= min_len else None


def _cjk_best_substring(text: str) -> str | None:
    """Extract the longest contiguous CJK substring.

    Args:
      text: Input string.

    Returns:
      Longest CJK substring or None.
    """
    return _cjk_best_substring_min(text, 2)


# =============================================================================
# ASCII normalization helpers
# =============================================================================
def _clean_ascii_token(raw: str) -> str:
    """Normalize ASCII tokens and reduce OCR noise.

    Args:
      raw: Raw OCR token.

    Returns:
      Cleaned token.
    """
    # Remove accents/diacritics before ASCII cleanup.
    norm = unicodedata.normalize("NFKD", (raw or ""))
    norm = "".join(ch for ch in norm if not unicodedata.combining(ch))

    # Uppercase and remove non-alphanumerics first.
    s = re.sub(r"[^A-Z0-9_]", "", norm.upper()).strip("_")
    if not s:
        return ""

    # Strip huge digit tails that look like scores.
    m_tail = re.search(r"\d{6,}$", s)
    if m_tail:
        prefix = s[: m_tail.start()]
        if len(prefix) >= 3 and sum(ch.isalpha() for ch in prefix) >= 2:
            s = prefix

    # "polluted" heuristic (badge + OCR confusions)
    digit_count = sum(ch.isdigit() for ch in s)
    polluted = (len(s) >= 12 and digit_count >= 3)

    if polluted:
        # Normalize look-alike digits only for polluted cases.
        s_for_prefix = s.replace("1", "I")
        for p in _ROMAN_PREFIXES:
            if s_for_prefix.startswith(p) and (len(s_for_prefix) - len(p)) >= 3:
                s = s[len(p) :]
                break

        s = (
            s.replace("0", "O")
            .replace("1", "I")
            .replace("5", "S")
            .replace("8", "B")
            .replace("2", "Z")
        )

    return s


def _strip_rank_prefix_ascii(name: str) -> str:
    """
    Strip roman numeral rank prefixes when safe.

    Rules:
    - prefix is roman numeral (or OCR '1' -> 'I')
    - suffix length is at least 3 characters
    - suffix matches the ASCII name regex

    Args:
      name: Raw ASCII name token.

    Returns:
      Cleaned name token.
    """
    # Work on a normalized token.
    s = _clean_ascii_token(name)
    if not s:
        return s

    s_for_prefix = s.replace("1", "I")
    for p in _ROMAN_PREFIXES:
        if s_for_prefix.startswith(p):
            suffix = s[len(p) :]
            # avoid stripping "IVAN" (suffix too short)
            if len(suffix) >= 3 and re.fullmatch(RE_ASCII_NAME_MATCH, suffix):
                return suffix
            return s
    return s


def _strip_rank_prefix_ascii_with_top5_hint(name: str, top5_text: str | None) -> str:
    """
    Strip rank prefix only if TOP5 confirms the suffix.

    Args:
      name: Raw ASCII name token.
      top5_text: OCR text from TOP5 block.

    Returns:
      Cleaned name token.
    """
    # Strip prefix only if TOP5 confirms the suffix.
    base = _strip_rank_prefix_ascii(name)
    if not top5_text:
        return base

    # if base didn't change -> nothing to do
    if base == _clean_ascii_token(name):
        # it means no prefix stripped; still keep base
        return base

    up = (top5_text or "").upper()
    if re.search(rf"\b{re.escape(base)}\b", up):
        return base

    # If TOP5 doesn't confirm, keep original cleaned name
    return _clean_ascii_token(name)


def _normalize_ascii_for_compare(name: str) -> str:
    """Normalize ASCII strings for fuzzy comparison.

    Args:
      name: Input name string.

    Returns:
      Normalized string.
    """
    # Map common OCR digit/letter confusions.
    s = (name or "").upper()
    s = (
        s.replace("0", "O")
        .replace("1", "I")
        .replace("5", "S")
        .replace("8", "B")
        .replace("2", "Z")
    )
    return re.sub(r"[^A-Z0-9_]", "", s)


def _name_variants_ascii(name: str) -> set[str]:
    """Generate alternate ASCII variants for matching.

    Args:
      name: Input name string.

    Returns:
      Set of normalized variants.
    """
    # Include variants without roman prefixes.
    s = _normalize_ascii_for_compare(name)
    out = {s}
    for p in _ROMAN_PREFIXES:
        if s.startswith(p) and (len(s) - len(p)) >= 3:
            out.add(s[len(p) :])
            break
    return {v for v in out if v}


def _bigrams(s: str) -> set[str]:
    """Create a set of bigrams for similarity scoring.

    Args:
      s: Input string.

    Returns:
      Set of bigrams.
    """
    # Use a compact string to avoid whitespace noise.
    s = remove_all_whitespace(s)
    if len(s) < 2:
        return {s} if s else set()
    return {s[i : i + 2] for i in range(len(s) - 1)}


def _sim(a: str, b: str) -> float:
    """Compute Jaccard similarity over bigrams.

    Args:
      a: First string.
      b: Second string.

    Returns:
      Similarity score between 0 and 1.
    """
    # Compare bigram sets for a simple similarity metric.
    A = _bigrams(a)
    B = _bigrams(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    uni = len(A | B)
    return inter / uni if uni else 0.0


def _toggle_hangul_tense(ch: str) -> str | None:
    """Toggle a Hangul syllable's tense initial consonant if possible.

    Args:
      ch: Hangul syllable.

    Returns:
      Toggled Hangul syllable or None if unchanged.
    """
    # Only handle precomposed Hangul syllables.
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
    """Generate Hangul variants by toggling tense initials.

    Args:
      text: Input Hangul string.
      max_variants: Maximum number of variants to generate.

    Returns:
      List of unique variants including the original.
    """
    # Build a list of positions that can be toggled.
    positions: list[tuple[int, str]] = []
    for idx, ch in enumerate(text):
        toggled = _toggle_hangul_tense(ch)
        if toggled:
            positions.append((idx, toggled))

    if not positions:
        return [text]

    variants: set[str] = set()

    def _walk(i: int, current: list[str]) -> None:
        """Recursive builder for variant combinations."""
        if len(variants) >= max_variants:
            return
        if i >= len(positions):
            variants.add("".join(current))
            return
        pos, toggled = positions[i]
        # Keep original.
        _walk(i + 1, current)
        # Toggle this position.
        current2 = current.copy()
        current2[pos] = toggled
        _walk(i + 1, current2)

    _walk(0, list(text))
    variants.add(text)
    return list(variants)


def _pick_hangul_variant(name: str, evidence: list[str]) -> str:
    """Pick a Hangul variant that best matches evidence tokens.

    Args:
      name: Base Hangul name.
      evidence: List of OCR-derived CJK tokens from other regions.

    Returns:
      Best matching Hangul variant.
    """
    # If no evidence, keep the original.
    if not evidence:
        return name

    variants = _hangul_tense_variants(name)
    if len(variants) <= 1:
        return name

    def _score_variant(v: str) -> float:
        """Score variant by max similarity to evidence tokens."""
        return max((_sim(v, e) for e in evidence), default=0.0)

    base_score = _score_variant(name)
    best = name
    best_score = base_score
    for v in variants:
        if v == name:
            continue
        score = _score_variant(v)
        if score > best_score + 0.05:
            best_score = score
            best = v
    return best


def _score_name_candidate(c: OcrCandidate, cleaned_text: str) -> float:
    """Score a candidate name using script profile and confidence.

    Args:
      c: OCR candidate with metadata.
      cleaned_text: Cleaned candidate text.

    Returns:
      Score as a float.
    """
    # Favor candidates that match the expected script.
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
    """Pick the best name by clustering similar candidates.

    Args:
      cands: List of (name, score) candidates.

    Returns:
      Best name or None.
    """
    # Cluster candidates by similarity and pick the strongest group.
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
# NAME: Bottom-left extraction (source of truth)
# =============================================================================
def extract_name_from_bottom_left(
    bl_name_roi: np.ndarray,
    bl_alt_roi: np.ndarray,
) -> str | None:
    """Extract player name from bottom-left HUD regions.

    Args:
      bl_name_roi: Tight name ROI.
      bl_alt_roi: Alternate bottom-left ROI (unused for name extraction).

    Returns:
      Extracted name or None.
    """
    # Bottom-left is the primary source of truth for names.
    if bl_name_roi is None or bl_name_roi.size == 0:
        return None

    # Only use the tight name ROI to avoid HUD pollution.
    name_rois = [bl_name_roi]

    # ---- ASCII candidates (English OCR) ----
    ascii_scores: dict[str, float] = {}
    for roi in name_rois:
        if roi is None or roi.size == 0:
            continue
        lines = ocr_lines(roi, "en")
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

            digit_count = sum(ch.isdigit() for ch in tok)
            letter_count = sum(ch.isalpha() for ch in tok)
            if digit_count > 0 and letter_count < 4:
                continue

            if not re.fullmatch(RE_ASCII_NAME_MATCH, tok):
                continue

            score = avg_conf + 0.10 + max(0.0, (len(tok) - MIN_NAME_LEN) * 0.05)
            ascii_scores[tok] = ascii_scores.get(tok, 0.0) + score

    # ---- CJK candidates (Korean first, multi-variant) ----
    cjk_candidates: list[tuple[str, float]] = []

    def _collect_cjk_for_lang(lang: LanguageCode) -> None:
        """Collect CJK candidates for a specific language.

        Args:
          lang: Language code.

        Returns:
          None.
        """
        # Iterate over both ROI variants and preprocessing variants.
        for roi in name_rois:
            if roi is None or roi.size == 0:
                continue
            for v in build_cjk_name_variants(roi):
                for cand in ocr_with_labels(v, lang, "BL"):
                    cjk = _cjk_best_substring_min(cand.text, MIN_NAME_LEN)
                    if not cjk:
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
                    score = _score_name_candidate(cand, cjk)
                    cjk_candidates.append((cjk, score))

    _collect_cjk_for_lang("korean")

    # Only expand to JP/CH if Korean signals are weak.
    strong_korean = any(build_script_profile(n).hangul >= 0.55 for n, _ in cjk_candidates)
    if not strong_korean and not FAST_OCR:
        _collect_cjk_for_lang("japan")
        _collect_cjk_for_lang("ch")

    picked_cjk = _consensus_pick(cjk_candidates)
    picked_ascii = None
    if ascii_scores:
        picked_ascii = max(
            ascii_scores.items(),
            key=lambda kv: (kv[1], len(kv[0]), sum(ch.isalpha() for ch in kv[0])),
        )[0]

    if picked_cjk and count_cjk(picked_cjk) >= MIN_NAME_LEN:
        # Refine Hangul tense using only bottom-left evidence.
        evidence = [name for name, _ in cjk_candidates]
        return _pick_hangul_variant(picked_cjk, evidence)

    return picked_ascii


# =============================================================================
# NAME: Banner extraction (to decide if banner time is valid)
# =============================================================================
def extract_name_from_banner(text_banner: str) -> str | None:
    """Extract player name from the banner text line.

    Args:
      text_banner: OCR text from the banner.

    Returns:
      Extracted name or None.
    """
    # Banner is used as a fallback validation source.
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
    """Check whether two names likely refer to the same player.

    Args:
      a: First name.
      b: Second name.

    Returns:
      True if they match, otherwise False.
    """
    # Use script-aware logic for CJK vs ASCII.
    if not a or not b:
        return False

    if count_cjk(a) >= 2 or count_cjk(b) >= 2:
        aa = remove_all_whitespace(a)
        bb = remove_all_whitespace(b)
        if aa == bb:
            return True
        if len(aa) >= 3 and (aa in bb or bb in aa):
            return True
        return _sim(aa, bb) >= 0.78

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
    """Parse a noisy OCR numeric token into a float if possible.

    Args:
      raw_token: Raw numeric token.

    Returns:
      Parsed float or None.
    """
    # Normalize common OCR misreads before parsing.
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
    """Extract a time value from the banner text.

    Args:
      text: OCR text from the banner.

    Returns:
      Time in seconds or None.
    """
    # Normalize OCR noise to improve numeric parsing.
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
    )
    text = re.sub(RE_SPACES, " ", text).strip()

    time_idx = text.find("TIME")
    if time_idx != -1:
        # Prefer numbers near the TIME keyword.
        window = text[time_idx : time_idx + 90]
        window = re.sub(r"([0-9OQDBZGISL]{1,5})\s+([0-9OQDBZGISL]{1}\.\d{2})", r"\1\2", window)
        m = re.search(RE_PARSE_BANNER_TIME_SEARCH_WITH_SEC, window)
        if m:
            v = parse_loose_numeric_token(m.group(1))
            if v is not None:
                return v

    best: tuple[int, float] | None = None
    for m in re.finditer(RE_PARSE_BANNER_TIME_SEARCH_NO_SEC, text):
        # Score candidates by proximity to TIME and SEC.
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
    """Extract a time from the top-left HUD block.

    Args:
      text_top_left: OCR text from top-left.
      text_top_left_white: OCR text from white mask.

    Returns:
      Time in seconds or None.
    """
    # Use white mask OCR first for higher precision.
    src = f"{text_top_left_white or ''} {text_top_left or ''}".upper()
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


def extract_top5_text(top_right_crop: np.ndarray) -> tuple[str, str]:
    """Extract TOP5 text and a debug OCR line.

    Args:
      top_right_crop: Top-right ROI image.

    Returns:
      Tuple of (top5_text, debug_full_text).
    """
    # Focus on the TOP5 strips to avoid unrelated HUD hints.
    if top_right_crop is None or top_right_crop.size == 0:
        return "", ""

    tr1 = crop_within(top_right_crop, ROI_TR_TOP5_STRIP_1)
    tr2 = crop_within(top_right_crop, ROI_TR_TOP5_STRIP_2)

    top5_lines: list[tuple[str, float]] = []
    for strip in (tr1, tr2):
        if strip is None or strip.size == 0:
            continue

        top5_lines.extend(ocr_lines(strip, "en"))

        for v in build_cjk_name_variants(strip):
            top5_lines.extend(ocr_lines(v, "korean"))

        text_k = join_lines(top5_lines)
        if count_cjk(text_k) < 2 and not FAST_OCR:
            for v in build_cjk_name_variants(strip):
                top5_lines.extend(ocr_lines(v, "japan"))
                top5_lines.extend(ocr_lines(v, "ch"))

    top5_text = join_lines(top5_lines)

    dbg_lines: list[tuple[str, float]] = []
    dbg_lines.extend(ocr_lines(top_right_crop, "en"))
    dbg_full = join_lines(dbg_lines)

    return top5_text, dbg_full




def extract_time_from_top5(top5_text: str, bl_name: str | None) -> float | None:
    """Extract the player time from the TOP5 leaderboard.

    Args:
      top5_text: OCR text from TOP5 block.
      bl_name: Bottom-left name.

    Returns:
      Time in seconds or None.
    """
    # Match times by name, falling back to similarity.
    if not top5_text or not bl_name:
        return None

    upper_full = (top5_text or "").upper()
    m_top5 = RE_TOP5_SECTION.search(upper_full)
    block = upper_full[m_top5.start() :] if m_top5 else upper_full
    block = re.sub(r"(?<!\d)(\d{1,5})\s+(\d{2})(?!\d)", r"\1.\2", block)

    def _to_float(s: str) -> float | None:
        """Parse a numeric string to float safely.

        Args:
          s: Numeric string.

        Returns:
          Float or None.
        """
        # Normalize decimal separators before parsing.
        try:
            return float((s or "").replace(",", "."))
        except Exception:
            return None

    entries: list[tuple[int, str, float]] = []

    for m in re.finditer(RE_TOP5_TIME_FOR_NAME_ASCII, block):
        nm = (m.group(1) or "").upper()
        nm = _strip_rank_prefix_ascii(nm)
        if not nm or nm in _GENERIC_ASCII:
            continue
        t = _to_float(m.group(2))
        if t is None:
            continue
        entries.append((m.start(), nm, t))

    block_raw = top5_text
    m2 = RE_TOP5_SECTION.search(block_raw)
    block_raw = block_raw[m2.start() :] if m2 else block_raw
    block_raw = re.sub(r"(?<!\d)(\d{1,5})\s+(\d{2})(?!\d)", r"\1.\2", block_raw)

    for m in re.finditer(RE_TOP5_TIME_FOR_NAME_CJK, block_raw):
        nm = _cjk_best_substring(m.group(1) or "")
        if not nm or count_cjk(nm) < 2:
            continue
        t = _to_float(m.group(2))
        if t is None:
            continue
        entries.append((m.start(), nm, t))

    if not entries:
        return None

    entries.sort(key=lambda x: x[0])

    if count_cjk(bl_name) >= 2:
        target = remove_all_whitespace(bl_name)
        best: tuple[int, float] | None = None
        for pos, nm, t in entries:
            if count_cjk(nm) < 2:
                continue
            cand = remove_all_whitespace(nm)
            if cand == target or (len(target) >= 3 and (target in cand or cand in target)) or _sim(target, cand) >= 0.78:
                if best is None or t > best[1] + 1e-2 or (abs(t - best[1]) <= 1e-2 and pos < best[0]):
                    best = (pos, t)
        return best[1] if best else None

    target_vars = _name_variants_ascii(bl_name)
    best_choice: tuple[int, int, float] | None = None
    for pos, nm, t in entries:
        if count_cjk(nm) >= 2:
            continue
        cand_vars = _name_variants_ascii(nm)

        score = 0
        if target_vars & cand_vars:
            score = 3
        else:
            if any((cv in tv or tv in cv) and min(len(cv), len(tv)) >= 4 for tv in target_vars for cv in cand_vars):
                score = 2

        if score <= 0:
            continue

        if best_choice is None or score > best_choice[0] or (score == best_choice[0] and t > best_choice[2] + 1e-2):
            best_choice = (score, pos, t)

    return best_choice[2] if best_choice else None


# =============================================================================
# TIME: final selection (your decision tree)
# =============================================================================
def pick_final_time(
    bl_name: str | None,
    banner_name: str | None,
    banner_time: float | None,
    top5_time: float | None,
    top_left_time: float | None,
) -> float | None:
    """Pick the best time candidate using the decision tree.

    Args:
      bl_name: Bottom-left name.
      banner_name: Banner name.
      banner_time: Banner time candidate.
      top5_time: TOP5 time candidate.
      top_left_time: Top-left time candidate.

    Returns:
      Final time in seconds or None.
    """
    # Validate and prioritize sources based on reliability.
    def _valid(t: float | None) -> float | None:
        """Validate and clamp a parsed time value.

        Args:
          t: Parsed time value.

        Returns:
          Validated time or None.
        """
        # Discard implausible values.
        if t is None:
            return None
        try:
            v = float(t)
        except Exception:
            return None
        if v < 30.0:
            return None
        if v > 15360:
            return None
        return round(v, 2)

    bt = _valid(banner_time)
    t5 = _valid(top5_time)
    tl = _valid(top_left_time)

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
    """Normalize and validate a candidate map code string.

    Rules:
    - 4-6 alphanumeric characters
    - must contain at least one digit when required
    - common HUD words and known non-codes are rejected

    Args:
      raw_code_text: Raw OCR string for a possible map code.
      require_digit: Whether a digit is required.

    Returns:
      Cleaned map code if valid, otherwise None.
    """
    # Early exits for empty or noisy inputs.
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
    """Extract the map code using heuristic passes.

    Args:
      top_left_text: OCR text from top-left.
      top_left_white_text: OCR text from white mask.
      top_left_cyan_text: OCR text from cyan mask.

    Returns:
      Map code or None.
    """
    # Combine all OCR sources before matching.
    all_text = " ".join([top_left_text or "", top_left_white_text or "", top_left_cyan_text or ""]).upper()
    normalized = re.sub(RE_MAP_CODE_NORMALIZATION, "MAP CODE", all_text)

    m_keyword = re.search(RE_CODE_KEYWORD_EXTRACT, normalized)
    if m_keyword:
        cand = normalize_map_code(m_keyword.group(1) or "", require_digit=False)
        if cand:
            return cand

    colon_candidates: dict[str, int] = {}
    for m in re.finditer(RE_CODE_AFTER_COLON, normalized):
        token = m.group(1) or ""
        cand = normalize_map_code(token, require_digit=False)
        if not cand:
            continue
        if not any(ch.isalpha() for ch in cand):
            continue
        colon_candidates[cand] = colon_candidates.get(cand, 0) + 1

    if colon_candidates:
        best, _ = max(colon_candidates.items(), key=lambda kv: kv[1])
        return best

    scores_all: dict[str, float] = defaultdict(float)
    scores_with_letter: dict[str, float] = defaultdict(float)

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
        if has_letter:
            scores_with_letter[cand] += score

    if not scores_all:
        return None

    pool = scores_with_letter if scores_with_letter else scores_all
    best_code, _ = max(pool.items(), key=lambda kv: (kv[1], sum(c.isdigit() for c in kv[0])))
    return best_code


# =============================================================================
# OCR pipeline (sync)
# =============================================================================
def run_ocr_pipeline(img: np.ndarray) -> ApiResponse:
    """Run the full OCR pipeline on a decoded image.

    Args:
      img: Decoded BGR image.

    Returns:
      ApiResponse with extracted fields.
    """
    # Measure end-to-end OCR time for diagnostics.
    t0 = perf_counter()

    # ---- crops ----
    top_left = crop_by_frac_roi(img, ROI_TOPLEFT)
    top_left_wide = crop_by_frac_roi(img, ROI_TOPLEFT_WIDE)
    banner = crop_by_frac_roi(img, ROI_BANNER_TIGHT)
    top_right = crop_by_frac_roi(img, ROI_TOPRIGHT)
    bottom_left = crop_by_frac_roi(img, ROI_BOTTOMLEFT)
    bottom_left_name = crop_by_frac_roi(img, ROI_BOTTOMLEFT)

    # ---- TOP LEFT OCR (code + fallback time) ----
    tl_lines = []
    tl_lines.extend(ocr_lines(top_left, "en"))
    tl_lines.extend(ocr_lines(top_left_wide, "en"))
    text_top_left_en = join_lines(tl_lines)

    tl_white_mask = mask_white_regions(top_left_wide)
    tl_cyan_mask = mask_cyan_regions(top_left_wide)
    text_top_left_white_en = join_lines(ocr_lines(tl_white_mask, "en")) if tl_white_mask is not None else ""
    text_top_left_cyan_en = join_lines(ocr_lines(tl_cyan_mask, "en")) if tl_cyan_mask is not None else ""

    code = extract_code(text_top_left_en, text_top_left_white_en, text_top_left_cyan_en)

    # ---- BL debug + name (source of truth) ----
    bl_debug = join_lines(ocr_lines(bottom_left, "en"))
    bl_name_raw = extract_name_from_bottom_left(bottom_left_name, bottom_left)

    # ---- TOP5 OCR (needed for rank-prefix normalization + timing) ----
    top5_text, tr_debug_full = extract_top5_text(top_right)

    # FIX: if BL name contains rank prefix, prefer the stripped version ONLY if TOP5 confirms it
    bl_name = bl_name_raw
    if bl_name and count_cjk(bl_name) == 0:
        bl_name = _strip_rank_prefix_ascii_with_top5_hint(bl_name, top5_text)

    # ---- Banner OCR ----
    banner_white = mask_white_regions(banner)
    banner_gray = enhance_contrast_grayscale(banner)
    banner_binary = cv2.adaptiveThreshold(
        banner_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )

    banner_lines = []
    banner_lines.extend(ocr_lines(banner, "en"))
    banner_lines.extend(ocr_lines(banner_white, "en"))
    banner_lines.extend(ocr_lines(banner_binary, "en"))
    banner_lines.extend(ocr_lines(banner, "korean"))
    banner_lines.extend(ocr_lines(banner_white, "korean"))
    banner_lines.extend(ocr_lines(banner_binary, "korean"))
    banner_lines.extend(ocr_lines(banner_gray, "korean"))
    if not FAST_OCR:
        banner_lines.extend(ocr_lines(banner, "japan"))
        banner_lines.extend(ocr_lines(banner, "ch"))
    text_banner = normalize_banner_fragment(join_lines(banner_lines))

    banner_name = extract_name_from_banner(text_banner)
    banner_time = extract_banner_time_seconds(text_banner)

    # ---- TOP5 time for that name ----
    top5_time = extract_time_from_top5(top5_text, bl_name)

    # ---- Top-left fallback ----
    top_left_time = extract_time_from_top_left(text_top_left_en, text_top_left_white_en)

    seconds = pick_final_time(
        bl_name=bl_name,
        banner_name=banner_name,
        banner_time=banner_time,
        top5_time=top5_time,
        top_left_time=top_left_time,
    )

    # ---- Name is always sourced from bottom-left ----
    final_name = bl_name

    top_right_debug = ""
    if tr_debug_full:
        top_right_debug += f"FULL: {tr_debug_full} "
    if top5_text:
        top_right_debug += f"TOP5: {top5_text}"
    top_right_debug = top_right_debug.strip()

    elapsed = perf_counter() - t0
    logger.info(f"[ocr] done t={elapsed:.2f}s fast={FAST_OCR}")

    return ApiResponse(
        extracted=ExtractedResult(
            name=final_name,
            time=seconds,
            code=code,
            texts=ExtractedTexts(
                top_left=text_top_left_en,
                top_left_white=text_top_left_white_en,
                top_left_cyan=text_top_left_cyan_en,
                banner=text_banner,
                top_right=top_right_debug,
                bottom_left=bl_debug,
            ),
        )
    )


# =============================================================================
# FastAPI
# =============================================================================
class ImageURLPayload(BaseModel):
    image_url: HttpUrl


@asynccontextmanager
async def warm_models_on_startup(app: FastAPI) -> AsyncIterator:
    """Warm models and create the HTTP session on startup.

    Args:
      app: FastAPI application.

    Returns:
      Async iterator for lifespan.
    """
    # Load models once and reuse the shared HTTP client.
    log_model_dirs()
    warm_ocr_engines()
    # Set a sane timeout to avoid hanging on slow or blocked URLs.
    timeout = aiohttp.ClientTimeout(total=20, connect=5, sock_read=15)
    app.state.http_session = aiohttp.ClientSession(timeout=timeout)
    yield
    await app.state.http_session.close()


app = FastAPI(title="GenjiPK OCR", lifespan=warm_models_on_startup)


@app.get("/ping")
def ping() -> dict:
    """Health check endpoint for warmed models.

    Args:
      None.

    Returns:
      Status payload with model list.
    """
    # Return model list for quick diagnostics.
    return {"ok": True, "models": sorted(OCR_ENGINES.keys())}


@app.post("/extract", response_model=ApiResponse)
async def extract_ocr_data(payload: ImageURLPayload, request: Request) -> ApiResponse:
    """Extract name, time, and code from an image URL payload.

    Args:
      payload: Request payload with image URL.
      request: FastAPI request object.

    Returns:
      ApiResponse containing extracted data.
    """
    # Ensure models are warm before processing requests.
    if "en" not in OCR_ENGINES:
        raise HTTPException(status_code=503, detail="OCR models not ready yet")

    try:
        # Fetch the image from the provided URL.
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
        return await asyncio.wait_for(
            asyncio.to_thread(run_ocr_pipeline, img),
            timeout=OCR_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="ocr timeout")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
