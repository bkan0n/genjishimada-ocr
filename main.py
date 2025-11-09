from __future__ import annotations

import base64
import io
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal, get_args

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from paddleocr import PaddleOCR
from PIL import Image, ImageFile
from pydantic import BaseModel, ConfigDict

# --- CPU safe flags (MKL/oneDNN off, thread caps) ---
os.environ.setdefault("OPENBLAS_CORETYPE", "NEHALEM")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

ImageFile.LOAD_TRUNCATED_IMAGES = True


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("genjipk-ocr")


RE_DIGITS_LOOSE_CLEANUP1 = re.compile(r"[^\d\.,]")
RE_DIGITS_LOOSE_CLEANUP2 = re.compile(r"(\d{1,5}\.\d{2})")

RE_PARSE_BANNER_TIME_SEARCH_WITH_SEC = re.compile(r"([0-9OQDBZGISL\,\.]{3,12})\s*(?:SEC|초)?")
RE_PARSE_BANNER_TIME_SEARCH_NO_SEC = re.compile(r"([0-9OQDBZGISL\,\.]{3,12})")
RE_PARSE_BANNER_TIME_SEARCH_ONLY_SEC = re.compile(r"(SEC|초)")
RE_CLEAN_BANNER_FRAGMENT = re.compile(r"\s{2,}")
RE_PARSE_TIME_AGAIN = re.compile(r"\b(\d{1,4}[.,]\d{2})\s*SEC")

RE_SPACES = re.compile(r"\s+")

RE_ASCII_NAME_MATCH = re.compile(r"[A-Z][A-Z0-9_]{2,23}")
RE_XYZ_MISSION_COMPLETE = re.compile(r"([A-Z][A-Z0-9_]{2,24})\s+MISSION\s+COMPLETE")
RE_GET_USER_NAME_FROM_TOP_5 = re.compile(r"\b([A-Z][A-Z0-9_]{2,24})\b.*?(\d{1,4}[.,]\d{2})\s*SEC")

RE_MAP_CODE_FULL = re.compile(r"MAP\s*(?:C(?:O|0)?DE)\s*[:\-]?\s*([A-Z0-9]{4,6})\b")
RE_MAP_CODE_SHORT = re.compile(r"(?:C(?:O|0)?DE)\s*[:\-]?\s*([A-Z0-9]{4,6})\b")
RE_MAP_CODE_CAPTURE = re.compile(r"\b([A-Z0-9]{4,6})\b")
RE_MAP_CODE_FIND = re.compile(r"\b[A-Z0-9]{4,6}\b")

RE_BASIC_NORMALIZATION = re.compile(r"[^A-Z0-9]")
RE_MAP_CODE_NORMALIZATION = re.compile(r"MAP\s*[CLO0][O0D]{2}E", flags=re.IGNORECASE)

PADDLE_WHL_DIR = Path.home() / ".paddleocr" / "whl"


LanguageCode = Literal["en", "ch", "korean", "japan"]
RoiLabel = Literal["BL", "BAN"]  # extend later if you add more


def to_camel(s: str) -> str:
    """Convert a string to camel case."""
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


def _v3_dirs_for_language_code(language_code: LanguageCode) -> tuple[str | None, str | None]:
    """Return PP-OCRv3 detector/recognizer directories for a given language.

    The function resolves pre-baked PP-OCRv3 paths inside ~/.paddleocr/whl for the
    specified language and logs whether each directory exists. If a directory is
    missing, None is returned for that path so PaddleOCR may fall back to its
    default behavior.

    Args:
      language_code: Language code ("en", "ch", "korean", or "japan").

    Returns:
      A tuple (det_model_dir, rec_model_dir) where each element is a string path
      or None if the expected directory does not exist.
    """
    normalized_language = language_code.lower()
    if normalized_language == "en":
        det = PADDLE_WHL_DIR / "det" / "en" / "en_PP-OCRv3_det_infer"
        rec = PADDLE_WHL_DIR / "rec" / "en" / "en_PP-OCRv3_rec_infer"
    elif normalized_language == "ch":
        det = PADDLE_WHL_DIR / "det" / "ch" / "ch_PP-OCRv3_det_infer"
        rec = PADDLE_WHL_DIR / "rec" / "ch" / "ch_PP-OCRv3_rec_infer"
    elif normalized_language == "japan":
        det = PADDLE_WHL_DIR / "det" / "ml" / "Multilingual_PP-OCRv3_det_infer"
        rec = PADDLE_WHL_DIR / "rec" / "japan" / "japan_PP-OCRv3_rec_infer"
    elif normalized_language == "korean":
        det = PADDLE_WHL_DIR / "det" / "ml" / "Multilingual_PP-OCRv3_det_infer"
        rec = PADDLE_WHL_DIR / "rec" / "korean" / "korean_PP-OCRv3_rec_infer"
    else:
        return None, None

    det_dir = str(det) if det.exists() else None
    rec_dir = str(rec) if rec.exists() else None
    logger.info(
        f"[models] {language_code} v3 det_dir={'OK' if det_dir else 'MISS'} "
        f"rec_dir={'OK' if rec_dir else 'MISS'} base={PADDLE_WHL_DIR}"
    )
    return det_dir, rec_dir


SUPPORTED_LANGUAGES: tuple[LanguageCode, ...] = ("en", "ch", "korean", "japan")
OCR_ENGINES: dict[LanguageCode, PaddleOCR] = {}


def _build_ocr_engine(language_code: LanguageCode) -> PaddleOCR:
    """Build and return a PaddleOCR engine configured for PP-OCRv3 (CPU).

    The engine is constructed with deterministic CPU settings, MKLDNN disabled, and
    model directories pointing to pre-baked PP-OCRv3 weights when available.

    Args:
      language_code: Language code to initialize ("en", "ch", "korean", or "japan").

    Returns:
      A configured PaddleOCR instance ready for inference.

    Raises:
      RuntimeError: If engine creation fails due to invalid configuration or missing
        runtime dependencies.
    """
    det_dir, rec_dir = _v3_dirs_for_language_code(language_code)
    return PaddleOCR(
        ocr_version="PP-OCRv3",
        use_angle_cls=False,
        lang=language_code,
        show_log=False,
        enable_mkldnn=False,
        use_gpu=False,
        det_batch_num=1,
        rec_batch_num=1,
        cls_batch_num=1,
        det_model_dir=det_dir,
        rec_model_dir=rec_dir,
    )


def warm_ocr_engines(languages: tuple[LanguageCode, ...] = SUPPORTED_LANGUAGES) -> None:
    """Initialize and cache PaddleOCR engines for the given languages.

    Iterates the provided language sequence and warms each engine once. Engines are
    stored in the global MODELS registry and reused across requests.

    Args:
      languages: Tuple of language codes to load. Defaults to LANGS.

    Returns:
      None

    Raises:
      Exception: Propagates unexpected errors encountered during engine creation
        (errors are also logged with context).
    """
    for language_code in languages:
        if language_code in OCR_ENGINES:
            continue
        logger.info(f"📥 Warming PaddleOCR model: {language_code} (PP-OCRv3, MKLDNN OFF)")
        try:
            OCR_ENGINES[language_code] = _build_ocr_engine(language_code)
            logger.info(f"✅ Model ready: {language_code}")
        except Exception as e:
            logger.exception(f"❌ Failed to warm {language_code}: {e}")


def get_ocr_engine(language_code: LanguageCode) -> PaddleOCR:
    """Retrieve a previously initialized PaddleOCR engine from the registry.

    Args:
      language_code: Language code for the desired engine.

    Returns:
      The PaddleOCR engine associated with the given language.

    Raises:
      HTTPException: If the engine was not initialized (503 Service Unavailable).
    """
    engine = OCR_ENGINES.get(language_code)
    if engine is None:
        raise HTTPException(status_code=503, detail=f"OCR model '{language_code}' not loaded")
    return engine


def log_model_dirs() -> None:
    """Log resolved PP-OCRv3 model directories for all supported languages.

    Emits an info-level line per language indicating detector and recognizer paths.

    Returns:
      None
    """
    for language_code in ("en", "korean", "japan", "ch"):
        det_dir, rec_dir = _v3_dirs_for_language_code(language_code)
        logger.info(f"[models] {language_code}: det={det_dir} rec={rec_dir}")


ROI_TOPLEFT = [0.010, 0.020, 0.360, 0.300]
ROI_TOPLEFT_WIDE = [0.005, 0.010, 0.420, 0.340]
ROI_BANNER_TIGHT = [0.220, 0.180, 0.780, 0.360]
ROI_TOPRIGHT = [0.800, 0.170, 0.985, 0.470]
ROI_BOTTOMLEFT = [0.050, 0.825, 0.330, 0.990]
ROI_NAME_SPECIFIC = [0.050, 0.800, 0.400, 0.900]


def normalize_base64_padding(b64_string: str) -> str:
    """Normalize a base64 string by fixing URL-safe characters and padding.

    Removes whitespace, converts URL-safe base64 to standard, and appends the
    required '=' padding to make the length a multiple of 4.

    Args:
      b64_string: Base64-encoded string (possibly URL-safe and/or unpadded).

    Returns:
      A normalized base64 string suitable for decoding.
    """
    cleaned = re.sub(RE_SPACES, "", b64_string).replace("-", "+").replace("_", "/").replace(" ", "+")
    missing = (-len(cleaned)) % 4
    return cleaned + ("=" * missing if missing else "")


def decode_base64_image(image_b64: str) -> np.ndarray:
    """Decode a base64-encoded image into a BGR numpy array.

    Handles data URLs, normalizes base64 padding, decodes to bytes, loads via PIL,
    converts to RGB, and returns an OpenCV-compatible BGR image.

    Args:
      image_b64: Base64-encoded image string. May be a data URL or raw base64.

    Returns:
      A numpy ndarray in BGR color order suitable for OpenCV.

    Raises:
      HTTPException: With status 400 for missing input, invalid base64, or an
        unreadable image stream; with status 500 for unexpected decode errors.
    """
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


def crop_by_frac_roi(image: np.ndarray, roi_frac: list[float]) -> np.ndarray:
    """Crop an image using fractional ROI coordinates normalized to width/height.

    The ROI is provided as [x1, y1, x2, y2] where each value is in [0, 1] relative
    to the image size. The result is clipped to image bounds and copied.

    Args:
      image: Source image (H*W*C) in BGR or grayscale.
      roi_frac: List of four floats [x1, y1, x2, y2] normalized to the image extent.

    Returns:
      The cropped image as a new numpy array.
    """
    height, width = image.shape[:2]
    x1 = int(width * roi_frac[0])
    y1 = int(height * roi_frac[1])
    x2 = int(width * roi_frac[2])
    y2 = int(height * roi_frac[3])
    return image[max(y1, 0) : min(y2, height), max(x1, 0) : min(x2, width)].copy()


def enhance_contrast_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Enhance contrast in grayscale using CLAHE and light denoising.

    Converts a BGR image to grayscale, applies CLAHE for local contrast
    enhancement, and smooths with a small Gaussian blur to reduce speckle noise.

    Args:
      image_bgr: Source image in BGR color order.

    Returns:
      A single-channel uint8 grayscale image after enhancement.
    """
    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(grayscale)
    return cv2.GaussianBlur(grayscale, (3, 3), 0)


def mask_white_regions(image_bgr: np.ndarray) -> np.ndarray:
    """Create a binary mask emphasizing white regions in a BGR image.

    Converts the input image to HSV, thresholds for high-value/low-saturation areas
    (near white), applies median filtering to remove noise, and performs a closing
    operation to connect fragmented white areas.

    Args:
      image_bgr: Input BGR image.

    Returns:
      A binary (uint8) mask highlighting white regions.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 190], np.uint8), np.array([179, 60, 255], np.uint8))
    mask = cv2.medianBlur(mask, 3)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 1)  # type: ignore


def mask_cyan_regions(image_bgr: np.ndarray) -> np.ndarray:
    """Create a binary mask emphasizing cyan regions in a BGR image.

    Converts the input image to HSV, thresholds within the cyan hue range, applies
    median filtering to suppress isolated pixels, and performs a closing operation
    to connect nearby cyan areas.

    Args:
      image_bgr: Input BGR image.

    Returns:
      A binary (uint8) mask highlighting cyan regions.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([80, 50, 120], np.uint8), np.array([105, 255, 255], np.uint8))
    mask = cv2.medianBlur(mask, 3)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 1)  # type: ignore


def ocr_lines(image: np.ndarray, language_code: LanguageCode) -> list[tuple[str, float]]:
    """Run OCR on an image and return recognized text lines with confidences.

    Wraps the PaddleOCR `ocr()` method for a specific language model. Handles both
    BGR and grayscale images and safely catches OCR runtime errors.

    Args:
      image: Input image (BGR or grayscale) as numpy array.
      language_code: Language code corresponding to a preloaded PaddleOCR engine.

    Returns:
      A list of (text, confidence) tuples for each recognized line. Empty list if
      OCR fails or no text is detected.
    """
    if image is None or image.size == 0:
        return []
    engine = get_ocr_engine(language_code)
    if engine is None:
        return []
    bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image
    try:
        # cls=False here, and the mobile v2 cls model is not loaded
        ocr_result = engine.ocr(bgr_image, cls=False) or []
    except Exception as e:
        logger.warning(f"OCR({language_code}) failed: {e}")
        return []
    recognized_lines: list[tuple[str, float]] = []
    result_blocks = ocr_result[0] if (len(ocr_result) > 0 and isinstance(ocr_result[0], list)) else ocr_result
    for block in result_blocks or []:
        if not block or len(block) < 2:
            continue
        info = block[1]
        if not isinstance(info, (list, tuple)) or len(info) < 2:
            continue
        text = str(info[0] or "").strip()
        try:
            confidence = float(info[1]) if info[1] is not None else -1.0
        except Exception:
            confidence = -1.0
        if text:
            recognized_lines.append((text, confidence))
    return recognized_lines


def join_lines(lines: list[tuple[str, float]]) -> str:
    """Join recognized text lines into a single space-separated string.

    Args:
      lines: List of (text, confidence) tuples returned by `ocr_lines()`.

    Returns:
      A single string containing all non-empty text segments separated by spaces.
    """
    return " ".join([t for t, _ in lines]).strip()


def parse_loose_numeric_token(raw_token: str) -> float | None:
    """Convert loosely formatted OCR digit tokens into a float value.

    Attempts to correct common OCR misreads by replacing similar-looking letters
    (e.g., O->0, S->5) and extracting a numeric token with two decimal digits.

    Args:
      raw_token: Raw OCR token string possibly containing misread digits or symbols.

    Returns:
      Parsed float value if valid, otherwise None.
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
    """Parse and extract a time value (seconds) from banner text.

    Handles noisy OCR text by normalizing common errors (e.g., 'T1ME' -> 'TIME'),
    locating nearby numeric tokens around 'TIME' or 'SEC', and returning the best
    match based on proximity and context.

    Args:
      text: Raw banner text recognized by OCR.

    Returns:
      Extracted time value as float (seconds), or None if no valid pattern is found.
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
    )
    text = re.sub(RE_SPACES, " ", text).strip()
    time_keyword_index = text.find("TIME")
    if time_keyword_index != -1:
        search_window = text[time_keyword_index : time_keyword_index + 80]
        with_sec_match = re.search(RE_PARSE_BANNER_TIME_SEARCH_WITH_SEC, search_window)
        if with_sec_match:
            value = parse_loose_numeric_token(with_sec_match.group(1))
            if value is not None:
                return value
    best_scored_candidate = None
    for numeric_match in re.finditer(RE_PARSE_BANNER_TIME_SEARCH_NO_SEC, text):
        candidate = parse_loose_numeric_token(numeric_match.group(1))
        if candidate is None:
            continue

        score = 0
        distance = numeric_match.start()
        if time_keyword_index != -1 and 0 <= (distance - time_keyword_index) <= 80:
            score += 2
        if re.search(RE_PARSE_BANNER_TIME_SEARCH_ONLY_SEC, text[numeric_match.end() : numeric_match.end() + 8]):
            score += 1
        if best_scored_candidate is None or score > best_scored_candidate[0]:
            best_scored_candidate = (score, candidate)
    return best_scored_candidate[1] if best_scored_candidate else None


_HANGUL = r"\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F"
_HIRAKATA = r"\u3040-\u30FF\u31F0-\u31FF\uFF66-\uFF9F"
_HAN = r"\u3400-\u4DBF\u4E00-\u9FFF"
_LATIN = r"A-Za-z"

_CJK_ALL = f"{_HANGUL}{_HIRAKATA}{_HAN}"

# Generic ASCII words that should never be used as a player name
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
}

RE_CJK_CHARACTERS = re.compile(f"[{_CJK_ALL}]")


def count_cjk_characters(text: str) -> int:
    """Count the number of CJK (Chinese, Japanese, Korean) characters in a string.

    Args:
      text: Input text.

    Returns:
      Number of CJK characters found in the string.
    """
    """Number of CJK characters in the string."""
    return len(re.findall(RE_CJK_CHARACTERS, text or ""))


def remove_all_whitespace(text: str) -> str:
    """Remove all whitespace characters from a string.

    Args:
      text: Input text.

    Returns:
      The string with all spaces and newlines removed.
    """
    return re.sub(RE_SPACES, "", text or "")


def fraction_of_unicode_class(unicode_class_pattern: str, text: str) -> float:
    """Compute the fraction of characters in a string matching a given Unicode class.

    Useful for measuring the proportion of Hangul, Kana, Han, or Latin characters
    in OCR output.

    Args:
      unicode_class_pattern: Unicode character class regex range (e.g., _HANGUL).
      text: Input string.

    Returns:
      Fraction of matching characters in the string (0.0-1.0).
    """
    compact = remove_all_whitespace(text)
    return 0.0 if not compact else len(re.findall(f"[{unicode_class_pattern}]", compact)) / len(compact)


def build_script_profile(text: str) -> ScriptProfile:
    """Generate a script composition profile for a string.

    Calculates the relative fraction of Hangul, Kana, Han, and Latin characters in
    the given text.

    Args:
      text: Input text.

    Returns:
      Dictionary mapping script names ('hangul', 'kana', 'han', 'latin') to their
      respective fractions (0.0-1.0).
    """
    return ScriptProfile(
        hangul=fraction_of_unicode_class(_HANGUL, text),
        kana=fraction_of_unicode_class(_HIRAKATA, text),
        han=fraction_of_unicode_class(_HAN, text),
        latin=fraction_of_unicode_class(_LATIN, text),
    )


def roi_label_weight(roi_label: str) -> float:
    """Return weighting factor for a given ROI label.

    Provides a small scoring bias depending on the region of interest (ROI) from
    which OCR text was extracted, emphasizing more reliable HUD areas.

    Args:
      roi_label: ROI label string (e.g., "BL", "BAN").

    Returns:
      Float weight value used to slightly bias confidence scoring.
    """
    return {"BL": 0.20, "BAN": 0.10}.get(roi_label, 0.0)


def expected_script_for_language(language_code: str) -> str:
    """Return the dominant expected script for a given OCR language.

    Maps PaddleOCR language identifiers to the most likely primary script category
    for subsequent scoring (Hangul, Kana, Han, or Latin).

    Args:
      language_code: PaddleOCR language code ("korean", "japan", "ch", "en").

    Returns:
      Script key string used for scoring ('hangul', 'kana', 'han', or 'latin').
    """
    return {"korean": "hangul", "japan": "kana", "ch": "han", "en": "latin"}.get(language_code, "latin")


def normalize_banner_fragment(fragment_text: str) -> str:
    """Clean noisy banner text fragments for consistent scoring.

    Removes duplicate whitespace, trims decorative punctuation, and cuts off any
    portion following "MISSION COMPLETE" if present.

    Args:
      fragment_text: Raw OCR text fragment extracted from the banner.

    Returns:
      Cleaned and normalized text fragment.
    """
    value = (fragment_text or "").upper()
    if "MISSION COMPLETE" in value:
        fragment_text = fragment_text[: value.index("MISSION COMPLETE")]
    return re.sub(RE_CLEAN_BANNER_FRAGMENT, " ", fragment_text).strip(" :|~!.,*_-").strip()


def ocr_with_labels(image: np.ndarray, language_code: LanguageCode, roi_label: RoiLabel) -> list[OcrCandidate]:
    """Run OCR on an image for a given language and attach metadata labels.

    Executes text recognition using the specified language engine, computes script
    composition profiles, and returns structured entries for each recognized block.

    Args:
      image: Input image (BGR or grayscale).
      language_code: Language code to use for OCR.
      roi_label: ROI label describing the image region (e.g., "BL" or "BAN").

    Returns:
      A list of dictionaries containing text, confidence, language, ROI, and script
      profile metrics.
    """
    out: list[OcrCandidate] = []
    for text, confidence in ocr_lines(image, language_code):
        out.append(
            OcrCandidate(
                text=text.strip(),
                confidence=float(confidence or 0.0),
                language_code=language_code,
                roi_label=roi_label,
                profile=build_script_profile(text),
            )
        )
    return out


def select_best_name_candidate(candidates: list[OcrCandidate]) -> str | None:
    """Select the best candidate name among multiple OCR detections.

    Scores each candidate using a mix of OCR confidence, script consistency,
    expected script-language alignment, and ROI weighting to determine the most
    likely player name.

    Args:
      candidates: List of OCR candidate dictionaries produced by `_labelled()`.

    Returns:
      The highest-scoring text string if any candidates remain, otherwise None.
    """
    if not candidates:
        return None

    if any(c.profile.hangul >= 0.40 for c in candidates):
        candidates = [c for c in candidates if c.profile.hangul >= 0.20 or c.language_code == "korean"]

    def score(c: OcrCandidate) -> float:
        exp = expected_script_for_language(c.language_code)
        bonus = 0.0
        if exp == "hangul" and c.profile.hangul >= 0.50:
            bonus += 0.35
        if exp == "kana" and (c.profile.kana >= 0.40 or (c.profile.han >= 0.25 and c.profile.kana >= 0.20)):
            bonus += 0.30
        if exp == "han" and c.profile.han >= 0.60:
            bonus += 0.25
        bonus += 0.30 * c.profile.hangul + 0.20 * c.profile.kana + 0.10 * c.profile.han
        bonus += roi_label_weight(c.roi_label)
        return c.confidence + bonus

    pool: list[OcrCandidate] = []
    for c in candidates:
        normalized_text = normalize_banner_fragment(c.text)
        if not normalized_text:
            continue
        if max(c.profile.hangul, c.profile.kana, c.profile.han) < 0.20:
            continue
        if len(remove_all_whitespace(normalized_text)) < 2:
            continue
        pool.append(c.model_copy(update={"text": normalized_text}))

    if not pool:
        return None
    return max(pool, key=score).text


# -------- ASCII helpers --------
def ascii_name_from_bottom_left(text: str) -> str | None:
    """Extract an ASCII player name from the bottom-left HUD text.

    Typical Overwatch HUD format: "250 DAHMX" -> last token is the player name.
    Rejects generic terms (e.g., 'MISSION', 'TIME') and invalid patterns.

    Args:
      text: OCR text from the bottom-left HUD in English.

    Returns:
      Extracted ASCII name string, or None if no valid name is found.
    """
    normalized = (text or "").upper().strip()
    if not normalized:
        return None

    # Split into words and only keep the last one
    tokens = re.split(RE_SPACES, normalized)
    last = tokens[-1]

    # Name must be at least 3 chars, start with a letter,
    # and must not be a generic word (MISSION, TIME, etc.).
    if not re.fullmatch(RE_ASCII_NAME_MATCH, last):
        return None
    if last in _GENERIC_ASCII:
        return None

    return last


def ascii_name_from_banner_or_top_right(text_banner_en: str, text_top_right_en: str) -> str | None:
    """Extract an ASCII player name from banner or top-right texts as fallback.

    Searches for player-like tokens appearing before "MISSION COMPLETE" in the
    banner or preceding time strings in the top-right region.

    Args:
      text_banner_en: OCR text from the banner in English.
      text_top_right_en: OCR text from the top-right region in English.

    Returns:
      Extracted ASCII name if found, otherwise None.
    """
    banner_text_upper = (text_banner_en or "").upper()

    # Pattern "XYZ MISSION COMPLETE"
    mission_match = re.search(RE_XYZ_MISSION_COMPLETE, banner_text_upper)
    if mission_match:
        candidate = mission_match.group(1)
        if candidate not in _GENERIC_ASCII:
            return candidate

    top_right_text_upper = (text_top_right_en or "").upper()
    top5_match = re.search(RE_GET_USER_NAME_FROM_TOP_5, top_right_text_upper)
    if top5_match:
        candidate = top5_match.group(1)
        if candidate not in _GENERIC_ASCII:
            return candidate

    return None


def extract_name(  # noqa: PLR0913
    image_bottom_left: np.ndarray,
    image_bottom_left_alt: np.ndarray,
    image_banner: np.ndarray,
    image_banner_white: np.ndarray,
    image_banner_binary: np.ndarray,
    text_banner_en: str,
    text_bottom_left_en: str,
    text_top_right_en: str,
) -> str | None:
    """Determine the most likely player name across multiple OCR regions.

    Combines ASCII and CJK name candidates from bottom-left, banner, and top-right
    regions. Preference order:
      1. Clean ASCII name from bottom-left HUD.
      2. CJK name from bottom-left (if ≥2 CJK chars and no ASCII found).
      3. Global CJK name or ASCII fallback from other regions.

    Strategy:
      1) Look at the bottom-left HUD first:
         - if a clean ASCII name is found -> keep it (DAHMX, ARROW, ...).
         - otherwise, if a clean CJK name in BL -> keep it (Korean/Japanese).
      2) If BL gives nothing, look elsewhere (banner / TOP5) then global CJK.

    Args:
      image_bottom_left: Cropped bottom-left name region.
      image_bottom_left_alt: Alternate bottom-left crop for redundancy.
      image_banner: Banner region image.
      image_banner_white: White-enhanced banner mask.
      image_banner_binary: Binarized banner variant.
      text_banner_en: Banner OCR text in English.
      text_bottom_left_en: Bottom-left OCR text in English.
      text_top_right_en: Top-right OCR text in English.

    Returns:
      Final extracted player name string, or None if no valid name is detected.
    """
    # 1. ASCII candidate from bottom-left HUD
    ascii_bottom_left = ascii_name_from_bottom_left(text_bottom_left_en)

    # 2. CJK candidates (BL + banner)
    candidates: list[OcrCandidate] = []
    for language_code in get_args(LanguageCode):
        if language_code == "en":
            continue
        candidates += ocr_with_labels(image_bottom_left, language_code, "BL")
        candidates += ocr_with_labels(image_bottom_left_alt, language_code, "BL")
        candidates += ocr_with_labels(image_banner, language_code, "BAN")
        candidates += ocr_with_labels(image_banner_white, language_code, "BAN")
        candidates += ocr_with_labels(image_banner_binary, language_code, "BAN")

    # 2.a. CJK candidate specifically in bottom-left HUD
    candidates_bottom_left = [c for c in candidates if c.roi_label == "BL"]
    name_cjk_bottom_left = select_best_name_candidate(candidates_bottom_left)

    # IMPORTANT:
    # - if we have a real CJK name in BL (≥2 CJK chars) AND NO ASCII BL name,
    #   use it (full Korean/Japanese HUD case).
    if name_cjk_bottom_left and count_cjk_characters(name_cjk_bottom_left) >= 2 and not ascii_bottom_left:
        return name_cjk_bottom_left

    # - otherwise, if we have an ASCII BL name, it wins over everything (DAHMX here).
    if ascii_bottom_left:
        return ascii_bottom_left

    # 3. Global fallback (banner / TOP5 / etc.)
    name_cjk_all = select_best_name_candidate(candidates)
    if name_cjk_all and count_cjk_characters(name_cjk_all) >= 2:
        return name_cjk_all

    ascii_other = ascii_name_from_banner_or_top_right(text_banner_en, text_top_right_en)
    if ascii_other:
        return ascii_other

    return None


def normalize_map_code(raw_code_text: str | None) -> str | None:
    """Normalize and validate a candidate map code string.

    Removes non-alphanumeric characters, uppercases letters, converts 'O'→'0',
    enforces length constraints (4-6 chars), and requires at least one digit.

    Args:
      raw_code_text: Raw OCR string representing a possible map code.

    Returns:
      Cleaned map code if valid, otherwise None.
    """
    if not raw_code_text:
        return None
    # Basic normalization
    raw_code_text = re.sub(RE_BASIC_NORMALIZATION, "", raw_code_text.upper().replace("O", "0"))
    min_length = 4
    max_length = 6
    if not (min_length <= len(raw_code_text) <= max_length):
        return None
    # Map codes always contain at least one digit
    if not any(ch.isdigit() for ch in raw_code_text):
        # Examples: "KUMA", "MANTA" -> reject
        return None
    return raw_code_text


def extract_code(top_left_text: str, top_left_white_text: str, top_left_cyan_text: str) -> str | None:
    """Extract a valid map code from OCR text using multiple heuristics.

    Merges text from multiple ROIs, normalizes "MAP CODE" variants, and searches
    for patterns in decreasing strictness: explicit "MAP CODE: XXXX", short-range
    context after "MAP", or generic 4-6 character alphanumeric tokens.

    Args:
      top_left_text: OCR text from the top-left region.
      top_left_white_text: OCR text from white-emphasized top-left mask.
      top_left_cyan_text: OCR text from cyan-emphasized top-left mask.

    Returns:
      Cleaned 4-6 character map code string if found, otherwise None.
    """
    all_text = " ".join([top_left_text or "", top_left_white_text or "", top_left_cyan_text or ""]).upper()

    # Normalize around "MAP CODE"
    normalized = re.sub(
        RE_MAP_CODE_NORMALIZATION,  # Matches MAPCODE / MAP C0DE / MAP COOE / MAP LODE / MAP L0DE
        "MAP CODE",
        all_text,
    )

    # 1) strict pattern: "MAP CODE: XXXX"
    strict_pattern_match = re.search(RE_MAP_CODE_FULL, normalized)
    if strict_pattern_match:
        candidate = normalize_map_code(strict_pattern_match.group(1))
        if candidate:
            return candidate

    # 2) if there is "MAP", search in a short window after it
    map_keyword_index = normalized.find("MAP")
    if map_keyword_index != -1:
        search_window = normalized[map_keyword_index : map_keyword_index + 80]
        short_pattern_match = re.search(RE_MAP_CODE_SHORT, search_window)
        if short_pattern_match:
            candidate = normalize_map_code(short_pattern_match.group(1))
            if candidate:
                return candidate
        loose_pattern_match = re.search(RE_MAP_CODE_CAPTURE, search_window)
        if loose_pattern_match:
            candidate = normalize_map_code(loose_pattern_match.group(1))
            if candidate:
                return candidate

    # 3) last resort: scan all 4-6 char tokens
    for token in re.findall(RE_MAP_CODE_FIND, normalized):
        if token in {"MADE", "BY", "TIME", "SEC", "SPLIT", "LEVEL", "TOP", "PLAYTEST"}:
            continue
        candidate = normalize_map_code(token)
        if candidate:
            return candidate

    return None


class ImageBase64Payload(BaseModel):
    """Pydantic model for a base64-encoded image payload.

    Attributes:
      image_b64: Base64-encoded image data. May be a data URL or raw base64.
    """

    image_b64: str


@asynccontextmanager
async def warm_models_on_startup(app: FastAPI) -> AsyncIterator:
    """FastAPI startup hook that warms all configured OCR engines.

    This function logs resolved model directories and calls init_models() to
    synchronously build engines for all languages in LANGS.

    Returns:
      None

    Notes:
      - If the application runs with multiple workers, each process will warm its
        own set of engines and hold them in its process-local registry.
    """
    log_model_dirs()
    warm_ocr_engines()
    yield


app = FastAPI(title="GenjiPK OCR", lifespan=warm_models_on_startup)


@app.get("/ping")
def ping() -> dict:
    """Liveness endpoint that reports warmed OCR languages.

    Returns:
      A JSON-serializable dict containing:
        ok: Always True when the endpoint is reachable.
        models: Sorted list of language codes present in the MODELS registry.
    """
    return {"ok": True, "models": sorted(OCR_ENGINES.keys())}


@app.post("/extract", response_model=ApiResponse)
def extract_ocr_data(payload: ImageBase64Payload) -> ApiResponse:
    """Extract structured data (name, time, code) from a base64 image.

    Performs decoding, ROI crops, masking/thresholding, OCR across multiple
    languages, robust parsing of banner time, and multi-heuristic extraction of
    player name and map code.

    Args:
      payload: Pydantic model containing a base64-encoded image string.

    Returns:
      A JSON-serializable dict with the extracted fields:
        extracted.name: Detected player name or None.
        extracted.time: Run time in seconds as float or None.
        extracted.code: Map code (4-6 chars) or None.
        extracted.texts: Raw OCR strings by region for debugging.

    Raises:
      HTTPException:
        503 if required OCR models are not warmed.
        400 for invalid/missing base64 or unreadable image data.
        500 for unexpected decode errors.
    """
    if "en" not in OCR_ENGINES:
        raise HTTPException(status_code=503, detail="OCR models not ready yet")
    try:
        img = decode_base64_image(payload.image_b64)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"unexpected decode error: {e}")

    top_left = crop_by_frac_roi(img, ROI_TOPLEFT)
    top_left_wide = crop_by_frac_roi(img, ROI_TOPLEFT_WIDE)
    banner = crop_by_frac_roi(img, ROI_BANNER_TIGHT)
    top_right = crop_by_frac_roi(img, ROI_TOPRIGHT)
    bottom_left = crop_by_frac_roi(img, ROI_BOTTOMLEFT)
    bottom_left_name = crop_by_frac_roi(img, ROI_NAME_SPECIFIC)

    banner_white_mask = mask_white_regions(banner)
    banner_gray = enhance_contrast_grayscale(banner)
    banner_binary = cv2.adaptiveThreshold(banner_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)

    text_banner_en = join_lines(
        ocr_lines(banner, "en") + ocr_lines(banner_white_mask, "en") + ocr_lines(banner_binary, "en")
    )
    text_top_right_en = join_lines(ocr_lines(top_right, "en"))
    text_bottom_left_en = join_lines(ocr_lines(bottom_left, "en"))
    text_top_left_en = join_lines(ocr_lines(top_left, "en") + ocr_lines(top_left_wide, "en"))

    top_left_white_mask = mask_white_regions(top_left_wide)
    top_left_cyan_mask = mask_cyan_regions(top_left_wide)
    text_top_left_white_en = join_lines(ocr_lines(top_left_white_mask, "en")) if top_left_white_mask is not None else ""
    text_top_left_cyan_en = join_lines(ocr_lines(top_left_cyan_mask, "en")) if top_left_cyan_mask is not None else ""

    seconds = extract_banner_time_seconds(text_banner_en)
    if seconds is None:
        _m = re.search(RE_PARSE_TIME_AGAIN, (text_top_right_en or "").upper())
        if _m:
            try:
                seconds = float(_m.group(1).replace(",", "."))
            except Exception:
                seconds = None

    code = extract_code(text_top_left_en, text_top_left_white_en, text_top_left_cyan_en)

    name = extract_name(
        image_bottom_left=bottom_left_name,
        image_bottom_left_alt=bottom_left,
        image_banner=banner,
        image_banner_white=banner_white_mask,
        image_banner_binary=banner_binary,
        text_banner_en=text_banner_en,
        text_bottom_left_en=text_bottom_left_en,
        text_top_right_en=text_top_right_en,
    )

    return ApiResponse(
        extracted=ExtractedResult(
            name=name,
            time=seconds,
            code=code,
            texts=ExtractedTexts(
                top_left=text_top_left_en,
                top_left_white=text_top_left_white_en,
                top_left_cyan=text_top_left_cyan_en,
                banner=text_banner_en,
                top_right=text_top_right_en,
                bottom_left=text_bottom_left_en,
            ),
        )
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
