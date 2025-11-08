from __future__ import annotations

import base64
import io
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageFile
from pydantic import BaseModel  # <-- import placé AVANT usage

if TYPE_CHECKING:
    from paddleocr import PaddleOCR
# --- CPU safe flags (désactiver MKL-DNN / oneDNN & limiter threads) ---
os.environ.setdefault("OPENBLAS_CORETYPE", "NEHALEM")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("FLAGS_use_mkldnn", "0")  # coupe oneDNN côté Paddle

ImageFile.LOAD_TRUNCATED_IMAGES = True
app = FastAPI(title="GenjiPK OCR")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("genjipk-ocr")

# ======================================================================
# PP-OCRv3 model directory helpers (empêche les téléchargements v4)
# ======================================================================
PWHL = Path.home() / ".paddleocr" / "whl"


def _v3_dirs_for_lang(lang: str) -> tuple[str | None, str | None]:
    """Get the directory for each language.

    Retourne (det_model_dir, rec_model_dir) pour PP-OCRv3 selon la langue.
    Si le dossier n'existe pas encore (premier run), renvoie None pour que
    Paddle télécharge la bonne version v3 (grâce à ocr_version='PP-OCRv3').
    """
    normalized_language = lang.lower()
    if normalized_language == "en":
        det = PWHL / "det" / "en" / "en_PP-OCRv3_det_infer"
        rec = PWHL / "rec" / "en" / "en_PP-OCRv3_rec_infer"
    elif normalized_language == "ch":
        det = PWHL / "det" / "ch" / "ch_PP-OCRv3_det_infer"
        rec = PWHL / "rec" / "ch" / "ch_PP-OCRv3_rec_infer"
    elif normalized_language == "japan":
        det = PWHL / "det" / "ml" / "Multilingual_PP-OCRv3_det_infer"
        rec = PWHL / "rec" / "japan" / "japan_PP-OCRv3_rec_infer"
    elif normalized_language == "korean":
        det = PWHL / "det" / "ml" / "Multilingual_PP-OCRv3_det_infer"
        rec = PWHL / "rec" / "korean" / "korean_PP-OCRv3_rec_infer"
    else:
        return None, None

    det_dir = str(det) if det.exists() else None
    rec_dir = str(rec) if rec.exists() else None
    logger.info(
        f"[models] {lang} v3 det_dir={'OK' if det_dir else 'MISS'} rec_dir={'OK' if rec_dir else 'MISS'} base={PWHL}"
    )
    return det_dir, rec_dir


# ========================== Lazy PaddleOCR (multi-lang) ==========================
class LazyPaddleOCR:
    def __init__(self) -> None:
        """Initialize the LazzyPddleOCR object."""
        self.models: Dict[str, PaddleOCR] = {}
        self.loading: set[str] = set()

    def load_one(self, lang: str) -> None:
        """Load a language."""
        if lang in self.models or lang in self.loading:
            return
        self.loading.add(lang)
        logger.info(f"📥 Loading PaddleOCR model: {lang} (PP-OCRv3, MKLDNN OFF)")
        try:
            from paddleocr import PaddleOCR  # noqa: PLC0415

            det_dir, rec_dir = _v3_dirs_for_lang(lang)

            # IMPORTANT: ocr_version='PP-OCRv3' + enable_mkldnn=False + use_gpu=False
            self.models[lang] = PaddleOCR(
                ocr_version="PP-OCRv3",
                use_angle_cls=False,
                lang=lang,
                show_log=False,
                enable_mkldnn=False,  # coupe oneDNN (en plus de FLAGS_use_mkldnn=0)
                use_gpu=False,
                det_batch_num=1,
                rec_batch_num=1,
                cls_batch_num=1,
                det_model_dir=det_dir,
                rec_model_dir=rec_dir,
                # cls_model_dir: on laisse par défaut (mobile v2.0) ; cls=False de toute façon
            )
            logger.info(f"✅ Model loaded: {lang} (forced v3)")
        except Exception as e:
            logger.error(f"❌ Failed to load {lang}: {e}")
        finally:
            self.loading.discard(lang)

    def get(self, lang: str = "en") -> PaddleOCR:
        """Get a model."""
        if lang not in self.models:
            self.load_one(lang)
        return self.models.get(lang)

    def ready(self) -> bool:
        """Check if ready."""
        return "en" in self.models


LAZY = LazyPaddleOCR()


def _log_model_dirs():
    for lang in ("en", "korean", "japan", "ch"):
        det_dir, rec_dir = _v3_dirs_for_lang(lang)
        logger.info(f"[models] {lang}: det={det_dir} rec={rec_dir}")


@app.on_event("startup")
async def _warm_en() -> None:
    _log_model_dirs()
    LAZY.load_one("en")


# ================================ ROIs (16:9) ==================================
ROI_TOPLEFT = [0.010, 0.020, 0.360, 0.300]
ROI_TOPLEFT_WIDE = [0.005, 0.010, 0.420, 0.340]
ROI_BANNER_TIGHT = [0.220, 0.180, 0.780, 0.360]
ROI_TOPRIGHT = [0.800, 0.170, 0.985, 0.470]
ROI_BOTTOMLEFT = [0.050, 0.825, 0.330, 0.990]
ROI_NAME_SPECIFIC = [0.050, 0.800, 0.400, 0.900]


# ================================ utils I/O ====================================
def _fix_b64_padding(s: str) -> str:
    s = re.sub(r"\s+", "", s).replace("-", "+").replace("_", "/").replace(" ", "+")
    missing = (-len(s)) % 4
    return s + ("=" * missing if missing else "")


def decode_b64(img_b64: str) -> np.ndarray:
    """Decode b64 image."""
    if not img_b64:
        raise HTTPException(status_code=400, detail="image_b64 is required")
    if img_b64.startswith("data:"):
        img_b64 = img_b64.split(",", 1)[1]
    img_b64 = _fix_b64_padding(img_b64)
    try:
        raw = base64.b64decode(img_b64, validate=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid base64: {e}")
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
        im = im.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image stream: {e}")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def crop_norm(img: np.ndarray, roi: List[float]) -> np.ndarray:
    """Crop and normalize."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = int(w * roi[0]), int(h * roi[1]), int(w * roi[2]), int(h * roi[3])
    return img[max(y1, 0) : min(y2, h), max(x1, 0) : min(x2, w)].copy()


def clahe_gray(img: np.ndarray) -> np.ndarray:
    """Process clahe gray."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    return cv2.GaussianBlur(g, (3, 3), 0)


def emphasize_white(img: np.ndarray) -> np.ndarray:
    """Emphasize white."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([0, 0, 190], np.uint8), np.array([179, 60, 255], np.uint8))
    m = cv2.medianBlur(m, 3)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 1)


def emphasize_cyan(img: np.ndarray) -> np.ndarray:
    """Emphasize cyan."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([80, 50, 120], np.uint8), np.array([105, 255, 255], np.uint8))
    m = cv2.medianBlur(m, 3)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 1)


# ================================ OCR wrappers =================================
def ocr_lines(img: np.ndarray, lang: str) -> List[Tuple[str, float]]:
    if img is None or img.size == 0:
        return []
    engine = LAZY.get(lang)
    if engine is None:
        return []
    src = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
    try:
        # cls=False ici, et le modèle cls mobile v2 n'est pas chargé
        res = engine.ocr(src, cls=False) or []
    except Exception as e:
        logger.warning(f"OCR({lang}) failed: {e}")
        return []
    out: List[Tuple[str, float]] = []
    blocks = res[0] if (len(res) > 0 and isinstance(res[0], list)) else res
    for item in blocks or []:
        if not item or len(item) < 2:
            continue
        info = item[1]
        if not isinstance(info, (list, tuple)) or len(info) < 2:
            continue
        txt = str(info[0] or "").strip()
        try:
            conf = float(info[1]) if info[1] is not None else -1.0
        except Exception:
            conf = -1.0
        if txt:
            out.append((txt, conf))
    return out


def join_lines(lines: List[Tuple[str, float]]) -> str:
    return " ".join([t for t, _ in lines]).strip()


# ================================ Time parsing =================================
def _digits_loose_to_float(token: str) -> Optional[float]:
    if not token:
        return None
    t = (
        token.upper()
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
    t = re.sub(r"[^\d\.,]", "", t).replace(",", ".")
    m = re.search(r"(\d{1,5}\.\d{2})", t)
    return float(m.group(1)) if m else None


def parse_banner_time_robust(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.upper().replace("T1ME", "TIME").replace("TLME", "TIME").replace("TI ME", "TIME")
    s = s.replace("5EC", "SEC").replace("SE€", "SEC").replace("SEL", "SEC")
    s = re.sub(r"\s+", " ", s).strip()
    i = s.find("TIME")
    if i != -1:
        win = s[i : i + 80]
        m = re.search(r"([0-9OQDBZGISL\,\.]{3,12})\s*(?:SEC|초)?", win)
        if m:
            v = _digits_loose_to_float(m.group(1))
            if v is not None:
                return v
    best = None
    for m in re.finditer(r"([0-9OQDBZGISL\,\.]{3,12})", s):
        cand = _digits_loose_to_float(m.group(1))
        if cand is None:
            continue
        j, score = m.start(), 0
        if i != -1 and 0 <= (j - i) <= 80:
            score += 2
        if re.search(r"(SEC|초)", s[m.end() : m.end() + 8]):
            score += 1
        if best is None or score > best[0]:
            best = (score, cand)
    return best[1] if best else None


# =============================== Name (script-aware) ===========================
_HANGUL = r"\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F"
_HIRAKATA = r"\u3040-\u30FF\u31F0-\u31FF\uFF66-\uFF9F"
_HAN = r"\u3400-\u4DBF\u4E00-\u9FFF"
_LATIN = r"A-Za-z"

_CJK_ALL = f"{_HANGUL}{_HIRAKATA}{_HAN}"

# mots ASCII génériques qu'on ne veut jamais comme pseudo
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


def _cjk_len(s: str) -> int:
    """Nombre de caractères CJK dans la chaîne."""
    return len(re.findall(f"[{_CJK_ALL}]", s or ""))


def _strip_spaces(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _frac(cls_pat: str, s: str) -> float:
    s0 = _strip_spaces(s)
    return 0.0 if not s0 else len(re.findall(f"[{cls_pat}]", s0)) / len(s0)


def _script_profile(s: str) -> Dict[str, float]:
    return {
        "hangul": _frac(_HANGUL, s),
        "kana": _frac(_HIRAKATA, s),
        "han": _frac(_HAN, s),
        "latin": _frac(_LATIN, s),
    }


def _roi_weight(roi: str) -> float:
    return {"BL": 0.20, "BAN": 0.10}.get(roi, 0.0)


def _lang_expect(lang: str) -> str:
    return {"korean": "hangul", "japan": "kana", "ch": "han", "en": "latin"}.get(lang, "latin")


def _clean_banner_fragment(t: str) -> str:
    U = (t or "").upper()
    if "MISSION COMPLETE" in U:
        t = t[: U.index("MISSION COMPLETE")]
    return re.sub(r"\s{2,}", " ", t).strip(" :|~!.,*_-").strip()


def _labelled(img: np.ndarray, lang: str, roi: str) -> List[Dict]:
    out = []
    for txt, conf in ocr_lines(img, lang):
        prof = _script_profile(txt)
        out.append(
            {
                "text": txt.strip(),
                "conf": float(conf or 0.0),
                "lang": lang,
                "roi": roi,
                "prof": prof,
            }
        )
    return out


def _pick_name(cands: List[Dict]) -> Optional[str]:
    if not cands:
        return None

    # Si beaucoup de hangul, on garde que les vrais blocs coréens
    if any(c["prof"]["hangul"] >= 0.40 for c in cands):
        cands = [c for c in cands if c["prof"]["hangul"] >= 0.20 or c["lang"] == "korean"]

    def score(c):
        conf, prof, roi, lang = c["conf"], c["prof"], c["roi"], c["lang"]
        exp = _lang_expect(lang)
        bonus = 0.0
        if exp == "hangul" and prof["hangul"] >= 0.50:
            bonus += 0.35
        if exp == "kana" and (prof["kana"] >= 0.40 or (prof["han"] >= 0.25 and prof["kana"] >= 0.20)):
            bonus += 0.30
        if exp == "han" and prof["han"] >= 0.60:
            bonus += 0.25
        bonus += 0.30 * prof["hangul"] + 0.20 * prof["kana"] + 0.10 * prof["han"]
        bonus += _roi_weight(roi)
        return conf + bonus

    pool = []
    for c in cands:
        t = _clean_banner_fragment(c["text"])
        if not t:
            continue
        if max(c["prof"]["hangul"], c["prof"]["kana"], c["prof"]["han"]) < 0.20:
            continue
        if len(_strip_spaces(t)) < 2:
            continue
        c2 = dict(c)
        c2["text"] = t
        pool.append(c2)

    if not pool:
        return None
    return max(pool, key=score)["text"]


# -------- ASCII helpers --------
def _ascii_name_from_bl(text_bl_en: str) -> Optional[str]:
    """Essaie de récupérer le pseudo ASCII dans le HUD bas-gauche.
    Overwatch affiche typiquement : "250 DAHMX" -> on prend le DERNIER token.
    """
    s = (text_bl_en or "").upper().strip()
    if not s:
        return None

    # On découpe en mots et on prend uniquement le dernier
    tokens = re.split(r"\s+", s)
    last = tokens[-1]

    # Le pseudo doit être au moins 3 caractères, commencer par une lettre,
    # et ne pas être un mot générique (MISSION, TIME, etc).
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{2,23}", last):
        return None
    if last in _GENERIC_ASCII:
        return None

    return last


def _ascii_name_from_banner_tr(text_ban_en: str, text_tr_en: str) -> Optional[str]:
    """Nom ASCII de secours (bannière / TOP5) quand le HUD ne donne rien."""
    s_ban = (text_ban_en or "").upper()

    # Pattern "XYZ MISSION COMPLETE"
    m = re.search(r"([A-Z][A-Z0-9_]{2,24})\s+MISSION\s+COMPLETE", s_ban)
    if m:
        cand = m.group(1)
        if cand not in _GENERIC_ASCII:
            return cand

    s_tr = (text_tr_en or "").upper()
    m = re.search(r"\b([A-Z][A-Z0-9_]{2,24})\b.*?(\d{1,4}[.,]\d{2})\s*SEC", s_tr)
    if m:
        cand = m.group(1)
        if cand not in _GENERIC_ASCII:
            return cand

    return None


def extract_name(
    img_bl: np.ndarray,
    img_bl_alt: np.ndarray,
    img_ban: np.ndarray,
    img_ban_white: np.ndarray,
    img_ban_bin: np.ndarray,
    text_ban_en: str,
    text_bl_en: str,
    text_tr_en: str,
) -> Optional[str]:
    """Extract the player name.

    Stratégie :
    1) On regarde d'abord le HUD bas-gauche :
       - si pseudo ASCII propre -> on le garde (DAHMX, ARROW, ...).
       - sinon, si pseudo CJK propre en BL -> on le garde (coréen/japonais).
    2) Si BL ne donne rien, on cherche ailleurs (bannière / TOP5) puis CJK global.
    """
    # 1. Candidat ASCII depuis le HUD bas-gauche
    ascii_bl = _ascii_name_from_bl(text_bl_en)

    # 2. Candidats CJK (BL + BAN)
    cands: List[Dict] = []
    for L in ["korean", "japan", "ch"]:
        LAZY.load_one(L)
        cands += _labelled(img_bl, L, "BL")
        cands += _labelled(img_bl_alt, L, "BL")
        cands += _labelled(img_ban, L, "BAN")
        cands += _labelled(img_ban_white, L, "BAN")
        cands += _labelled(img_ban_bin, L, "BAN")

    # 2.a. CJK spécifique au HUD bas-gauche
    cands_bl = [c for c in cands if c["roi"] == "BL"]
    name_cjk_bl = _pick_name(cands_bl)

    # IMPORTANT :
    # - si on a un vrai nom CJK BL (≥2 caractères CJK) ET PAS de pseudo ASCII BL,
    #   on le prend (cas full coréen/japonais).
    if name_cjk_bl and _cjk_len(name_cjk_bl) >= 2 and not ascii_bl:
        return name_cjk_bl

    # - sinon, si on a un pseudo ASCII BL, il passe avant tout (DAHMX ici).
    if ascii_bl:
        return ascii_bl

    # 3. Fallback global (bannière / TOP5 / etc.)
    name_cjk_all = _pick_name(cands)
    if name_cjk_all and _cjk_len(name_cjk_all) >= 2:
        return name_cjk_all

    ascii_other = _ascii_name_from_banner_tr(text_ban_en, text_tr_en)
    if ascii_other:
        return ascii_other

    return None


# =============================== Code extraction =================================
def clean_code(s: str | None) -> str | None:
    """Clean map code."""
    if not s:
        return None
    # Normalisation basique
    s = re.sub(r"[^A-Z0-9]", "", s.upper().replace("O", "0"))
    min_length = 4
    max_length = 6
    if not (min_length <= len(s) <= max_length):
        return None
    # Les codes de map ont toujours au moins un chiffre
    if not any(ch.isdigit() for ch in s):
        # Exemple: "KUMA", "MANTA" -> on rejette
        return None
    return s


def extract_code(tl_text: str, tl_white_text: str, tl_cyan_text: str) -> str | None:
    """Extract the code from the image."""
    all_text = " ".join([tl_text or "", tl_white_text or "", tl_cyan_text or ""]).upper()

    # Normalisation autour de "MAP CODE"
    norm = all_text.replace("MAPCODE", "MAP CODE").replace("MAPC0DE", "MAP CODE")
    norm = re.sub(r"MAP\s+C0DE", "MAP CODE", norm)
    norm = re.sub(r"MAP\s+COOE", "MAP CODE", norm)
    norm = re.sub(r"MAP\s+LODE", "MAP CODE", norm)
    norm = re.sub(r"MAP\s+L0DE", "MAP CODE", norm)

    # 1) motif strict : "MAP CODE: XXXX"
    m = re.search(r"MAP\s*(?:C(?:O|0)?DE)\s*[:\-]?\s*([A-Z0-9]{4,6})\b", norm)
    if m:
        cand = clean_code(m.group(1))
        if cand:
            return cand

    # 2) s'il y a "MAP", chercher dans une fenêtre courte après
    i = norm.find("MAP")
    if i != -1:
        win = norm[i : i + 80]
        m2 = re.search(r"(?:C(?:O|0)?DE)\s*[:\-]?\s*([A-Z0-9]{4,6})\b", win)
        if m2:
            cand = clean_code(m2.group(1))
            if cand:
                return cand
        t = re.search(r"\b([A-Z0-9]{4,6})\b", win)
        if t:
            cand = clean_code(t.group(1))
            if cand:
                return cand

    # 3) dernier recours : scanner tous les tokens 4-6 chars
    for token in re.findall(r"\b[A-Z0-9]{4,6}\b", norm):
        if token in {"MADE", "BY", "TIME", "SEC", "SPLIT", "LEVEL", "TOP", "PLAYTEST"}:
            continue
        cand = clean_code(token)
        if cand:
            return cand

    return None


# ================================ API ===========================================
class B64(BaseModel):
    image_b64: str


@app.get("/ping")
def ping() -> dict:
    """Ping the server."""
    return {"ok": True, "models": list(LAZY.models.keys())}


@app.post("/extract")
def extract(payload: B64) -> dict:
    """Extract the data from the image."""
    if LAZY.get("en") is None:
        raise HTTPException(status_code=503, detail="OCR models not ready yet")
    try:
        img = decode_b64(payload.image_b64)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"unexpected decode error: {e}")

    tl = crop_norm(img, ROI_TOPLEFT)
    tlw = crop_norm(img, ROI_TOPLEFT_WIDE)
    ban = crop_norm(img, ROI_BANNER_TIGHT)
    tr = crop_norm(img, ROI_TOPRIGHT)
    bl = crop_norm(img, ROI_BOTTOMLEFT)
    bl_n = crop_norm(img, ROI_NAME_SPECIFIC)

    ban_white = emphasize_white(ban)
    ban_gray = clahe_gray(ban)
    ban_bin = cv2.adaptiveThreshold(ban_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)

    text_ban_en = join_lines(ocr_lines(ban, "en") + ocr_lines(ban_white, "en") + ocr_lines(ban_bin, "en"))
    text_tr_en = join_lines(ocr_lines(tr, "en"))
    text_bl_en = join_lines(ocr_lines(bl, "en"))
    text_tl_en = join_lines(ocr_lines(tl, "en") + ocr_lines(tlw, "en"))

    tl_white_mask = emphasize_white(tlw)
    tl_cyan_mask = emphasize_cyan(tlw)
    text_tl_white_en = join_lines(ocr_lines(tl_white_mask, "en")) if tl_white_mask is not None else ""
    text_tl_cyan_en = join_lines(ocr_lines(tl_cyan_mask, "en")) if tl_cyan_mask is not None else ""

    sec = parse_banner_time_robust(text_ban_en)
    if sec is None:
        _m = re.search(r"\b(\d{1,4}[.,]\d{2})\s*SEC", (text_tr_en or "").upper())
        if _m:
            try:
                sec = float(_m.group(1).replace(",", "."))
            except Exception:
                sec = None

    code = extract_code(text_tl_en, text_tl_white_en, text_tl_cyan_en)

    name = extract_name(
        img_bl=bl_n,
        img_bl_alt=bl,
        img_ban=ban,
        img_ban_white=ban_white,
        img_ban_bin=ban_bin,
        text_ban_en=text_ban_en,
        text_bl_en=text_bl_en,
        text_tr_en=text_tr_en,
    )

    return {
        "extracted": {
            "name": name,
            "time": sec,
            "code": code,
            "texts": {
                "topLeft": text_tl_en,
                "topLeftWhite": text_tl_white_en,
                "topLeftCyan": text_tl_cyan_en,
                "banner": text_ban_en,
                "topRight": text_tr_en,
                "bottomLeft": text_bl_en,
            },
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
