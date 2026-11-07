"""Pure parsing and spatial evidence rules; importing this module never loads OCR models."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

Point = tuple[float, float]
Polygon = tuple[Point, ...]
COMPLETION = re.compile(
    r"MISSION\s*COMPLETE[!\uff01]?|任务完成|任務完成|挑战完成|挑戰完成|通关成功|通關成功|"
    r"ミッション(?:を)?(?:完了|クリア)|クリア成功|미션\s*완료|임무\s*완료|"
    r"(?:최종\s*목표에?\s*도달했습니다|[맵앱]을?\s*깨셨습니다)",
    re.IGNORECASE,
)
TIME_LABEL = r"(?:T[I1L]\s*ME|用时|用時|耗时|耗時|時間|タイム|시간)"
UNIT = r"(?:S[E5][CG€]|SECONDS?|秒|초)"
TIME = re.compile(
    rf"(?<![\d.,:])([\dOQDBZGISL]{{1,5}}\s*[.,:\uff0c\uff0e]\s*[\dOQDBZGISL]{{2}})"
    rf"(?![\d.,:])\s*({UNIT})?",
    re.IGNORECASE,
)
CODE_LABEL = re.compile(
    r"(?:(?<![A-Z0-9_])(?:MAP\s*)?C[O0]DE|[맵앱맨]\s*코드|地图\s*代码|地圖\s*代碼|"
    r"地图\s*码|地圖\s*碼|マップ\s*コード)\s*[:\uff1a=\-]?\s*",
    re.IGNORECASE,
)
NOISY_CODE_FIELD = re.compile(
    r"^(?:O\s*H(?:\s*[CF]){0,2}|H\s*(?:A\s*)?C)\s*[:\uff1a]\s*([A-Z0-9]{4,6})\s*$",
    re.IGNORECASE,
)
CREATOR_FIELD = re.compile(
    r"^(?:MADE\s+BY|CREATOR|AUTHOR|제작자|作者|製作者|制作者|作成者)\s*[:\uff1a]\s*\S.*$",
    re.IGNORECASE,
)
DISCORD_FIELD = re.compile(
    r"^DISCORD\s*[:\uff1a]\s*(?:https?://)?(?:dsc\.gg|discord\.gg|discord(?:app)?\.com/invite)/[A-Z0-9_-]+\s*$",
    re.IGNORECASE,
)
NOISE = {
    "mission",
    "complete",
    "time",
    "sec",
    "seconds",
    "top",
    "top5",
    "leaderboard",
    "hold",
    "restart",
    "split",
    "checkpoint",
    "level",
    "player",
    "health",
    "bhop",
    "climb",
    "map",
    "code",
    "시간",
    "초",
    "時間",
    "秒",
    "用时",
    "用時",
    "任务完成",
    "任務完成",
    "미션",
    "완료",
}
HUD_NOISE = re.compile(
    r"\b(?:SPLIT|CHECKPOINT|HOLD|RESTART|SPECTATE|DISCORD|MADE\s*BY|LEVEL|SERVER)\b|분할|체크포인트",
    re.IGNORECASE,
)
_DIGITS = str.maketrans("OQDBZGISL", "000826151")


@dataclass(frozen=True)
class Detection:
    """One observation, with both model-input and screenshot coordinates."""

    text: str
    confidence: float
    polygon: Polygon | None = None
    local_polygon: Polygon | None = None
    roi: str = ""
    language: str = ""
    model: str = ""
    variant: str = "raw"
    pass_id: str = ""

    @property
    def bounds(self) -> tuple[float, float, float, float] | None:
        """Return screenshot-relative bounds when geometry is available."""
        if not self.polygon:
            return None
        xs, ys = zip(*self.polygon)
        return min(xs), min(ys), max(xs), max(ys)


@dataclass(frozen=True)
class Row:
    """Fragments sharing a baseline within exactly one OCR pass."""

    detections: tuple[Detection, ...]

    @property
    def text(self) -> str:
        """Return fragments in reading order."""
        return " ".join(d.text for d in self.detections)

    @property
    def confidence(self) -> float:
        """Use the weakest fragment; duplicate variants cannot add confidence."""
        return min(d.confidence for d in self.detections)

    @property
    def bounds(self) -> tuple[float, float, float, float] | None:
        """Return the enclosing screenshot rectangle."""
        boxes = [d.bounds for d in self.detections if d.bounds]
        if not boxes:
            return None
        return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


@dataclass(frozen=True)
class Pair:
    """A name and time supported by one banner or leaderboard entry."""

    name: str
    centiseconds: int
    source: str
    confidence: float
    rows: tuple[Row, ...]


@dataclass(frozen=True)
class MetadataCodeRegion:
    """An unverified code-field location bounded by independent semantic anchors."""

    row: Row
    creator_bounds: tuple[float, float, float, float]
    timer_bounds: tuple[float, float, float, float]


def normalize_name(value: str) -> str:
    """Fold Latin accents for comparison while preserving other scripts' identity marks."""
    value = unicodedata.normalize("NFKC", value).casefold()
    normalized = []
    latin_base = False
    for char in unicodedata.normalize("NFD", value):
        category = unicodedata.category(char)
        if category[0] in {"L", "N"}:
            normalized.append(char)
            latin_base = category[0] == "L" and "LATIN" in unicodedata.name(char, "")
        elif category[0] == "M":
            # Dakuten/handakuten and East Asian tone marks are identity-bearing,
            # including unusual mixed-script names where they follow Latin.
            specific_mark = any(
                script in unicodedata.name(char, "") for script in ("KATAKANA", "HIRAGANA", "IDEOGRAPHIC", "HANGUL")
            )
            diacritic = (
                ("\u0300" <= char <= "\u036f" and char != "\u034f")
                or "\u1ab0" <= char <= "\u1aff"
                or "\u1dc0" <= char <= "\u1dff"
                or "\ufe20" <= char <= "\ufe2f"
            )
            if not latin_base or specific_mark or not diacritic:
                normalized.append(char)
        else:
            if not char.isspace() and category != "Cf":
                normalized.append(char)
            latin_base = False
    return unicodedata.normalize("NFC", "".join(normalized))


def name_score(left: str | None, right: str | None) -> float:
    """Allow exact normalized names and a single Latin glyph confusion in long names."""
    a, b = normalize_name(left or ""), normalize_name(right or "")
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) != len(b) or len(a) < 5 or not a.isascii() or not b.isascii():
        return 0.0
    confusions = {frozenset(p) for p in ("0o", "1i", "1l", "5s", "8b")}
    changes = [(x, y) for x, y in zip(a, b) if x != y]
    return 0.95 if len(changes) == 1 and frozenset(changes[0]) in confusions else 0.0


def clean_name(value: str) -> str | None:
    """Keep the observed Unicode spelling while rejecting HUD prose and bare numbers."""
    value = value.strip(" \t\n:\uff1a!\uff01|,.*~-")
    value = re.sub(r"^\s*[1-5][.)、:]\s*", "", value)
    value = re.sub(r"^\s*TOP\s*5\s*", "", value, flags=re.IGNORECASE)
    words = value.split()
    if not words or any(w.casefold() in NOISE for w in words) or HUD_NOISE.search(value):
        return None
    minimum = 2 if re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3]", value) else 3
    if not minimum <= len(normalize_name(value)) <= 32 or not any(c.isalpha() for c in value):
        return None
    if any(not (c.isalnum() or c in "_ -'" or unicodedata.category(c).startswith("M")) for c in value):
        return None
    return value


def to_centiseconds(value: float | str | None) -> int | None:
    """Convert exact decimal seconds without binary floating point comparisons."""
    try:
        dec = Decimal(str(value))
        if not dec.is_finite() or dec <= 0 or dec * 100 != (dec * 100).to_integral_value():
            return None
        return int(dec * 100)
    except (InvalidOperation, ValueError, TypeError):
        return None


def completion_name(prefix: str) -> str | None:
    """Strip observed completion-template grammar while retaining the player's spelling."""
    prefix = re.sub(r"\s+(?:님이?|이\[가\]|이\(가\)|이가)\s*$", "", prefix)
    return clean_name(prefix)


def times_equal(left: float | None, right: float | None, tolerance: float = 0.0) -> bool:
    """Compare positive finite times using an absolute tolerance in centiseconds."""
    a, b = to_centiseconds(left), to_centiseconds(right)
    return a is not None and b is not None and abs(a - b) <= Decimal(str(max(0.0, tolerance))) * 100


def normalize_time_text(text: str) -> str:
    """Join split numeric tokens only inside a numeric field or before a seconds unit."""
    text = text.translate(str.maketrans("\uff0c\uff0e\uff1a", ",.:"))
    text = re.sub(rf"(?<!\w)(\d{{1,5}})\s+(\d{{2}})\s*(?={UNIT}\b|秒|초)", r"\1.\2 ", text, flags=re.IGNORECASE)
    # An explicit TIME label disambiguates a leading numeric fragment from a rank.
    return re.sub(rf"({TIME_LABEL}\s*[:\uff1a]?\s*)(\d)\s+(\d{{3}}[.,:]\d{{2}})", r"\1\2\3", text, flags=re.IGNORECASE)


def time_values(text: str, minimum: float = 10.0) -> list[tuple[re.Match, int]]:
    """Find valid time fields, retaining positions for adjacent name parsing."""
    text = normalize_time_text(text)
    found = []
    for match in TIME.finditer(text):
        raw = re.sub(r"\s", "", unicodedata.normalize("NFKC", match[1]).upper()).translate(_DIGITS)
        raw = re.sub(r"[,:\uff0c\uff0e]", ".", raw)
        cs = to_centiseconds(raw)
        if cs is not None and minimum * 100 <= cs <= 1536000:
            found.append((match, cs))
    return found


def text_pairs(text: str, source: str = "topRight.top5", minimum: float = 10.0) -> list[tuple[str, int]]:
    """Parse adjacent name/time fields on explicit text lines, never search ahead past another name."""
    pairs = []
    for raw_line in text.splitlines():
        line = normalize_time_text(raw_line)
        completion = COMPLETION.search(line)
        if source == "banner":
            if not completion:
                continue
            name = completion_name(line[: completion.start()])
            values = time_values(line[completion.end() :], minimum)
            tail = line[completion.end() :]
            if name and len(values) == 1 and not HUD_NOISE.search(tail[: values[0][0].end()]):
                pairs.append((name, values[0][1]))
            continue
        start = 0
        for match, cs in time_values(line, minimum):
            prefix = line[start : match.start()].strip(" :\uff1a|-\t")
            # A separate digit before a decimal can belong to the time rather
            # than the identity: ``Alice 1 107.23`` must not verify Alice1's
            # 107.23 result. Text alone cannot resolve that boundary.
            leading = re.search(r"(?:^|\s)(\d{1,2})$", prefix)
            if leading:
                raw = unicodedata.normalize("NFKC", match[1]).upper()
                raw = re.sub(r"\s", "", raw).translate(_DIGITS)
                combined = to_centiseconds(re.sub(r"[,:\uff0c\uff0e]", ".", leading[1] + raw))
                if combined is not None and minimum * 100 <= combined <= 1536000:
                    start = match.end()
                    continue
            name = clean_name(prefix)
            if name:
                pairs.append((name, cs))
            start = match.end()
    return pairs


def normalize_code(value: str | None) -> str | None:
    """Validate literal Workshop codes without changing O/0, I/1 or other identities."""
    value = unicodedata.normalize("NFKC", value or "").upper().strip()
    if not re.fullmatch(r"[A-Z0-9]{4,6}", value) or value.casefold() in NOISE:
        return None
    return value


def code_values(text: str) -> list[str]:
    """Extract only labelled code fields; usernames and arbitrary substrings are excluded."""
    result = []
    for line in text.splitlines():
        for label in CODE_LABEL.finditer(line):
            after = line[label.end() :]
            # Metadata marks the end of the code field, including compact OCR output.
            after = re.split(r"\b(?:DISCORD|MADE\s*BY|TIME|SPLIT|BHOP)\b", after, flags=re.IGNORECASE)[0]
            token = re.match(r"([A-Z0-9]{1,6})(?:[ \t]+([A-Z0-9]{1,5}))?(?![A-Z0-9])", after, re.IGNORECASE)
            if not token:
                continue
            first, second = token.groups()
            code = normalize_code(first)
            # Text alone cannot establish whether adjacent alphanumeric chunks
            # are one code or different HUD fields, even when the first chunk
            # is too short to be a valid code. Spatial assembly happens below.
            if second:
                continue
            if code and code not in result:
                result.append(code)
    return result


def _polygon(value: Any) -> Polygon | None:  # noqa: ANN401 - external Paddle schema
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
            x1, y1, x2, y2 = value
            value = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        points = tuple((float(p[0]), float(p[1])) for p in value)
        return points if len(points) >= 4 and all(math.isfinite(v) for p in points for v in p) else None
    except (TypeError, ValueError, IndexError):
        return None


def parse_ocr_result(  # noqa: PLR0913 - explicit OCR provenance and coordinate transform
    result: Any,  # noqa: ANN401 - external Paddle schema
    *,
    roi: str = "",
    language: str = "",
    model: str = "",
    variant: str = "raw",
    pass_id: str = "",
    offset: Point = (0.0, 0.0),
    scale: Point = (1.0, 1.0),
) -> list[Detection]:
    """Adapt Paddle 2.x/3.x results, including ndarray scores and serialized Result.json."""
    out: list[Detection] = []

    def add(text: Any, score: Any = None, poly: Any = None) -> None:  # noqa: ANN401 - external Paddle schema
        if not isinstance(text, str) or not text.strip():
            return
        try:
            conf = float(score) if score is not None else 0.0
            conf = max(0.0, min(1.0, conf)) if math.isfinite(conf) else 0.0
        except (TypeError, ValueError):
            conf = 0.0
        local = _polygon(poly)
        original = tuple((x / scale[0] + offset[0], y / scale[1] + offset[1]) for x, y in local) if local else None
        out.append(Detection(text.strip(), conf, original, local, roi, language, model, variant, pass_id))

    def walk(data: Any) -> None:  # noqa: ANN401, PLR0912 - versioned Paddle result adapter
        if hasattr(data, "json"):
            data = data.json() if callable(data.json) else data.json
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except ValueError:
                return
        if isinstance(data, dict):
            texts = data.get("rec_texts")
            if texts is not None:
                scores = data.get("rec_scores")
                polygons = data.get("rec_polys")
                if polygons is None:
                    polygons = data.get("dt_polys")
                if polygons is None:
                    polygons = data.get("rec_boxes")
                for i, text in enumerate(texts):
                    add(
                        text,
                        scores[i] if scores is not None and i < len(scores) else None,
                        polygons[i] if polygons is not None and i < len(polygons) else None,
                    )
            elif "rec_text" in data or "text" in data:
                add(
                    data.get("rec_text", data.get("text")),
                    data.get("rec_score", data.get("score")),
                    data.get("rec_poly", data.get("polygon")),
                )
            for key in ("res", "overall_ocr_res", "ocr_res"):
                if key in data:
                    walk(data[key])
        elif isinstance(data, (list, tuple)):
            if len(data) == 2 and isinstance(data[1], (list, tuple)) and data[1] and isinstance(data[1][0], str):
                add(data[1][0], data[1][1] if len(data[1]) > 1 else None, data[0])
            else:
                for item in data:
                    walk(item)

    if result is None:
        return []
    if not isinstance(result, (dict, list, tuple, str)) and not hasattr(result, "json"):
        try:
            result = list(result)
        except TypeError:
            return []
    walk(result)
    return deduplicate(out)


def _overlap(a: Detection, b: Detection) -> float:
    if not a.bounds or not b.bounds:
        return 0.0
    ax, ay, ar, ab = a.bounds
    bx, by, br, bb = b.bounds
    intersection = max(0.0, min(ar, br) - max(ax, bx)) * max(0.0, min(ab, bb) - max(ay, by))
    area = min((ar - ax) * (ab - ay), (br - bx) * (bb - by))
    return intersection / area if area > 0 else 0.0


def deduplicate(detections: list[Detection]) -> list[Detection]:
    """Collapse overlapping repeated observations using maximum confidence, never a vote count."""
    out: list[Detection] = []
    for detection in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if any(
            normalize_name(d.text) == normalize_name(detection.text)
            and d.roi == detection.roi
            and _overlap(d, detection) >= 0.8
            for d in out
        ):
            continue
        out.append(detection)
    return out


def group_rows(detections: list[Detection]) -> list[Row]:
    """Reconstruct baselines within each pass, keeping boxless observations self-contained."""
    passes: dict[tuple[str, str, str, str], list[Detection]] = defaultdict(list)
    for d in detections:
        passes[d.pass_id, d.roi, d.language, d.variant].append(d)
    rows = []
    for observations in passes.values():
        groups: list[list[Detection]] = []
        for d in sorted(deduplicate(observations), key=lambda v: (v.bounds or (0, 0, 0, 0))[1]):
            box = d.bounds
            target = None
            for group in groups:
                anchor = group[0].bounds
                if not box or not anchor:
                    continue
                height = min(box[3] - box[1], anchor[3] - anchor[1])
                overlap = min(box[3], anchor[3]) - max(box[1], anchor[1])
                if height > 0 and overlap / height >= 0.55:
                    target = group
                    break
            if target is None:
                groups.append([d])
            else:
                target.append(d)
        for group in groups:
            fragments = sorted(group, key=lambda v: (v.bounds or (0, 0, 0, 0))[0])
            cluster = []
            right = None
            for fragment in fragments:
                box = fragment.bounds
                if box and right is not None and box[0] - right > (box[3] - box[1]) * 4.5:
                    rows.append(Row(tuple(cluster)))
                    cluster = []
                cluster.append(fragment)
                if box:
                    right = max(right if right is not None else box[2], box[2])
            rows.append(Row(tuple(cluster)))
    return rows


def labelled_codes(detections: list[Detection]) -> list[tuple[str, Row]]:
    """Read complete code fields, allowing another pass to establish the same observed label's position."""
    rows = group_rows(detections)
    labels = [row for row in rows if CODE_LABEL.search(row.text) and row.confidence >= 0.65 and row.bounds]
    result = []
    for row in rows:
        if row.confidence < 0.65:
            continue
        values = code_values(row.text)
        values.extend(value for value in _spatial_code_values(row) if value not in values)
        if not values and row.bounds:
            # Recognition can destroy a localized label while preserving its code.
            # Only a complete colon-delimited field at a separately observed code
            # label's screenshot position may use that spatial context.
            field = re.fullmatch(r"[^:\n]{1,12}:\s*([A-Z0-9]{4,6})\s*", row.text, re.IGNORECASE)
            if field and normalize_code(field[1]):
                x1, y1, _, y2 = row.bounds
                for label in labels:
                    assert label.bounds is not None
                    lx, ly, _, lb = label.bounds
                    height = min(y2 - y1, lb - ly)
                    if height > 0 and min(y2, lb) - max(y1, ly) >= height * 0.65 and abs(x1 - lx) <= height * 2:
                        values = [field[1].upper()]
                        break
        result.extend((value, row) for value in values)
    result.extend(candidate for candidate in _metadata_codes(rows) if candidate not in result)
    return result


def _metadata_neighbors(above: Row, below: Row) -> bool:
    """Require aligned, closely spaced metadata rows in screenshot coordinates."""
    a, b = above.bounds, below.bounds
    if not a or not b:
        return False
    ah, bh = a[3] - a[1], b[3] - b[1]
    if ah <= 0 or bh <= 0 or not 0.5 <= bh / ah <= 2.0:
        return False
    gap = b[1] - a[3]
    return -min(ah, bh) * 0.15 <= gap <= max(ah, bh) * 2 and abs(a[0] - b[0]) <= max(ah, bh) * 1.5


def _timer_field(row: Row) -> bool:
    """Recognize a complete labelled timer for metadata position, never completion evidence."""
    label = re.match(TIME_LABEL, row.text, re.IGNORECASE)
    if not label:
        return False
    tail = row.text[label.end() :].strip(" :\uff1a")
    values = time_values(tail)
    return len(values) == 1 and values[0][0].span() == (0, len(tail)) and values[0][0][2] is not None


def _metadata_blocks(rows: list[Row]) -> list[tuple[Row, Row]]:
    """Locate creator/timer anchors whose nearby Discord URL is recognized in the same pass."""
    passes: dict[tuple[str, str, str, str], list[Row]] = defaultdict(list)
    for row in rows:
        first = row.detections[0]
        if first.roi == "topLeft" and row.bounds and row.confidence >= 0.85:
            passes[first.pass_id, first.roi, first.language, first.variant].append(row)
    blocks = []
    for observations in passes.values():
        creators = [row for row in observations if CREATOR_FIELD.fullmatch(row.text)]
        timers = [row for row in observations if _timer_field(row)]
        discord_rows = [row for row in observations if DISCORD_FIELD.fullmatch(row.text)]
        # The observed Korean layout places Discord immediately above creator,
        # followed by the code and timer. All three semantic anchors must be
        # established together in one pass; only the literal code may come from
        # another model at that same screenshot location.
        blocks.extend(
            (creator, timer)
            for creator in creators
            for timer in timers
            if any(_metadata_neighbors(discord, creator) for discord in discord_rows)
        )
    return blocks


def metadata_code_regions(detections: list[Detection]) -> list[MetadataCodeRegion]:
    """Localize noisy code fields for an independent image pass, without verifying their text."""
    rows = group_rows(detections)
    blocks = _metadata_blocks(rows)
    result = []
    for row in rows:
        field = row.bounds
        if row.detections[0].roi != "topLeft" or not field or not NOISY_CODE_FIELD.fullmatch(row.text):
            continue
        for creator, timer in blocks:
            a, b = creator.bounds, timer.bounds
            assert a is not None and b is not None
            height = max(a[3] - a[1], b[3] - b[1])
            center = (field[1] + field[3]) / 2
            # Occlusion can enlarge a low-confidence detection into a nearby
            # timer. Only its center localizes the field; callers must crop
            # between the actual creator bottom and timer top before re-reading.
            if (
                0 < b[1] - a[3] <= height * 3
                and a[3] < center < b[1]
                and height * 0.5 <= field[3] - field[1] <= height * 2
                and abs(field[0] - a[0]) <= height * 1.5
                and abs(field[0] - b[0]) <= height * 1.5
            ):
                result.append(MetadataCodeRegion(row, a, b))
                break
    return result


def _metadata_codes(rows: list[Row]) -> list[tuple[str, Row]]:
    """Verify a noisy localized code field only when its own literal reading is confident."""
    blocks = _metadata_blocks(rows)
    return [
        (code, row)
        for row in rows
        if row.detections[0].roi == "topLeft" and row.bounds and row.confidence >= 0.80
        if (field := NOISY_CODE_FIELD.fullmatch(row.text)) and (code := normalize_code(field[1]))
        if any(_metadata_neighbors(creator, row) and _metadata_neighbors(row, timer) for creator, timer in blocks)
    ]


def _tight_neighbors(left: Detection, right: Detection) -> bool:
    """Require adjoining glyphs of the same size, rather than a shared HUD baseline."""
    a, b = left.bounds, right.bounds
    if not a or not b:
        return False
    ah, bh = a[3] - a[1], b[3] - b[1]
    if ah <= 0 or bh <= 0 or not 0.65 <= bh / ah <= 1.35:
        return False
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    characters = len(re.sub(r"\s", "", left.text))
    average_width = (a[2] - a[0]) / max(1, characters)
    gap = b[0] - a[2]
    return overlap >= min(ah, bh) * 0.65 and 0 <= gap <= min(ah * 0.65, average_width * 0.85)


def _spatial_code_values(row: Row) -> list[str]:
    """Join code suffix detections only when the image places their glyphs together."""
    found = []
    fragments = row.detections
    for index, left in enumerate(fragments[:-1]):
        label = CODE_LABEL.search(left.text)
        fragment = left.text[label.end() :].strip() if label else left.text.strip()
        if not re.fullmatch(r"[A-Z0-9]{1,5}", fragment, re.IGNORECASE):
            continue
        base = fragment.upper()
        if not label:
            if index == 0:
                continue
            previous = fragments[index - 1].text
            previous_label = CODE_LABEL.search(previous)
            if not previous_label or previous[previous_label.end() :].strip():
                continue
        code = base
        current = left
        for right in fragments[index + 1 :]:
            suffix = right.text.strip()
            if not re.fullmatch(r"[A-Z0-9]{1,5}", suffix, re.IGNORECASE) or not _tight_neighbors(current, right):
                break
            code += suffix.upper()
            current = right
            if len(code) > 6:
                code = ""
                break
        if code != base and normalize_code(code):
            found.append(code)
    return found


def _pair_text(row: Row) -> str:
    """Resolve isolated numeric fragments only when their spatial field is unambiguous."""
    fragments = row.detections
    parts = []
    for index, fragment in enumerate(fragments):
        separator = " "
        if index and re.fullmatch(r"\d{1,2}", fragments[index - 1].text.strip()):
            left = fragments[index - 1]
            number_starts_here = TIME.fullmatch(normalize_time_text(fragment.text).strip()) is not None
            # If that isolated digit is also close to a preceding name, the
            # identity/time boundary remains ambiguous and must stay explicit.
            close_to_name = index >= 2 and _tight_neighbors(fragments[index - 2], left)
            if number_starts_here and _tight_neighbors(left, fragment) and not close_to_name:
                separator = ""
        parts.append((separator if parts else "") + fragment.text)
    return "".join(parts)


def paired_evidence(detections: list[Detection], source: str, minimum: float = 10.0) -> list[Pair]:
    """Require same-row pairs or immediately stacked leaderboard fields in a single pass."""
    rows = group_rows(detections)
    pairs = []
    for row in rows:
        for name, cs in text_pairs(_pair_text(row), source, minimum):
            pairs.append(Pair(name, cs, source, row.confidence, (row,)))
    if source != "topRight.top5":
        return pairs
    # Workshop leaderboards often put each player's time directly below their name.
    for i, row in enumerate(rows[:-1]):
        name = clean_name(row.text)
        following = rows[i + 1]
        a, b = row.bounds, following.bounds
        first, second = row.detections[0], following.detections[0]
        same_pass = (first.pass_id, first.roi, first.language, first.variant) == (
            second.pass_id,
            second.roi,
            second.language,
            second.variant,
        )
        if not name or not a or not b or not same_pass:
            continue
        following_text = _pair_text(following)
        values = time_values(following_text, minimum)
        if len(values) != 1 or following_text[: values[0][0].start()].strip(" \t.,|"):
            continue
        height = max(a[3] - a[1], b[3] - b[1])
        if 0 <= b[1] - a[3] <= height * 0.8 and abs(a[0] - b[0]) <= height:
            pairs.append(Pair(name, values[0][1], source, min(row.confidence, following.confidence), (row, following)))
    return pairs
