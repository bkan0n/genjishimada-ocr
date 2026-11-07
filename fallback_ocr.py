"""Bounded, hint-independent Korean word recognition with an optional local engine."""

from __future__ import annotations

import csv
import io
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

Bounds = tuple[float, float, float, float]


@dataclass(frozen=True)
class WordPiece:
    """One actual TSV segment with source bounds and optional raw engine-input bounds.

    Padded-mask source bounds exclude artificial border pixels; local_bounds
    retains the exact scaled TSV rectangle, including logical box extensions.
    """

    text: str
    confidence: float
    bounds: Bounds
    local_bounds: Bounds | None = None


@dataclass(frozen=True)
class WordRead:
    """One complete word read from a single actual name-cell image."""

    text: str
    confidence: float
    bounds: Bounds
    pieces: tuple[WordPiece, ...]
    scale: float = 1.0
    model: str = "tesseract-best-kor"
    origin: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class PairedNameCrop:
    """A time-colour-derived name mask and its original image geometry."""

    image: np.ndarray
    bounds: Bounds
    border: int
    hue: int
    colour_dominance: float


def _scaled_bounds(bounds: Bounds, scale: float) -> Bounds:
    """Scale a four-coordinate rectangle without losing its fixed tuple shape."""
    x1, y1, x2, y2 = bounds
    return x1 / scale, y1 / scale, x2 / scale, y2 / scale


def _offset_bounds(bounds: Bounds, x: float, y: float) -> Bounds:
    """Translate coordinates to another image origin."""
    x1, y1, x2, y2 = bounds
    return x1 + x, y1 + y, x2 + x, y2 + y


def _korean_word_rows(tsv: str) -> list[tuple[str, float, Bounds]] | None:
    """Parse complete confident Hangul token boxes from one TSV response."""
    words = []
    marks = []
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t"):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            confidence = float(row["conf"]) / 100
            x, y = float(row["left"]), float(row["top"])
            width, height = float(row["width"]), float(row["height"])
        except (ValueError, KeyError, TypeError):
            return None
        valid = all(math.isfinite(value) for value in (confidence, x, y, width, height))
        if not valid or min(x, y) < 0 or width <= 0 or height <= 0:
            return None
        if not all("\uac00" <= char <= "\ud7a3" for char in text):
            # Never erase a digit, underscore or another script to manufacture
            # a simpler identity from a mixed-script name.
            if any(char.isalnum() or char in "_-'" for char in text):
                return None
            marks.append((x, y, x + width, y + height))
            continue
        if not 0.85 <= confidence <= 1:
            return None
        words.append((text, confidence, (x, y, x + width, y + height)))
    if words and any(
        min(word[2][0] for word in words) < (mark[0] + mark[2]) / 2 < max(word[2][2] for word in words)
        for mark in marks
    ):
        return None
    return sorted(words, key=lambda word: word[2][0])


def parse_korean_word_tsv(tsv: str, scale: float = 1.0) -> WordRead | None:
    """Join only adjoining Hangul segments from one pass, using minimum confidence."""
    if not math.isfinite(scale) or scale <= 0:
        return None
    words = _korean_word_rows(tsv)
    if not words:
        return None
    for left, right in pairwise(words):
        a, b = left[2], right[2]
        height = min(a[3] - a[1], b[3] - b[1])
        aligned = min(a[3], b[3]) - max(a[1], b[1]) >= height * 0.55
        width = min(a[2] - a[0], b[2] - b[0])
        # Tesseract Korean may give overlapping logical token boxes inside a
        # single word. Both edges must advance; duplicates/contained boxes do not.
        ordered = a[0] < b[0] and a[2] < b[2]
        adjacent = -width * 0.85 <= b[0] - a[2] <= max(4, height * 0.65)
        if not aligned or not adjacent or not ordered:
            return None
    text = "".join(word[0] for word in words)
    if not 2 <= len(text) <= 16:
        return None
    bounds = (
        min(word[2][0] for word in words) / scale,
        min(word[2][1] for word in words) / scale,
        max(word[2][2] for word in words) / scale,
        max(word[2][3] for word in words) / scale,
    )
    pieces = tuple(WordPiece(word[0], word[1], _scaled_bounds(word[2], scale), word[2]) for word in words)
    return WordRead(text, min(word[1] for word in words), bounds, pieces, scale)


def recognize_korean_word(
    image: np.ndarray,
    model_dir: Path,
    timeout: float = 4.0,
    *,
    scales: tuple[float, ...] = (1.0, 3.0),
) -> WordRead | None:
    """Read a small observed name cell; never accept a supplied name or search a full screenshot."""
    executable = shutil.which("tesseract")
    if not executable or image.size == 0 or not (model_dir / "kor.traineddata").is_file():
        return None
    if not math.isfinite(timeout) or timeout <= 0 or not scales or len(scales) > 2:
        return None
    if any(not math.isfinite(scale) or scale <= 0 for scale in scales):
        return None
    deadline = perf_counter() + timeout
    results = []
    for scale in scales:
        remaining = deadline - perf_counter()
        if remaining <= 0:
            break
        crop = image if scale == 1 else cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        encoded, buffer = cv2.imencode(".png", crop)
        if not encoded:
            continue
        command = [
            executable,
            "stdin",
            "stdout",
            "--tessdata-dir",
            str(model_dir),
            "-l",
            "kor",
            "--psm",
            "7",
            "-c",
            "tessedit_create_tsv=1",
        ]
        environment = dict(os.environ, OMP_THREAD_LIMIT="1")
        try:
            result = subprocess.run(
                command,
                input=buffer.tobytes(),
                capture_output=True,
                timeout=remaining,
                check=False,
                env=environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            word = parse_korean_word_tsv(result.stdout.decode("utf-8", errors="replace"), scale)
            if word is not None:
                results.append(word)
    # No votes or confidence accumulation from duplicate preprocessing variants.
    return max(results, key=lambda word: word.confidence) if results else None


def _entry_name_bounds(image: np.ndarray, name: Bounds, time: Bounds) -> Bounds | None:
    """Require adjoining aligned cells and clip the name crop above the time cell."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        return None
    height, width = image.shape[:2]
    for bounds in (name, time):
        if not all(math.isfinite(value) for value in bounds):
            return None
        x1, y1, x2, y2 = bounds
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            return None
    nh, th = name[3] - name[1], time[3] - time[1]
    if min(nh, th) < 10 or time[1] < name[1] + nh * 0.3:
        return None
    if not -0.25 * nh <= time[1] - name[3] <= 1.2 * nh or abs(name[0] - time[0]) > 1.25 * nh:
        return None
    margin = max(2, round(nh * 0.30))
    left = max(0, math.floor(name[0]) - margin)
    right = min(width, math.ceil(name[2]) + margin)
    top = max(0, math.floor(name[1]) - max(2, round(nh * 0.10)))
    bottom = min(height, math.ceil(name[3] + nh * 0.15), math.floor(time[1]))
    return float(left), float(top), float(right), float(bottom)


def _time_colour(image: np.ndarray, bounds: Bounds) -> tuple[int, float, float, float] | None:
    """Infer a dominant bright saturated palette from observed time-cell pixels."""
    x1, y1, x2, y2 = bounds
    crop = image[math.floor(y1) : math.ceil(y2), math.floor(x1) : math.ceil(x2)]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    eligible = (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 190)
    pixels = hsv[eligible]
    if len(pixels) < max(24, crop.shape[0] * crop.shape[1] * 0.015):
        return None
    histogram = np.bincount(pixels[:, 0], minlength=180)
    mass = np.zeros(180, dtype=np.int64)
    for delta in range(-8, 9):
        mass += np.roll(histogram, delta)
    centre = int(np.argmax(mass))
    dominance = float(mass[centre] / len(pixels))
    if dominance < 0.70:
        return None
    difference = np.abs(pixels[:, 0].astype(float) - centre)
    palette = pixels[np.minimum(difference, 180 - difference) <= 8]
    saturation, value = float(np.median(palette[:, 1])), float(np.median(palette[:, 2]))
    if saturation < 120 or value < 170:
        return None
    return centre, saturation, value, dominance


def paired_korean_name_crop(image: np.ndarray, name_bounds: Bounds, time_bounds: Bounds) -> PairedNameCrop | None:
    """Extract actual same-colour name strokes without changing their shape.

    The caller establishes the name/time relationship before using this helper.
    The palette comes exclusively from the time cell; no text or hint is accepted.
    White and ambiguous palettes are deliberately unsupported. Components left of
    the observed name are excluded to remove adjacent rank frames. No morphology
    or stroke synthesis is applied, and the time cell is excluded from the output.
    """
    bounds = _entry_name_bounds(image, name_bounds, time_bounds)
    if bounds is None or (palette := _time_colour(image, time_bounds)) is None:
        return None
    hue, saturation, value, dominance = palette
    left, top, right, bottom = (int(coordinate) for coordinate in bounds)
    colour = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    difference = np.abs(colour[:, :, 0].astype(float) - hue)
    mask = (
        (np.minimum(difference, 180 - difference) <= 10)
        & (colour[:, :, 1] >= max(85, saturation * 0.50))
        & (colour[:, :, 2] >= max(100, value * 0.55))
    ).astype(np.uint8)
    count, labels, _, centroids = cv2.connectedComponentsWithStats(mask)
    foreground = np.zeros_like(mask)
    for index in range(1, count):
        if centroids[index][0] + left >= name_bounds[0]:
            foreground[labels == index] = 1
    if not 0.03 <= float(np.mean(foreground)) <= 0.65:
        return None
    border = 10
    prepared = cv2.copyMakeBorder(
        255 - foreground * 255, border, border, border, border, cv2.BORDER_CONSTANT, value=255
    )
    return PairedNameCrop(prepared, bounds, border, hue, dominance)


def recognize_paired_korean_word(
    image: np.ndarray,
    name_bounds: Bounds,
    time_bounds: Bounds,
    model_dir: Path,
    timeout: float = 4.0,
) -> WordRead | None:
    """Read a coloured paired name cell, returning coordinates in the supplied image."""
    prepared = paired_korean_name_crop(image, name_bounds, time_bounds)
    if prepared is None:
        return None
    word = recognize_korean_word(prepared.image, model_dir, timeout, scales=(1.0,))
    if word is None:
        return None
    left, top, right, bottom = prepared.bounds
    x, y = left - prepared.border, top - prepared.border
    if not word.pieces or not math.isfinite(word.scale) or word.scale <= 0:
        return None
    pieces = []
    for piece in word.pieces:
        if not 0.85 <= piece.confidence <= 1:
            return None
        px1, py1, px2, py2 = _offset_bounds(piece.bounds, x, y)
        clipped = max(left, px1), max(top, py1), min(right, px2), min(bottom, py2)
        area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
        covered = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
        if not area or covered < area * 0.50:
            return None
        local = piece.local_bounds or _scaled_bounds(piece.bounds, 1 / word.scale)
        pieces.append(WordPiece(piece.text, piece.confidence, clipped, local))
    bounds = (
        min(piece.bounds[0] for piece in pieces),
        min(piece.bounds[1] for piece in pieces),
        max(piece.bounds[2] for piece in pieces),
        max(piece.bounds[3] for piece in pieces),
    )
    return WordRead(
        word.text, min(piece.confidence for piece in pieces), bounds, tuple(pieces), word.scale, word.model, (x, y)
    )
