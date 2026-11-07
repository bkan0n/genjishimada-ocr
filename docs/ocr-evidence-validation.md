# OCR evidence investigation and validation

Baseline: `c798d5a` (`fix_time_parsing` checkout). Work branch: `fix/ocr-evidence-validation`. Validation performed on 2026-11-07. Screenshots and model weights are cached locally and are not committed.

## Root causes

1. `ocr_lines()` discarded Paddle polygons; its 3.x adapter also assumed Python lists for scores. `join_lines()` then erased line boundaries. Banner and leaderboard helpers accumulated multiple variants/languages into one string, permitting cross-pass fragments and incorrect player/time associations. Repeated variants inflated candidate scores.
2. The names flow used `first_confirmed_name` plus an independent `text_contains_time_hint()` search, producing `timeVerifiedBy: ocrText` without a relationship. A second fallback accepted a top-left live timer. The normal flow could exit on the same timer before inspecting the banner or leaderboard.
3. ASCII-only name cleanup discarded diacritics; CJK-only regexes discarded Latin/digit parts of mixed names. Short Korean names were filtered by a three-character minimum in some paths. Japanese/Chinese passes were gated by fast mode, and banner names required English `MISSION COMPLETE`.
4. The banner ROI began at 24% and ended at 76% of screenshot width. In F it cut the beginning of `SNOWMÄN` and the end of the completion text. The replacement ROI covers 10–86% width and 12–40% height; spatial grouping separates distant HUD fragments sharing a baseline.
5. Map-code matching globally collapsed distinct letters/digits, allowed selected one-character substitutions, and searched substrings of arbitrary HUD tokens. Some extraction paths required digits, excluding valid codes such as `TXTMP`. Enlarging the whole HUD often downscaled text again at Paddle's detector size limit; a narrow code-line crop preserves trailing glyphs.
6. Caching by temporary ndarray object ID could reuse stale OCR after Python recycled an ID. Caches now use image content, shape, and language, with request-local lifetime.
7. Absent local model directories left model names unset, selecting Paddle defaults, including server models. Mobile model names are now explicit. Paddle is imported only after CPU environment configuration, allowing deterministic tests without model loading.
8. Cancelling `asyncio.to_thread()` on timeout did not stop native inference. The queue worker could therefore start the next job while the first still ran. It now shields inference and holds the queue slot until native work exits, including on shutdown.

## Implementation and false-positive controls

`evidence.py` contains pure dataclasses and parsing rules. Each observation retains recognized text, confidence, model/language, ROI, preprocessing pass, original input polygon, and screenshot-relative polygon. It handles legacy 2.x blocks, 3.x dictionaries/Result JSON, ndarray scores/polygons, missing metadata, and malformed/empty results.

Name/time rows are reconstructed within a single pass. Horizontal fragments must share a baseline and form a contiguous spatial cluster. Stacked Workshop entries require an immediately following time field, aligned left edges and a bounded vertical gap; an intervening name blocks pairing. Confidence uses the weakest fragment; duplicate observations never add votes. Missing geometry cannot join separate fragments, though a complete recognized name/time string remains usable.

Display spelling is retained. Name comparison alone applies Unicode/case/diacritic normalization and a tightly limited Latin glyph-confusion rule. CJK identity cannot be relaxed just because a time matches. All supplied aliases remain possible identities, but only a supported pair can confirm one.

Codes require a complete literal 4–6 character reading. Localized label recognition supports observed Korean label errors (`맵`/`앱`/`맨`) without changing the code. A label independently located by one pass may establish the screenshot position of another pass's complete colon-delimited code field; unrelated creator/Discord fields at other coordinates are excluded. `FW2K`, `3BSB`, `K1TTI`, `GA3JC`, and `ZJ7G` cannot verify their differing submitted codes by similarity.

A completion banner takes priority over leaderboard evidence for a matching allowed identity. Conflicting time readings remain unconfirmed. A visible top-left timer is never definitive completion evidence. Decimal time matching uses integer centiseconds, with the existing absolute tolerance and valid range. Hints cannot fill missing characters or times, including when the legacy trust-hints flag is enabled.

The API keeps `extracted.name`, `time`, `code`, nested `sources` and `texts`. Added diagnostics explain candidates, geometry/provenance, observed name, mismatch/unconfirmed reasons, and OCR pass count. Detailed observations require `OCR_DEBUG_TEXTS=1`; routine logs contain timing/counts rather than OCR identities.

## Image corpus and expectations

The original report did not provide request aliases for most screenshots. For reproducible identity checks, `tests/corpus.json` records manually transcribed visible names: A/B/C `송자`, E `클라스`, F `Snowmän`, G `MANTA`, H `ぽんぽんた`, I `BELLATRIX`. D uses no alias because its Japanese completion name is uncertain. These are evaluation inputs, never production defaults. In E the test deliberately asks for the visible HUD/banner player `클라스`; the original report's unspecified player cannot justify assuming the separate `536.62` row belongs to that player.

F and I are confirmed positives with assertions on every field. D and E are confirmed mismatches on the disputed field. A/B/C/G/H remain recognition improvement targets: assertions prohibit unsupported returned values but allow null fields, rather than making recognition guarantees from submission hints. In this run A/B/C/H all confirmed successfully. G's code and D's completion identity/time remain unconfirmed.

Both versions below used identical cached PP-OCRv5 mobile weights, PaddleOCR 3.3.3 / PaddlePaddle 3.2.2, Python 3.11 on Windows CPU, four engine threads, and `OCR_MKLDNN=0`. Cases ran sequentially, each in a fresh process after model warming, with a 75-second process watchdog. Times measure pipeline execution and image decoding, excluding model warmup/download. All controlled cases completed without the watchdog firing.

`—` means null/unconfirmed. Values are actual responses, including baseline's altered O/0 spellings.

| Case | Baseline name / time / code | Updated name / time / code | Evidence and outcome | Baseline → updated seconds |
|---|---|---|---|---|
| A | 송자 / 4392.87 / — | 송자 / 4392.87 / FW2KJ | Same-player stacked leaderboard; focused pass independently reads final J. No live-timer fallback. | 27.080 → 10.838 |
| B | 송자 / 2180.92 / TXTMP | 송자 / 2180.92 / TXTMP | Korean localized completion banner; actual name/time relationship. | 13.609 → 10.125 |
| C | 송자 / 2056.06 / — | 송자 / 2056.06 / JBBSH | Korean final-goal banner; literal code read supported by localized label geometry. | 13.131 → 11.636 |
| D | GARLICBREAD / 634.29 / — | GARLICBREAD / — / — | Baseline associated HUD identity with live timer. Updated rejects JZJG against visible ZJ7G; Japanese completion name remains insufficiently recognized. HUD identity is retained, without a time. | 27.533 → 43.572 |
| E | 클라스 / **536.62** / NHY0R | 클라스 / **—** / NHYOR | Reproduced baseline false positive via `ocrText`. 클라스 completes in **557.02**; **536.62 belongs to another row**. Updated rejects the pairing and preserves literal O. | 22.084 → 9.247 |
| F | — / — / WZJ79 | Snowmän / 1092.35 / WZJ79 | Unicode name and full completion banner; recognized spelling SNOWMÄN retained in diagnostics. | 25.752 → 10.182 |
| G | MANTA / 888.08 / — | MANTA / 888.08 / — | Name/time confirmed by banner; no independent complete K1TTT reading, so code stays unconfirmed. | 14.376 → 10.095 |
| H | — / — / 0FF1N | ぽんぽんた / 1153.39 / OFF1N | Japanese same-player stacked leaderboard; literal O preserved. | 46.793 → 11.655 |
| I | BELLATRIX / 530.15 / — | BELLATRIX / 530.15 / 3BSB1 | Named completion banner; focused pass independently reads complete code. | 17.232 → 7.984 |

Six submissions have all fields confirmed: A/B/C/F/H/I. G has a supported name/time but an unconfirmed code. D has unconfirmed submitted code/time; E correctly rejects the submitted time for the specified player.

Mean measured latency decreased from **23.07s to 13.93s**; median from **22.08s to 10.18s**. D is slower because safe recognition fallbacks replace the former live-timer exit. These are one-run local measurements, not a production performance guarantee. Production defaults enable oneDNN, which was disabled for this platform comparison. An earlier uncontrolled server-model baseline exceeded the service timeout on D; it was stopped and is excluded from this table.

The two additional negative examples have no supplied image URLs. Deterministic tests reject Snowmän `351.69` against completion `6910.32`, and CoralMage `27.85` against `185.93`; separate tests reject a matching live timer without completion evidence.

## Validation

- `pytest -q`: **130 passed, 9 skipped**; the skipped tests are the opt-in images. No model downloads are required for unit tests.
- Opt-in real image suite: **9/9 passed** with classified expectations. The final per-process harness also passed all nine image assertions and generated per-image responses/latencies. A target returning null counts as conservative behavior, not a recognition success.
- `pyright`: **0 errors, 0 warnings**. Resolved-interpreter baseline had 7 errors and 2 warnings, including nullable times, incorrect future result type, language literals, and Windows affinity lookup.
- `ruff check .`: **86 remaining findings**, all inherited rules/functions from the original `main.py` (164 baseline findings). New `evidence.py` is clean. Remaining findings are mostly existing docstring style, old helper complexity, long lines, and import/type annotations; they are not hidden by new global suppressions. Existing configuration excludes tests from Ruff.
- `git diff --check`: passed. No images, models, generated responses, virtual environment, or credentials are tracked.
- Regression tests explicitly cover separate-player/time rejection, per-pass boundaries, stacked rows, weak/missing evidence, alias selection, conflicting results, Unicode display preservation, strict code disagreements, oneDNN fallback reuse, and queue serialization after timeout.

## Reproduce

Install `requirements.txt` and dev tools, then see README for ordinary unit/image commands. A reproducible isolated-process comparison is available without a second checkout:

```bash
python tests/benchmark_corpus.py --baseline-ref c798d5a --report-dir .ocr-cache/baseline --fixture-dir .ocr-cache --models /path/to/official_models
python tests/benchmark_corpus.py --report-dir .ocr-cache/updated --fixture-dir .ocr-cache --models /path/to/official_models
```

Use the same `OCR_MKLDNN`, CPU thread count, model directory and image cache for both commands. `--models` is optional if runtime configuration already resolves the same weights. The model directory must contain `PP-OCRv5_mobile_det`, `en_PP-OCRv5_mobile_rec`, `korean_PP-OCRv5_mobile_rec`, and `PP-OCRv5_mobile_rec`. Reports contain source observations and should remain in ignored local storage. A nonzero benchmark exit means a process failed, an assertion failed, or the watchdog fired; consult that case's log.

## Files changed

- `main.py`, `evidence.py`: structured inference, parsing/verification, targeted fallback orchestration, response metadata, and queue safety.
- `tests/test_evidence.py`, `tests/test_pipeline.py`, `tests/test_queue.py`, `tests/conftest.py`: deterministic regressions.
- `tests/corpus.json`, `tests/test_images.py`, `tests/benchmark_corpus.py`: image expectations, opt-in OCR, and isolated baseline capture.
- `README.md`, this report: actual API, configuration, evidence policy, investigation, and measured results.
- `.gitignore`, `pyproject.toml`, `pyrightconfig.json`: ignored caches, test/dev configuration, and Python 3.11 environment discovery.
