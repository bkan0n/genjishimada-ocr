"""Pipeline control flow regressions with deterministic OCR observations."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import main
from evidence import Detection


def observation(text, roi, y=0, confidence=0.99, **kwargs):
    return Detection(
        text, confidence, ((0, y), (300, y), (300, y + 16), (0, y + 16)), roi=roi, pass_id=f"{roi}:en:raw", **kwargs
    )


@pytest.fixture
def pipeline(monkeypatch):
    def run(regions, *, names=None, time=536.62, code="ABCDE"):
        def read(image, language, *, roi, **kwargs):
            return regions.get(roi, [])

        monkeypatch.setattr(main, "read_roi_evidence", read)
        monkeypatch.setattr(main, "OCR_TIMEOUT_S", 60)
        return main.run_ocr_pipeline(np.zeros((200, 400, 3), np.uint8), code=code, time=time, names=names or [])[
            "extracted"
        ]

    return run


def test_same_row_confirmation_and_response_compatibility(pipeline):
    r = pipeline(
        {
            "topLeft": [observation("MAP CODE: ABCDE", "topLeft")],
            "topRight": [observation("SNOWMÄN 536.62 SEC", "topRight")],
        },
        names=["Missing", "Snowmän"],
    )
    assert (r["name"], r["time"], r["code"]) == ("Snowmän", 536.62, "ABCDE")
    assert r["sources"]["timeVerifiedBy"] == "topRight.top5"
    assert {"texts", "sources", "name", "time", "code"} <= r.keys()
    assert r["diagnostics"]["observedName"] == "SNOWMÄN"


def test_name_in_one_row_and_expected_time_in_another_rejected(pipeline):
    r = pipeline(
        {"topRight": [observation("Alice 557.02 SEC", "topRight"), observation("Bob 536.62 SEC", "topRight", 30)]},
        names=["Alice"],
    )
    assert r["name"] == "Alice" and r["time"] is None
    assert "time_mismatch" in r["diagnostics"]["reasons"]


@pytest.mark.parametrize("names", [[], ["Alice"]])
def test_live_timer_cannot_trigger_early_success(pipeline, monkeypatch, names):
    monkeypatch.setattr(main, "OCR_EARLY_EXIT", True)
    r = pipeline(
        {
            "topLeft": [observation("MAP CODE: ABCDE TIME 536.62 SEC", "topLeft")],
            "bottomLeft": [observation("Alice", "bottomLeft")],
        },
        names=names,
    )
    assert r["name"] == "Alice" and r["time"] is None
    assert "timeVerifiedBy" not in r["sources"]


@pytest.mark.parametrize(("name", "submitted", "actual"), [("Snowmän", 351.69, 6910.32), ("CoralMage", 27.85, 185.93)])
def test_reported_negative_time_examples(pipeline, name, submitted, actual):
    r = pipeline(
        {"banner": [observation(f"{name} MISSION COMPLETE! TIME {actual} SEC", "banner")]}, names=[name], time=submitted
    )
    assert r["name"] == name and r["time"] is None
    assert "time_mismatch" in r["diagnostics"]["reasons"]


@pytest.mark.parametrize(
    ("observed", "hint"),
    [
        ("3BSB", "3BSB1"),
        ("K1TTI", "K1TTT"),
        ("GA3JC", "GA3JG"),
        ("FW2K", "FW2KJ"),
        ("ZJ7G", "JZJG"),
        ("OFF1N", "0FF1N"),
        ("A1BCD", "AIBCD"),
    ],
)
def test_map_codes_remain_distinct(pipeline, observed, hint):
    r = pipeline({"topLeft": [observation(f"MAP CODE: {observed}", "topLeft")]}, code=hint)
    assert r["code"] is None
    assert not main.codes_match_for_hint(observed, hint)


def test_full_code_supported_by_targeted_pass_can_confirm(pipeline):
    ds = [
        observation("MAP CODE: 3BSB", "topLeft"),
        Detection("MAP CODE: 3BSB1", 0.99, roi="topLeft", variant="upscale", pass_id="code-target"),
    ]
    r = pipeline({"topLeft": ds}, code="3BSB1")
    assert r["code"] == "3BSB1"


def test_unrelated_username_is_not_code(pipeline):
    r = pipeline({"topLeft": [observation("MADE BY: ABCDE", "topLeft")]})
    assert r["code"] is None


def test_aliases_and_trust_hints_never_supply_missing_evidence(pipeline, monkeypatch):
    monkeypatch.setattr(main, "OCR_TRUST_HINTS", True)
    r = pipeline({}, names=["Alice", "Bob"])
    assert r["name"] is None and r["time"] is None and r["code"] is None
    assert main.prefer_provided_time(None, 536.62) is None


def test_wrong_name_with_correct_time_rejected(pipeline):
    r = pipeline({"banner": [observation("Bob MISSION COMPLETE! TIME 536.62 SEC", "banner")]}, names=["Alice"])
    assert r["name"] is None and r["time"] is None


def test_weak_ocr_cannot_verify(pipeline):
    r = pipeline(
        {"banner": [observation("Alice MISSION COMPLETE! TIME 536.62 SEC", "banner", confidence=0.2)]}, names=["Alice"]
    )
    assert r["time"] is None


def test_conflicting_same_player_times_remain_unconfirmed(pipeline):
    ds = [observation("Alice 536.62 SEC", "topRight"), observation("Alice 557.02 SEC", "topRight", 30)]
    assert pipeline({"topRight": ds}, names=["Alice"])["time"] is None


def test_completion_banner_has_priority_over_live_and_old_leaderboard(pipeline):
    r = pipeline(
        {
            "banner": [observation("Alice MISSION COMPLETE! TIME 557.02 SEC", "banner")],
            "topRight": [observation("Alice 536.62 SEC", "topRight")],
            "topLeft": [observation("TIME 536.62 SEC", "topLeft")],
        },
        names=["Alice"],
    )
    assert r["time"] is None


def test_no_hint_uses_observed_completion_name(pipeline):
    r = pipeline({"banner": [observation("猫_Player MISSION COMPLETE! TIME 536.62 SEC", "banner")]}, time=None, code="")
    assert r["name"] == "猫_Player" and r["time"] == 536.62


@pytest.mark.parametrize("time", [float("nan"), float("inf"), -1, 1, 20000])
def test_invalid_time_hint_cannot_confirm(pipeline, time):
    r = pipeline(
        {"banner": [observation("Alice MISSION COMPLETE! TIME 536.62 SEC", "banner")]}, names=["Alice"], time=time
    )
    assert r["time"] is None


def test_lightweight_import_does_not_load_paddle():
    # main uses a local PaddleOCR import only when an engine is built.
    subprocess.run(
        [sys.executable, "-c", "import sys, main; assert 'paddleocr' not in sys.modules"],
        check=True,
    )


def test_content_cache_reuses_copies_and_distinguishes_pixels(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "ocr_lines", lambda image, lang: calls.append(image.copy()) or [("Alice", 0.9)])
    cache = {}
    a = np.zeros((10, 10), np.uint8)
    main.ocr_lines_cached(a, "en", cache)
    main.ocr_lines_cached(a.copy(), "en", cache)
    main.ocr_lines_cached(a + 1, "en", cache)
    assert len(calls) == 2


def test_geometry_is_projected_from_enlarged_roi(monkeypatch):
    d = Detection("Alice", 0.9, ((0, 0), (20, 0), (20, 10), (0, 10)), ((0, 0), (20, 0), (20, 10), (0, 10)))
    monkeypatch.setattr(main, "ocr_detections", lambda image, lang: [d])
    result = main.read_roi_evidence(
        np.zeros((40, 80, 3), np.uint8),
        "en",
        roi="banner",
        offset=(100, 200),
        original_shape=(20, 40, 3),
        variant="upscale",
        cache={},
    )
    assert result[0].bounds == (100, 200, 110, 205)
    assert result[0].local_polygon == d.local_polygon


def test_unknown_player_cannot_use_an_arbitrary_leaderboard_entry(pipeline):
    r = pipeline({"topRight": [observation("Alice 536.62 SEC", "topRight")]}, time=536.62)
    assert r["time"] is None


def test_extra_fractional_digits_are_not_rounded_into_a_match(pipeline):
    r = pipeline(
        {"banner": [observation("Alice MISSION COMPLETE! TIME 536.62 SEC", "banner")]}, names=["Alice"], time=536.621
    )
    assert r["time"] is None


def test_api_model_retains_nested_texts_sources_and_diagnostics(pipeline):
    result = pipeline({}, names=["Alice"])
    serialized = main.ApiResponse.model_validate({"extracted": result}).model_dump(by_alias=True)
    assert serialized["extracted"]["sources"] == result["sources"]
    assert serialized["extracted"]["diagnostics"] == result["diagnostics"]


def test_later_allowed_alias_can_match_its_own_supported_row(pipeline):
    r = pipeline(
        {
            "topRight": [
                observation("OlderAlias 557.02 SEC", "topRight"),
                observation("CurrentAlias 536.62 SEC", "topRight", 30),
            ]
        },
        names=["OlderAlias", "CurrentAlias"],
    )
    assert r["name"] == "CurrentAlias" and r["time"] == 536.62


def test_low_configured_threshold_does_not_make_an_absent_name_match():
    assert not main.names_match_strict(None, "Alice", threshold=0)
    assert not main.names_match_strict("Bob", "Alice", threshold=0)


def test_text_helper_does_not_cross_another_player():
    assert main.extract_time_after_name_hint_from_text("Alice Bob 536.62 SEC", "Alice") is None
    assert main.extract_time_after_name_hint_from_text("SNOWMÄN 1092.35 SEC", "Snowmän") == 1092.35


def test_banner_helper_cannot_combine_name_from_one_variant_with_another_players_time(monkeypatch):
    def read(image, language, *, variant, **kwargs):
        text = "Alice MISSION COMPLETE!" if variant == "banner0" else "Bob MISSION COMPLETE! TIME 536.62 SEC"
        return [observation(text, "banner")]

    monkeypatch.setattr(main, "read_roi_evidence", read)
    value, _, _ = main.extract_confirmed_time_from_banner(np.zeros((10, 10, 3), np.uint8), {}, "Alice")
    assert value is None


def test_onednn_retry_preserves_evidence_and_reuses_fallback_engine(monkeypatch):
    def broken(image):
        raise RuntimeError("onednn_instruction")

    fallback = SimpleNamespace(
        predict=lambda image: [
            {
                "rec_texts": ["Alice 536.62 SEC"],
                "rec_scores": [0.99],
                "rec_polys": [[[0, 0], [10, 0], [10, 10], [0, 10]]],
            }
        ]
    )
    builds = []
    monkeypatch.setattr(main, "OCR_MKLDNN", True)
    monkeypatch.setattr(main, "OCR_ENGINES_NO_MKLDNN", {})
    monkeypatch.setattr(main, "get_ocr_engine", lambda language: SimpleNamespace(predict=broken))
    monkeypatch.setattr(main, "_build_ocr_engine", lambda *args, **kwargs: builds.append(kwargs) or fallback)
    for _ in range(2):
        ds = main.ocr_detections(np.zeros((10, 10, 3), np.uint8), "en")
        assert ds[0].text == "Alice 536.62 SEC" and ds[0].bounds == (0, 0, 10, 10)
    assert len(builds) == 1 and builds[0]["enable_mkldnn"] is False
