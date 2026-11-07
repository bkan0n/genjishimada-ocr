"""Deterministic recognition/verification regressions; no Paddle import or downloads."""

import math
from dataclasses import replace

import pytest

import evidence as e


def detection(text, x=0, y=0, w=100, h=16, **kwargs):
    return e.Detection(
        text,
        kwargs.pop("confidence", 0.98),
        ((x, y), (x + w, y), (x + w, y + h), (x, y + h)),
        roi=kwargs.pop("roi", "topRight"),
        pass_id=kwargs.pop("pass_id", "raw:en"),
        **kwargs,
    )


@pytest.mark.parametrize("name", ["SNOWMÄN", "송자", "あくす", "小明", "한글_Player44", "猫_42", "Étoile", "カタカナ"])
def test_unicode_display_and_adjacent_time(name):
    assert e.text_pairs(f"{name} 1153.39 SEC") == [(name, 115339)]
    assert e.clean_name(name) == name


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("SNOWMÄN", "Snowmän"),
        ("SNOWMA\u0308N", "SNOWMÄN"),
        ("한글_Player44", "한글_player44"),
        ("ＡＢＣ", "abc"),
        ("SN0WMAN", "SNOWMAN"),
    ],
)
def test_name_comparison_normalization(left, right):
    assert e.name_score(left, right) >= 0.9


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("송자", "쏭자"),
        ("猫_42", "猫_43"),
        ("Alice", "Malice"),
        ("Player44", "Player45"),
        ("한글Player", "한글Other"),
        ("りん", "りんご"),
        ("一郎", "二郎"),
    ],
)
def test_distinct_names_do_not_match(left, right):
    assert e.name_score(left, right) < 0.9


@pytest.mark.parametrize(
    "marker",
    ["MISSION COMPLETE!", "任务完成", "任務完成", "ミッション完了", "ミッションクリア", "미션 완료", "임무 완료"],
)
def test_localized_banner(marker):
    assert e.text_pairs(f"猫_Player {marker} 用时 6910.32 SEC", "banner") == [("猫_Player", 691032)]


@pytest.mark.parametrize(
    "text",
    [
        "TIME 888.08 SEC",
        "用时 6910.32 SEC",
        "시간 483,04 초",
        "時間 483：04 秒",
        "TIME483.04SEC",
        "TIME 483 . 04 SEC",
        "TIME 483 04 SEC",
        "TIME 4 830.04 SEC",
    ],
)
def test_noisy_time_fields(text):
    assert len(e.time_values(text)) == 1


@pytest.mark.parametrize(
    ("a", "b", "match"),
    [
        (351.69, 6910.32, False),
        (27.85, 185.93, False),
        (4392.87, 4392.87, True),
        (530.15, 530.16, False),
        (None, 530.15, False),
        (math.nan, 530.15, False),
        (math.inf, 530.15, False),
        (0, 0, False),
        (530.151, 530.15, False),
    ],
)
def test_strict_centisecond_comparison(a, b, match):
    assert e.times_equal(a, b) is match


def test_absolute_tolerance_without_float_error():
    assert e.times_equal(530.15, 530.16, 0.01)
    assert not e.times_equal(530.15, 530.17, 0.01)


@pytest.mark.parametrize("code", ["TXTMP", "NHYOR", "FW2KJ", "JZJG", "OFF1N", "3BSB1", "ABCD", "123456"])
def test_literal_codes(code):
    assert e.code_values(f"MAP CODE: {code}") == [code]
    assert e.normalize_code(code) == code


@pytest.mark.parametrize("label", ["맵 코드:", "地图代码：", "地圖代碼:", "マップコード:", "MAP C0DE:"])
def test_localized_code_labels(label):
    assert e.code_values(f"{label} JBBSH") == ["JBBSH"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("MAP CODE: FW2 KJ", ["FW2KJ"]),
        ("MAP CODE: 3BSB 1", ["3BSB1"]),
        ("MADE BY: KUMA DISCORD: ABCDE TIME 185.93 SEC", []),
        ("PLAYER_FW2KJ", []),
        ("MAP CODE: FW2KJEXTRA", []),
        ("MAP CODE: ABCDE DISCORD: FW2KJ", ["ABCDE"]),
    ],
)
def test_code_context_and_split_tokens(text, expected):
    assert e.code_values(text) == expected


def test_same_row_and_split_name_segments():
    ds = [detection("SNOW", 0), detection("MÄN", 55, w=45), detection("1092.35 SEC", 120)]
    pairs = e.paired_evidence(ds, "topRight.top5")
    assert [(e.normalize_name(p.name), p.centiseconds) for p in pairs] == [("snowman", 109235)]


def test_neighbor_rows_never_cross_pair():
    ds = [
        detection("Alice", 0, 0),
        detection("557.02 SEC", 150, 0),
        detection("Bob", 0, 25),
        detection("536.62 SEC", 150, 25),
    ]
    assert [(p.name, p.centiseconds) for p in e.paired_evidence(ds, "topRight.top5")] == [
        ("Alice", 55702),
        ("Bob", 53662),
    ]


def test_stacked_leaderboard_entries():
    ds = [
        detection("SNOWMÄN", 10, 0),
        detection("1092.35 SEC", 10, 20),
        detection("Other", 10, 44),
        detection("557.02 SEC", 10, 64),
    ]
    assert [(p.name, p.centiseconds) for p in e.paired_evidence(ds, "topRight.top5")] == [
        ("SNOWMÄN", 109235),
        ("Other", 55702),
    ]


def test_stacked_time_cannot_skip_an_intervening_name():
    ds = [detection("Alice", 10, 0), detection("Bob", 10, 20), detection("536.62 SEC", 10, 40)]
    assert [(p.name, p.centiseconds) for p in e.paired_evidence(ds, "topRight.top5")] == [("Bob", 53662)]


def test_no_cross_variant_or_language_fragments():
    ds = [
        detection("Alice", 0, 0, pass_id="raw:en"),
        detection("536.62 SEC", 150, 0, pass_id="white:en"),
        detection("557.02 SEC", 150, 0, pass_id="raw:ch"),
    ]
    assert e.paired_evidence(ds, "topRight.top5") == []


def test_boxless_fragments_cannot_form_a_row():
    ds = [e.Detection("Alice", 0.99), e.Detection("536.62 SEC", 0.99)]
    assert e.paired_evidence(ds, "topRight.top5") == []
    assert len(e.paired_evidence([e.Detection("Alice 536.62 SEC", 0.99)], "topRight.top5")) == 1


def test_duplicate_variants_do_not_boost_confidence():
    d = detection("Alice 536.62 SEC", confidence=0.7)
    duplicates = [replace(d, variant=str(i), pass_id=str(i)) for i in range(10)]
    assert len(e.deduplicate(duplicates)) == 1
    assert all(p.confidence == 0.7 for p in e.paired_evidence(duplicates, "topRight.top5"))


def test_paddle3_ndarray_scores_polygons_and_coordinate_transform():
    import numpy as np

    result = {
        "res": {
            "rec_texts": ["Alice", "536.62 SEC", "trailing"],
            "rec_scores": np.array([0.95, 0.93]),
            "rec_polys": np.array([[[2, 4], [22, 4], [22, 24], [2, 24]], [[30, 4], [50, 4], [50, 24], [30, 24]]]),
        }
    }
    ds = e.parse_ocr_result(
        result,
        offset=(100, 200),
        scale=(2, 2),
        roi="topRight",
        language="en",
        model="mobile",
        variant="upscale",
        pass_id="one",
    )
    assert len(ds) == 3
    assert ds[0].bounds == (101, 202, 111, 212)
    assert ds[0].local_polygon is not None and ds[0].local_polygon[0] == (2, 4)
    assert ds[0].roi == "topRight" and ds[0].language == "en" and ds[0].model == "mobile"
    assert ds[2].confidence == 0 and ds[2].polygon is None


def test_paddle2_and_serialized_paddle3():
    ds = e.parse_ocr_result([[[[[0, 0], [10, 0], [10, 10], [0, 10]], ("Alice", 0.9)]]])
    assert ds[0].text == "Alice" and ds[0].confidence == 0.9

    class Result:
        json = '{"res":{"rec_texts":["Bob"],"rec_scores":[0.8]}}'

    assert e.parse_ocr_result([Result()])[0].text == "Bob"


@pytest.mark.parametrize(
    "text",
    ["Alice\nBob 536.62 SEC", "Alice Bob 536.62 SEC", "Alice HOLD LEADERBOARD Bob 536.62 SEC", "Alice\n536.62 SEC"],
)
def test_text_does_not_search_for_hint_then_unrelated_time(text):
    assert not any(e.name_score(n, "Alice") >= 0.9 for n, _ in e.text_pairs(text))


@pytest.mark.parametrize("text", ["Alice TIME 27.85 SEC", "Alice SPLIT 536.62 SEC", "Alice CHECKPOINT 536.62 SEC"])
def test_hud_and_split_numbers_are_not_leaderboard_pairs(text):
    assert e.text_pairs(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "송자 님이 최종 목표에 도달했습니다!시간2056.06 SEC",
        "송자 이[가]앱을 깨셨습니다! 시간 2180.92 초",
    ],
)
def test_captured_korean_completion_templates(text):
    assert e.text_pairs(text, "banner")[0][0] == "송자"


def test_localized_code_label_geometry_can_corrobate_an_independent_literal_read():
    label = detection("맨코드:IBBSH", 0, 0, roi="topLeft", pass_id="korean")
    literal = detection("HC:JBBSH", 0, 0, roi="topLeft", pass_id="ch")
    elsewhere = detection("BY:ABCDE", 0, 40, roi="topLeft", pass_id="en")
    assert {value for value, _ in e.labelled_codes([label, literal, elsewhere])} == {"IBBSH", "JBBSH"}


def test_username_suffix_is_not_a_code_label():
    assert e.code_values("USERNAME_UNICODE: ABCDE") == []


def test_stacked_fields_with_same_id_but_different_language_cannot_pair():
    ds = [detection("Alice", 10, 0, language="en"), detection("536.62 SEC", 10, 20, language="ch")]
    assert e.paired_evidence(ds, "topRight.top5") == []


def test_distant_hud_text_does_not_join_a_banner_baseline():
    ds = [
        detection("TIME 536.62 SEC", 0, 0, w=100, roi="banner"),
        detection("Alice MISSION COMPLETE! TIME 557.02 SEC", 500, 0, w=400, roi="banner"),
    ]
    assert [(p.name, p.centiseconds) for p in e.paired_evidence(ds, "banner")] == [("Alice", 55702)]


def test_long_banner_fragments_form_one_contiguous_spatial_line():
    ds = [
        detection("Alice", 0, 0, w=50, h=20, roi="banner"),
        detection("MISSION COMPLETE! TIME", 65, 3, w=210, h=20, roi="banner"),
        detection("536.62 SEC", 285, 1, w=150, h=20, roi="banner"),
    ]
    assert [(p.name, p.centiseconds) for p in e.paired_evidence(ds, "banner")] == [("Alice", 53662)]


def test_comparison_normalization_does_not_change_the_displayed_identity():
    name = "ＳＮＯＷＭÄＮ_４４"
    assert e.text_pairs(name + " １０９２.３５ SEC") == [(name, 109235)]
    assert e.normalize_name(name) == "snowman_44"


@pytest.mark.parametrize("result", [None, 42, [], {}])
def test_empty_or_unsupported_ocr_results_are_safe(result):
    assert e.parse_ocr_result(result) == []
