from __future__ import annotations

import pandas as pd
import pytest

from survey_summary.models import SurveyData, QuestionMeta, build_survey
from survey_summary.segment import segment_stats, SegmentResult


def _make_survey(data: dict, meta: list[dict]) -> SurveyData:
    df = pd.DataFrame(data)
    return build_survey(df, meta=meta)


class TestNumericQuantile:
    def test_quantile_values(self):
        meta = [{"name": "score", "type": "numeric", "label": "成绩"}]
        survey = _make_survey(
            {"score": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
            meta=meta,
        )
        results = segment_stats(survey)
        assert len(results) == 1
        seg = results[0]
        assert seg.qtype == "numeric"
        s = seg.stats
        assert s["min"] == 10.0
        assert s["max"] == 100.0
        assert s["median"] == 55.0
        assert s["q1"] == pytest.approx(32.5, abs=0.5)
        assert s["q3"] == pytest.approx(77.5, abs=0.5)
        assert s["mean"] == 55.0

    def test_quantile_with_nan(self):
        meta = [{"name": "val", "type": "numeric"}]
        survey = _make_survey({"val": [1.0, 2.0, None, 4.0, 5.0]}, meta=meta)
        results = segment_stats(survey)
        seg = results[0]
        assert seg.missing == 1
        assert seg.stats["mean"] == pytest.approx(3.0, abs=0.01)
        assert seg.stats["median"] == 3.0


class TestCategoricalFrequency:
    def test_frequency_pct(self):
        meta = [
            {"name": "color", "type": "categorical", "options": ["红", "绿", "蓝"]},
        ]
        survey = _make_survey(
            {"color": ["红", "红", "红", "绿", "蓝"]},
            meta=meta,
        )
        results = segment_stats(survey)
        seg = results[0]
        freq = seg.stats["frequencies"]
        assert freq["红"]["count"] == 3
        assert freq["红"]["pct"] == pytest.approx(60.0, abs=0.1)
        assert freq["绿"]["count"] == 1
        assert freq["绿"]["pct"] == pytest.approx(20.0, abs=0.1)
        assert freq["蓝"]["count"] == 1
        assert freq["蓝"]["pct"] == pytest.approx(20.0, abs=0.1)
        assert freq["红"]["pct"] + freq["绿"]["pct"] + freq["蓝"]["pct"] == pytest.approx(
            100.0, abs=0.5
        )

    def test_zero_count_option(self):
        meta = [
            {"name": "fruit", "type": "categorical", "options": ["苹果", "香蕉", "橘子"]},
        ]
        survey = _make_survey({"fruit": ["苹果", "苹果", "香蕉"]}, meta=meta)
        results = segment_stats(survey)
        freq = results[0].stats["frequencies"]
        assert freq["橘子"]["count"] == 0
        assert freq["橘子"]["pct"] == 0.0


class TestTextLength:
    def test_text_length_stats(self):
        meta = [{"name": "comment", "type": "text"}]
        survey = _make_survey(
            {"comment": ["你好", "测试数据", "很长很长的反馈内容"]},
            meta=meta,
        )
        results = segment_stats(survey)
        seg = results[0]
        s = seg.stats
        assert s["count"] == 3
        assert s["unique"] == 3
        assert s["avg_len"] == pytest.approx(5.0, abs=0.1)
        assert s["max_len"] == 9

    def test_text_with_nan(self):
        meta = [{"name": "note", "type": "text"}]
        survey = _make_survey({"note": ["短", None, "中等长度"]}, meta=meta)
        results = segment_stats(survey)
        seg = results[0]
        assert seg.missing == 1
        assert seg.stats["count"] == 2
