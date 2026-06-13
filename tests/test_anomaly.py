from __future__ import annotations

import pandas as pd

from survey_summary.models import SurveyData, QuestionMeta, build_survey
from survey_summary.anomaly import detect_anomalies


def _make_survey(data: dict, meta: list[dict]) -> SurveyData:
    df = pd.DataFrame(data)
    return build_survey(df, meta=meta)


class TestStraightLining:
    def test_straight_line_hit(self):
        meta = [
            {"name": "q1", "type": "categorical"},
            {"name": "q2", "type": "categorical"},
            {"name": "q3", "type": "categorical"},
        ]
        survey = _make_survey(
            {
                "q1": ["A", "A", "B"],
                "q2": ["A", "A", "B"],
                "q3": ["A", "A", "C"],
            },
            meta=meta,
        )
        anomalies = detect_anomalies(survey, straight_line_threshold=0.95)
        sl = [a for a in anomalies if a.reason == "straight_lining"]
        assert len(sl) >= 1
        assert sl[0].row_index == 0


class TestNumericOutlier:
    def test_zscore_outlier_hit(self):
        meta = [{"name": "score", "type": "numeric"}]
        values = [50, 52, 49, 51, 48, 53, 500]
        survey = _make_survey({"score": values}, meta=meta)
        anomalies = detect_anomalies(survey, zscore_threshold=2.0)
        outliers = [a for a in anomalies if a.reason == "numeric_outlier"]
        assert len(outliers) >= 1
        assert 500 in [survey.df.loc[a.row_index, "score"] for a in outliers]


class TestHighMissing:
    def test_high_missing_hit(self):
        meta = [
            {"name": "a", "type": "numeric"},
            {"name": "b", "type": "numeric"},
            {"name": "c", "type": "numeric"},
            {"name": "d", "type": "numeric"},
            {"name": "e", "type": "numeric"},
        ]
        survey = _make_survey(
            {
                "a": [1, 2],
                "b": [1, None],
                "c": [1, None],
                "d": [1, None],
                "e": [1, None],
            },
            meta=meta,
        )
        anomalies = detect_anomalies(survey, missing_ratio_threshold=0.6)
        high_miss = [a for a in anomalies if a.reason == "high_missing"]
        assert len(high_miss) == 1
        assert high_miss[0].row_index == 1
