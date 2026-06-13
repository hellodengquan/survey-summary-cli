from __future__ import annotations

import csv
import io

import pandas as pd
import pytest

from survey_summary.models import SurveyData, QuestionMeta, build_survey
from survey_summary.segment import segment_stats
from survey_summary.anomaly import detect_anomalies, Anomaly
from survey_summary.report import generate_report


def _make_survey(data: dict, meta: list[dict]) -> SurveyData:
    df = pd.DataFrame(data)
    return build_survey(df, meta=meta)


def _make_segments_and_anomalies(survey: SurveyData):
    segments = segment_stats(survey)
    anomalies = detect_anomalies(survey, zscore_threshold=2.0)
    return segments, anomalies


class TestFormatTextConsistency:
    def test_format_text_matches_default(self):
        meta = [
            {"name": "score", "type": "numeric"},
            {"name": "gender", "type": "categorical", "options": ["男", "女"]},
            {"name": "comment", "type": "text"},
        ]
        survey = _make_survey(
            {
                "score": [10, 20, 30, 40, 50],
                "gender": ["男", "女", "男", "女", "男"],
                "comment": ["好", "不错", "很好", "差", "一般"],
            },
            meta=meta,
        )
        segments, anomalies = _make_segments_and_anomalies(survey)
        text_default = generate_report(survey, segments, anomalies)
        text_explicit = generate_report(survey, segments, anomalies, fmt="text")
        assert text_default == text_explicit


class TestFormatCsvParseable:
    def test_csv_can_be_parsed_by_csv_reader(self):
        meta = [
            {"name": "age", "type": "numeric"},
            {"name": "city", "type": "categorical", "options": ["北京", "上海"]},
            {"name": "note", "type": "text"},
        ]
        survey = _make_survey(
            {
                "age": [20, 30, 25],
                "city": ["北京", "上海", "北京"],
                "note": ["a", "bb", "ccc"],
            },
            meta=meta,
        )
        segments, anomalies = _make_segments_and_anomalies(survey)
        csv_text = generate_report(survey, segments, anomalies, fmt="csv")

        reader = csv.reader(io.StringIO(csv_text))
        rows = list(reader)
        assert len(rows) > 0

        found_profile = any("[问卷概况]" in r for r in rows if r)
        found_segment = any("[分段统计]" in r for r in rows if r)
        found_anomaly = any("[异常答案标记]" in r for r in rows if r)
        assert found_profile
        assert found_segment
        assert found_anomaly


class TestCsvAnomalyColumnCount:
    def test_anomaly_block_has_three_columns(self):
        meta = [
            {"name": "q1", "type": "categorical"},
            {"name": "q2", "type": "categorical"},
            {"name": "q3", "type": "categorical"},
            {"name": "val", "type": "numeric"},
            {"name": "extra1", "type": "numeric"},
            {"name": "extra2", "type": "numeric"},
            {"name": "extra3", "type": "numeric"},
        ]
        survey = _make_survey(
            {
                "q1": ["A", "A", "B"],
                "q2": ["A", "A", "B"],
                "q3": ["A", "A", "C"],
                "val": [50, 52, 500],
                "extra1": [1, None, None],
                "extra2": [1, None, None],
                "extra3": [1, None, None],
            },
            meta=meta,
        )
        segments, anomalies = _make_segments_and_anomalies(survey)
        csv_text = generate_report(survey, segments, anomalies, fmt="csv")

        reader = csv.reader(io.StringIO(csv_text))
        rows = list(reader)

        anomaly_header_idx = None
        for i, row in enumerate(rows):
            if row and row[0] == "题目编号" and "检测类型" in row and "严重程度" in row:
                anomaly_header_idx = i
                break
        assert anomaly_header_idx is not None, "未找到异常答案表头"

        for row in rows[anomaly_header_idx + 1:]:
            if not row:
                continue
            if row[0].startswith("["):
                break
            assert len(row) == 3, f"异常行列数不为3: {row}"

    def test_anomaly_block_columns_when_no_anomalies(self):
        meta = [
            {"name": "a", "type": "numeric"},
        ]
        survey = _make_survey({"a": [1, 2, 3]}, meta=meta)
        segments, anomalies = _make_segments_and_anomalies(survey)
        csv_text = generate_report(survey, segments, anomalies, fmt="csv")

        reader = csv.reader(io.StringIO(csv_text))
        rows = list(reader)

        anomaly_header_idx = None
        for i, row in enumerate(rows):
            if row and row[0] == "题目编号" and "检测类型" in row and "严重程度" in row:
                anomaly_header_idx = i
                break
        assert anomaly_header_idx is not None

        placeholder_row = rows[anomaly_header_idx + 1]
        assert len(placeholder_row) == 3
        assert placeholder_row == ["-", "-", "-"]
