from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np

from survey_summary.models import SurveyData


@dataclass
class Anomaly:
    row_index: int
    reason: str
    detail: str


def detect_anomalies(
    survey: SurveyData,
    straight_line_threshold: float = 0.95,
    zscore_threshold: float = 3.0,
    missing_ratio_threshold: float = 0.6,
) -> list[Anomaly]:
    df = survey.df
    numeric_cols = [q.name for q in survey.questions if q.qtype == "numeric" and q.name in df.columns]
    anomalies: list[Anomaly] = []

    anomalies.extend(_detect_straight_lining(df, survey, straight_line_threshold))
    anomalies.extend(_detect_numeric_outliers(df, numeric_cols, zscore_threshold))
    anomalies.extend(_detect_high_missing(df, missing_ratio_threshold))

    return sorted(anomalies, key=lambda a: a.row_index)


def _detect_straight_lining(
    df: pd.DataFrame, survey: SurveyData, threshold: float
) -> list[Anomaly]:
    results: list[Anomaly] = []
    cat_cols = [
        q.name for q in survey.questions
        if q.qtype == "categorical" and q.name in df.columns
    ]
    if len(cat_cols) < 3:
        return results

    subset = df[cat_cols].astype(str)
    for idx, row in subset.iterrows():
        mode_val = row.mode().iloc[0] if len(row.mode()) > 0 else None
        if mode_val is None:
            continue
        same_ratio = (row == mode_val).sum() / len(row)
        if same_ratio >= threshold:
            results.append(
                Anomaly(
                    row_index=int(idx),
                    reason="straight_lining",
                    detail=f"连续相同答案比例 {same_ratio:.0%}，值='{mode_val}'",
                )
            )
    return results


def _detect_numeric_outliers(
    df: pd.DataFrame, numeric_cols: list[str], zscore_threshold: float
) -> list[Anomaly]:
    results: list[Anomaly] = []
    if not numeric_cols:
        return results

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 3:
            continue
        mean = series.mean()
        std = series.std()
        if std == 0:
            continue
        zscores = (series - mean).abs() / std
        for idx in zscores[zscores > zscore_threshold].index:
            val = df.loc[idx, col]
            results.append(
                Anomaly(
                    row_index=int(idx),
                    reason="numeric_outlier",
                    detail=f"题目'{col}'值={val}，Z-score={zscores[idx]:.2f}",
                )
            )
    return results


def _detect_high_missing(
    df: pd.DataFrame, threshold: float
) -> list[Anomaly]:
    results: list[Anomaly] = []
    total_cols = len(df.columns)
    if total_cols == 0:
        return results
    missing_ratio = df.isna().sum(axis=1) / total_cols
    for idx in missing_ratio[missing_ratio >= threshold].index:
        ratio = missing_ratio[idx]
        results.append(
            Anomaly(
                row_index=int(idx),
                reason="high_missing",
                detail=f"缺失比例 {ratio:.0%}",
            )
        )
    return results
