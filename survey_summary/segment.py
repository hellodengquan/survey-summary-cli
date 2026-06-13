from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from survey_summary.models import SurveyData, QuestionMeta


@dataclass
class SegmentResult:
    question: str
    label: str
    qtype: str
    total: int
    missing: int
    stats: dict


def segment_stats(survey: SurveyData) -> list[SegmentResult]:
    results: list[SegmentResult] = []
    for q in survey.questions:
        if q.name not in survey.df.columns:
            continue
        series = survey.df[q.name]
        total = len(series)
        missing = int(series.isna().sum())
        valid = series.dropna()

        if q.qtype == "numeric":
            stats = _numeric_stats(valid)
        elif q.qtype == "categorical":
            stats = _categorical_stats(valid, q.options)
        else:
            stats = _text_stats(valid)

        results.append(
            SegmentResult(
                question=q.name,
                label=q.label,
                qtype=q.qtype,
                total=total,
                missing=missing,
                stats=stats,
            )
        )
    return results


def _numeric_stats(series: pd.Series) -> dict:
    desc = series.describe()
    return {
        "mean": round(float(desc["mean"]), 2),
        "std": round(float(desc["std"]), 2),
        "min": float(desc["min"]),
        "q1": float(desc["25%"]),
        "median": float(desc["50%"]),
        "q3": float(desc["75%"]),
        "max": float(desc["max"]),
    }


def _categorical_stats(series: pd.Series, options: list[str] | None = None) -> dict:
    vc = series.value_counts()
    total_valid = int(vc.sum())
    freq = {}
    items = options if options else [str(k) for k in vc.index]
    for opt in items:
        count = int(vc.get(opt, 0))
        freq[opt] = {"count": count, "pct": round(count / total_valid * 100, 1) if total_valid else 0}
    return {"frequencies": freq, "top": str(vc.index[0]) if len(vc) else "", "top_count": int(vc.iloc[0]) if len(vc) else 0}


def _text_stats(series: pd.Series) -> dict:
    lengths = series.astype(str).str.len()
    return {
        "count": len(series),
        "unique": int(series.nunique()),
        "avg_len": round(float(lengths.mean()), 1),
        "max_len": int(lengths.max()),
    }


def cross_tab(survey: SurveyData, row_q: str, col_q: str) -> pd.DataFrame:
    if row_q not in survey.df.columns or col_q not in survey.df.columns:
        raise ValueError(f"列不存在: {row_q} 或 {col_q}")
    return pd.crosstab(survey.df[row_q], survey.df[col_q], margins=True)
