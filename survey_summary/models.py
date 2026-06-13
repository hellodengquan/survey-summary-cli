from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class QuestionMeta:
    name: str
    qtype: str
    label: str = ""
    options: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, raw: dict) -> QuestionMeta:
        return cls(
            name=raw["name"],
            qtype=raw.get("type", "unknown"),
            label=raw.get("label", raw["name"]),
            options=raw.get("options", []),
        )


@dataclass
class SurveyData:
    df: pd.DataFrame
    questions: list[QuestionMeta]

    @property
    def row_count(self) -> int:
        return len(self.df)

    @property
    def col_names(self) -> list[str]:
        return list(self.df.columns)


def load_csv(path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    return pd.read_csv(path, encoding=encoding)


def build_survey(df: pd.DataFrame, meta: list[dict] | None = None) -> SurveyData:
    if meta is None:
        questions = [
            QuestionMeta(name=col, qtype=_infer_type(df[col]), label=col)
            for col in df.columns
        ]
    else:
        questions = [QuestionMeta.parse(m) for m in meta]
    return SurveyData(df=df, questions=questions)


def _infer_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    nunique = series.nunique(dropna=True)
    if nunique <= 10:
        return "categorical"
    return "text"
