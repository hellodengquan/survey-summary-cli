from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Literal, TextIO

from survey_summary.models import SurveyData
from survey_summary.segment import SegmentResult, segment_stats
from survey_summary.anomaly import Anomaly


def generate_report(
    survey: SurveyData,
    segments: list[SegmentResult],
    anomalies: list[Anomaly],
    output: TextIO | None = None,
    fmt: Literal["text", "csv"] = "text",
) -> str:
    if fmt == "csv":
        text = _generate_csv(survey, segments, anomalies)
    else:
        text = _generate_text(survey, segments, anomalies)

    if output is not None:
        output.write(text)
    return text


def _generate_text(
    survey: SurveyData,
    segments: list[SegmentResult],
    anomalies: list[Anomaly],
) -> str:
    lines: list[str] = []
    _header(lines, survey)
    _segment_section(lines, segments)
    _anomaly_section(lines, anomalies)
    _footer(lines)
    return "\n".join(lines)


def _generate_csv(
    survey: SurveyData,
    segments: list[SegmentResult],
    anomalies: list[Anomaly],
) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)

    writer.writerow(["[问卷概况]"])
    writer.writerow(["总作答数", survey.row_count])
    writer.writerow(["题目数", len(survey.questions)])
    writer.writerow([])

    writer.writerow(["[分段统计]"])
    for seg in segments:
        writer.writerow([])
        if seg.qtype == "numeric":
            writer.writerow([
                "题目编号", "题目名称", "类型", "有效数", "缺失数",
                "均值", "标准差", "最小值", "Q1", "中位数", "Q3", "最大值",
            ])
            s = seg.stats
            writer.writerow([
                seg.question, seg.label, seg.qtype,
                seg.total - seg.missing, seg.missing,
                s["mean"], s["std"], s["min"], s["q1"], s["median"], s["q3"], s["max"],
            ])
        elif seg.qtype == "categorical":
            writer.writerow([
                "题目编号", "题目名称", "类型", "有效数", "缺失数",
                "选项", "频次", "百分比",
            ])
            s = seg.stats
            for opt, info in s["frequencies"].items():
                writer.writerow([
                    seg.question, seg.label, seg.qtype,
                    seg.total - seg.missing, seg.missing,
                    opt, info["count"], info["pct"],
                ])
        else:
            writer.writerow([
                "题目编号", "题目名称", "类型", "有效数", "缺失数",
                "唯一数", "平均长度", "最大长度",
            ])
            s = seg.stats
            writer.writerow([
                seg.question, seg.label, seg.qtype,
                seg.total - seg.missing, seg.missing,
                s["unique"], s["avg_len"], s["max_len"],
            ])

    writer.writerow([])
    writer.writerow(["[异常答案标记]"])
    writer.writerow(["题目编号", "检测类型", "严重程度"])
    if not anomalies:
        writer.writerow(["-", "-", "-"])
    else:
        reason_labels = {
            "straight_lining": "直线作答",
            "numeric_outlier": "数值离群",
            "high_missing": "高缺失率",
        }
        for a in anomalies:
            severity = _severity(a)
            writer.writerow([
                a.row_index,
                reason_labels.get(a.reason, a.reason),
                severity,
            ])

    return buf.getvalue()


def _severity(anomaly: Anomaly) -> str:
    if anomaly.reason == "numeric_outlier":
        return "高"
    if anomaly.reason == "straight_lining":
        return "中"
    return "低"


def _header(lines: list[str], survey: SurveyData) -> None:
    lines.append("=" * 60)
    lines.append("  问卷结果归纳报告")
    lines.append("=" * 60)
    lines.append(f"  生成时间 : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  总作答数 : {survey.row_count}")
    lines.append(f"  题目数   : {len(survey.questions)}")
    lines.append("")


def _segment_section(lines: list[str], segments: list[SegmentResult]) -> None:
    lines.append("-" * 60)
    lines.append("  一、分段统计")
    lines.append("-" * 60)
    for seg in segments:
        lines.append("")
        lines.append(f"  【{seg.label}】({seg.qtype})")
        lines.append(f"    有效回答 : {seg.total - seg.missing} / {seg.total}（缺失 {seg.missing}）")
        if seg.qtype == "numeric":
            s = seg.stats
            lines.append(f"    均值={s['mean']}  标准差={s['std']}")
            lines.append(f"    最小={s['min']}  Q1={s['q1']}  中位数={s['median']}  Q3={s['q3']}  最大={s['max']}")
        elif seg.qtype == "categorical":
            s = seg.stats
            lines.append(f"    最频繁 : {s['top']}（{s['top_count']}次）")
            for opt, info in s["frequencies"].items():
                bar = "█" * int(info["pct"] / 2.5)
                lines.append(f"    {opt:>20s} | {info['count']:>4d} ({info['pct']:5.1f}%) {bar}")
        else:
            s = seg.stats
            lines.append(f"    有效数={s['count']}  唯一数={s['unique']}  平均长度={s['avg_len']}  最大长度={s['max_len']}")
    lines.append("")


def _anomaly_section(lines: list[str], anomalies: list[Anomaly]) -> None:
    lines.append("-" * 60)
    lines.append("  二、异常答案标记")
    lines.append("-" * 60)
    if not anomalies:
        lines.append("  未检测到异常答案。")
    else:
        reason_labels = {
            "straight_lining": "直线作答",
            "numeric_outlier": "数值离群",
            "high_missing": "高缺失率",
        }
        grouped: dict[str, list[Anomaly]] = {}
        for a in anomalies:
            grouped.setdefault(a.reason, []).append(a)

        for reason, items in grouped.items():
            label = reason_labels.get(reason, reason)
            lines.append(f"")
            lines.append(f"  ▸ {label}（{len(items)} 条）")
            for a in items:
                lines.append(f"    行 {a.row_index:>4d} : {a.detail}")
    lines.append("")


def _footer(lines: list[str]) -> None:
    lines.append("=" * 60)
    lines.append("  报告结束")
    lines.append("=" * 60)
