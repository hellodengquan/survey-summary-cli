from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from survey_summary.models import load_csv, build_survey
from survey_summary.segment import segment_stats, cross_tab
from survey_summary.anomaly import detect_anomalies
from survey_summary.report import generate_report

app = typer.Typer(
    name="survey",
    help="问卷结果快速归纳工具：分段统计、异常答案标记和文本报告",
    add_completion=False,
)
console = Console()


@app.command()
def summary(
    file: Path = typer.Argument(..., exists=True, help="问卷 CSV 数据文件路径"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="报告输出文件路径（默认输出到终端）"),
    format: str = typer.Option("text", "-f", "--format", help="输出格式：text 或 csv"),
    encoding: str = typer.Option("utf-8", "-e", "--encoding", help="CSV 文件编码"),
    straight_line: float = typer.Option(0.95, "--straight-line", help="直线作答检测阈值（0~1）"),
    zscore: float = typer.Option(3.0, "--zscore", help="数值离群 Z-score 阈值"),
    missing_ratio: float = typer.Option(0.6, "--missing-ratio", help="高缺失率检测阈值（0~1）"),
) -> None:
    """生成问卷归纳报告：分段统计 + 异常标记"""
    if format not in ("text", "csv"):
        console.print(f"[red]不支持的格式: {format}，请使用 text 或 csv[/red]")
        raise typer.Exit(code=1)

    df = load_csv(file, encoding=encoding)
    survey = build_survey(df)

    segments = segment_stats(survey)
    anomalies = detect_anomalies(
        survey,
        straight_line_threshold=straight_line,
        zscore_threshold=zscore,
        missing_ratio_threshold=missing_ratio,
    )

    if output is not None:
        with open(output, "w", encoding="utf-8", newline="") as f:
            generate_report(survey, segments, anomalies, output=f, fmt=format)
        console.print(f"[green]✓ 报告已保存至 {output}[/green]")
    else:
        text = generate_report(survey, segments, anomalies, fmt=format)
        console.print(text)


@app.command()
def stats(
    file: Path = typer.Argument(..., exists=True, help="问卷 CSV 数据文件路径"),
    encoding: str = typer.Option("utf-8", "-e", "--encoding", help="CSV 文件编码"),
    question: Optional[str] = typer.Option(None, "-q", "--question", help="仅查看指定题目"),
) -> None:
    """查看分段统计（不生成完整报告）"""
    df = load_csv(file, encoding=encoding)
    survey = build_survey(df)
    segments = segment_stats(survey)

    if question:
        segments = [s for s in segments if s.question == question or s.label == question]

    for seg in segments:
        console.rule(seg.label)
        console.print(f"  类型 : {seg.qtype}")
        console.print(f"  有效 : {seg.total - seg.missing} / {seg.total}")
        console.print(f"  统计 : {seg.stats}")


@app.command()
def anomaly(
    file: Path = typer.Argument(..., exists=True, help="问卷 CSV 数据文件路径"),
    encoding: str = typer.Option("utf-8", "-e", "--encoding", help="CSV 文件编码"),
    straight_line: float = typer.Option(0.95, "--straight-line", help="直线作答检测阈值"),
    zscore: float = typer.Option(3.0, "--zscore", help="数值离群 Z-score 阈值"),
    missing_ratio: float = typer.Option(0.6, "--missing-ratio", help="高缺失率检测阈值"),
) -> None:
    """仅检测异常答案"""
    df = load_csv(file, encoding=encoding)
    survey = build_survey(df)
    anomalies = detect_anomalies(
        survey,
        straight_line_threshold=straight_line,
        zscore_threshold=zscore,
        missing_ratio_threshold=missing_ratio,
    )

    if not anomalies:
        console.print("[green]未检测到异常答案[/green]")
        return

    reason_labels = {
        "straight_lining": "直线作答",
        "numeric_outlier": "数值离群",
        "high_missing": "高缺失率",
    }
    grouped: dict[str, list] = {}
    for a in anomalies:
        grouped.setdefault(a.reason, []).append(a)

    for reason, items in grouped.items():
        label = reason_labels.get(reason, reason)
        console.rule(label)
        for a in items:
            console.print(f"  行 {a.row_index:>4d} : {a.detail}")


@app.command()
def crosstab(
    file: Path = typer.Argument(..., exists=True, help="问卷 CSV 数据文件路径"),
    row: str = typer.Option(..., "-r", "--row", help="行变量题目名"),
    col: str = typer.Option(..., "-c", "--col", help="列变量题目名"),
    encoding: str = typer.Option("utf-8", "-e", "--encoding", help="CSV 文件编码"),
) -> None:
    """生成交叉表"""
    df = load_csv(file, encoding=encoding)
    survey = build_survey(df)
    ct = cross_tab(survey, row, col)
    console.print(ct.to_string())


if __name__ == "__main__":
    app()
