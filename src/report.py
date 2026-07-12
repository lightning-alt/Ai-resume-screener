"""PDF report generation using Reportlab.

Produces a polished, multi-section PDF summarising the resume analysis:
scores, skill gap, ATS checks, and suggestions. Returns raw bytes so
Streamlit can offer it as a download without writing to disk.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)

from .matching import MatchResult
from .skills import SkillAnalysis
from .ats import ATSResult


# ---------------------------------------------------------------------------
# Color palette (must match dashboard)
# ---------------------------------------------------------------------------

C_PRIMARY = colors.HexColor("#1E40AF")
C_ACCENT = colors.HexColor("#0EA5E9")
C_SUCCESS = colors.HexColor("#16A34A")
C_WARNING = colors.HexColor("#F59E0B")
C_ERROR = colors.HexColor("#DC2626")
C_NEUTRAL = colors.HexColor("#64748B")
C_LIGHT = colors.HexColor("#E2E8F0")
C_BG = colors.HexColor("#F8FAFC")


def _score_color(score: float) -> colors.Color:
    if score >= 75:
        return C_SUCCESS
    if score >= 50:
        return C_WARNING
    return C_ERROR


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def _build_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    custom = {
        "title": ParagraphStyle(
            "CustomTitle", parent=styles["Title"],
            fontSize=24, textColor=C_PRIMARY, spaceAfter=4, alignment=1,
        ),
        "subtitle": ParagraphStyle(
            "CustomSubtitle", parent=styles["Normal"],
            fontSize=10, textColor=C_NEUTRAL, spaceAfter=12, alignment=1,
        ),
        "h1": ParagraphStyle(
            "H1", parent=styles["Heading1"],
            fontSize=16, textColor=C_PRIMARY, spaceBefore=14, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2", parent=styles["Heading2"],
            fontSize=13, textColor=C_NEUTRAL, spaceBefore=10, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body", parent=styles["Normal"],
            fontSize=10, textColor=colors.black, leading=15, spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=styles["Normal"],
            fontSize=10, textColor=colors.black, leading=14,
            leftIndent=16, bulletIndent=6, spaceAfter=3,
        ),
        "small": ParagraphStyle(
            "Small", parent=styles["Normal"],
            fontSize=8, textColor=C_NEUTRAL, alignment=1,
        ),
    }
    return custom


def _score_bar_table(score: float, max_score: float, label: str, color: colors.Color) -> Table:
    """A visual score bar rendered as a coloured table cell."""
    pct = (score / max_score) * 100 if max_score else 0
    bar_width = pct * 1.6  # mm -> points approx
    inner = Table(
        [[""]],
        colWidths=[bar_width],
        rowHeights=[10],
    )
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("ROUNDEDCORNERS", [3, 3, 3, 3]),
    ]))
    outer = Table(
        [[label, Paragraph(f"<b>{score:.1f}/{max_score:.0f}</b>", ParagraphStyle("s", fontSize=10, textColor=color, alignment=2))],
         [inner, ""]],
        colWidths=[120, 50],
    )
    outer.setStyle(TableStyle([
        ("SPAN", (1, 1), (1, 1)),
        ("SPAN", (0, 1), (1, 1)),
        ("BACKGROUND", (0, 1), (-1, -1), C_LIGHT),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return outer


def _skills_table(title: str, skills: list[str], color: colors.Color) -> Any:
    """Render a list of skills as a wrapped-paragraph table."""
    if not skills:
        text = "None"
    else:
        text = "  •  ".join(skills)
    style = ParagraphStyle(
        "skill", fontSize=10, leading=16, textColor=colors.black,
    )
    header_style = ParagraphStyle(
        "skillh", fontSize=12, textColor=color, spaceAfter=4,
    )
    return [Paragraph(title, header_style), Paragraph(text, style), Spacer(1, 8)]


def generate_report(
    match_result: MatchResult,
    skill_analysis: SkillAnalysis,
    ats_result: ATSResult,
    resume_stats: dict,
    job_stats: dict,
) -> bytes:
    """Generate a complete PDF report and return it as bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title="Resume Analysis Report",
    )
    styles = _build_styles()
    story: list[Any] = []

    # -- Header -----------------------------------------------------------
    story.append(Paragraph("AI Resume Screener &amp; Job Matcher", styles["title"]))
    story.append(Paragraph(
        f"Analysis Report &nbsp;•&nbsp; {datetime.now().strftime('%B %d, %Y at %H:%M')}",
        styles["subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=C_PRIMARY))
    story.append(Spacer(1, 12))

    # -- Score summary ----------------------------------------------------
    story.append(Paragraph("Score Summary", styles["h1"]))

    match_color = _score_color(match_result.score_percent)
    ats_color = _score_color(ats_result.score)

    summary_data = [
        ["Metric", "Score", "Rating"],
        ["Resume Match Score", f"{match_result.score_percent:.1f}%", _rating_label(match_result.score_percent)],
        ["ATS Compatibility", f"{ats_result.score:.1f}/100", _rating_label(ats_result.score)],
        ["Skills Matched", f"{len(skill_analysis.matching_skills)}/{len(skill_analysis.job_skills)}",
         f"{(len(skill_analysis.matching_skills) / max(len(skill_analysis.job_skills), 1)) * 100:.0f}%"],
    ]
    summary_table = Table(summary_data, colWidths=[80 * mm, 40 * mm, 50 * mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_BG, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, C_LIGHT),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TEXTCOLOR", (1, 1), (1, 1), match_color),
        ("TEXTCOLOR", (1, 2), (1, 2), ats_color),
        ("FONTNAME", (1, 1), (1, -1), "Helvetica-Bold"),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 14))

    # -- Resume statistics ------------------------------------------------
    story.append(Paragraph("Document Statistics", styles["h1"]))
    stats_data = [
        ["Metric", "Resume", "Job Description"],
        ["Word Count", str(resume_stats.get("word_count", 0)), str(job_stats.get("word_count", 0))],
        ["Reading Time", resume_stats.get("reading_time", "0 min"), job_stats.get("reading_time", "0 min")],
        ["Detected Skills", str(resume_stats.get("skill_count", 0)), str(job_stats.get("skill_count", 0))],
    ]
    stats_table = Table(stats_data, colWidths=[55 * mm, 55 * mm, 55 * mm])
    stats_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_NEUTRAL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_BG, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, C_LIGHT),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 14))

    # -- Skill gap analysis -----------------------------------------------
    story.append(Paragraph("Skill Gap Analysis", styles["h1"]))
    story.extend(_skills_table("Matching Skills", skill_analysis.matching_skills, C_SUCCESS))
    story.extend(_skills_table("Missing Skills", skill_analysis.missing_skills, C_ERROR))
    story.extend(_skills_table("Additional Skills in Resume", skill_analysis.extra_skills, C_ACCENT))

    # -- ATS checks -------------------------------------------------------
    story.append(Paragraph("ATS Compatibility Breakdown", styles["h1"]))
    ats_data = [["Criterion", "Status", "Detail"]]
    for check in ats_result.checks:
        status = "PASS" if check.passed else "FAIL"
        ats_data.append([
            Paragraph(check.name, ParagraphStyle("c", fontSize=9, leading=12)),
            Paragraph(
                f"<font color='{_hex(_score_color(100 if check.passed else 0))}'><b>{status}</b></font>",
                ParagraphStyle("st", fontSize=9, leading=12, alignment=1),
            ),
            Paragraph(check.detail, ParagraphStyle("d", fontSize=9, leading=12)),
        ])
    ats_table = Table(ats_data, colWidths=[40 * mm, 20 * mm, 100 * mm])
    ats_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_BG, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, C_LIGHT),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(ats_table)
    story.append(Spacer(1, 14))

    # -- Suggestions ------------------------------------------------------
    story.append(Paragraph("Improvement Suggestions", styles["h1"]))
    for i, suggestion in enumerate(ats_result.suggestions, 1):
        story.append(Paragraph(f"{i}. {suggestion}", styles["bullet"]))
        story.append(Spacer(1, 3))

    # -- Top keywords -----------------------------------------------------
    story.append(Spacer(1, 8))
    story.append(Paragraph("Top Matching Keywords", styles["h1"]))
    if match_result.overlap_terms:
        kw_text = "  •  ".join(match_result.overlap_terms[:25])
        story.append(Paragraph(kw_text, ParagraphStyle("kw", fontSize=10, leading=16)))
    else:
        story.append(Paragraph("No overlapping keywords detected.", styles["body"]))

    # -- Footer -----------------------------------------------------------
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=C_LIGHT))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Generated by AI Resume Screener &amp; Job Matcher — "
        "scores are heuristic estimates for guidance only.",
        styles["small"],
    ))

    doc.build(story)
    return buffer.getvalue()


def _rating_label(score: float) -> str:
    if score >= 75:
        return "Excellent"
    if score >= 50:
        return "Needs Work"
    return "Low"


def _hex(color: colors.Color) -> str:
    return f"#{int(color.red * 255):02X}{int(color.green * 255):02X}{int(color.blue * 255):02X}"
