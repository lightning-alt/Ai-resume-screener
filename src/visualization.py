"""Interactive Plotly visualizations for the resume analysis dashboard.

Every function returns a ``plotly.graph_objects.Figure`` so Streamlit can
render them with ``st.plotly_chart``.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .matching import MatchResult
from .skills import SkillAnalysis


# ---------------------------------------------------------------------------
# Color palette (ATS-inspired, professional blues / teals / greens)
# ---------------------------------------------------------------------------

COLOR_PRIMARY = "#1E40AF"      # deep blue
COLOR_ACCENT = "#0EA5E9"       # sky blue
COLOR_SUCCESS = "#16A34A"     # green
COLOR_WARNING = "#F59E0B"     # amber
COLOR_ERROR = "#DC2626"       # red
COLOR_NEUTRAL = "#64748B"     # slate
COLOR_LIGHT = "#E2E8F0"


def _score_color(score: float) -> str:
    """Pick a semantic color based on a 0-100 score."""
    if score >= 75:
        return COLOR_SUCCESS
    if score >= 50:
        return COLOR_WARNING
    return COLOR_ERROR


# ---------------------------------------------------------------------------
# Gauge chart
# ---------------------------------------------------------------------------

def match_gauge(score_percent: float, title: str = "Resume Match Score") -> go.Figure:
    """A semicircular gauge showing the overall match percentage."""
    color = _score_color(score_percent)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_percent,
        number={"suffix": "%", "font": {"size": 48, "color": color}},
        title={"text": title, "font": {"size": 18, "color": COLOR_NEUTRAL}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLOR_NEUTRAL},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": COLOR_LIGHT,
            "steps": [
                {"range": [0, 50], "color": "#FEE2E2"},
                {"range": [50, 75], "color": "#FEF3C7"},
                {"range": [75, 100], "color": "#DCFCE7"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.8,
                "value": score_percent,
            },
        },
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# ATS gauge
# ---------------------------------------------------------------------------

def ats_gauge(score: float) -> go.Figure:
    """Gauge for the ATS compatibility score."""
    color = _score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 40, "color": color}},
        title={"text": "ATS Compatibility", "font": {"size": 18, "color": COLOR_NEUTRAL}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COLOR_NEUTRAL},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": COLOR_LIGHT,
            "steps": [
                {"range": [0, 50], "color": "#FEE2E2"},
                {"range": [50, 75], "color": "#FEF3C7"},
                {"range": [75, 100], "color": "#DCFCE7"},
            ],
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20),
                      paper_bgcolor="white", plot_bgcolor="white")
    return fig


# ---------------------------------------------------------------------------
# Skill comparison bar chart
# ---------------------------------------------------------------------------

def skill_comparison_chart(analysis: SkillAnalysis) -> go.Figure:
    """Horizontal bar chart comparing resume vs job skill counts by category."""
    categories = ["Matching", "Missing", "Extra (in resume)"]
    counts = [
        len(analysis.matching_skills),
        len(analysis.missing_skills),
        len(analysis.extra_skills),
    ]
    colors = [COLOR_SUCCESS, COLOR_ERROR, COLOR_ACCENT]

    fig = go.Figure(go.Bar(
        x=counts,
        y=categories,
        orientation="h",
        marker_color=colors,
        text=counts,
        textposition="auto",
        textfont={"size": 16, "color": "white"},
        width=0.5,
    ))

    fig.update_layout(
        title="Skill Gap Summary",
        title_font_size=18,
        xaxis_title="Number of Skills",
        yaxis=dict(autorange="reversed"),
        height=280,
        margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=COLOR_LIGHT)
    return fig


# ---------------------------------------------------------------------------
# Top TF-IDF terms bar chart
# ---------------------------------------------------------------------------

def top_terms_chart(match_result: MatchResult, top_n: int = 15) -> go.Figure:
    """Grouped horizontal bar chart of top TF-IDF terms in resume vs job."""
    # Collect union of top terms.
    all_terms = list(dict.fromkeys(
        list(match_result.resume_terms.keys()) + list(match_result.job_terms.keys())
    ))
    # Sort by combined importance.
    combined = {t: match_result.resume_terms.get(t, 0) + match_result.job_terms.get(t, 0)
                for t in all_terms}
    top = sorted(combined, key=combined.get, reverse=True)[:top_n]
    top.reverse()  # so highest is at top in horizontal bar

    resume_scores = [match_result.resume_terms.get(t, 0) for t in top]
    job_scores = [match_result.job_terms.get(t, 0) for t in top]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Resume",
        y=top,
        x=resume_scores,
        orientation="h",
        marker_color=COLOR_PRIMARY,
    ))
    fig.add_trace(go.Bar(
        name="Job Description",
        y=top,
        x=job_scores,
        orientation="h",
        marker_color=COLOR_ACCENT,
    ))

    fig.update_layout(
        title="Top Keywords by TF-IDF Weight",
        title_font_size=18,
        barmode="group",
        height=max(350, len(top) * 28),
        margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="TF-IDF Weight",
    )
    fig.update_xaxes(gridcolor=COLOR_LIGHT)
    return fig


# ---------------------------------------------------------------------------
# ATS checks breakdown
# ---------------------------------------------------------------------------

def ats_checks_chart(checks_data: list[dict]) -> go.Figure:
    """Horizontal bar chart showing pass/fail per ATS criterion."""
    names = [c["name"] for c in checks_data]
    scores = [c["weight"] if c["passed"] else 0 for c in checks_data]
    colors = [COLOR_SUCCESS if c["passed"] else COLOR_ERROR for c in checks_data]

    fig = go.Figure(go.Bar(
        x=scores,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{s}/{c['weight']}" for s, c in zip(scores, checks_data)],
        textposition="auto",
        textfont={"size": 13},
        width=0.5,
    ))

    fig.update_layout(
        title="ATS Criteria Breakdown",
        title_font_size=18,
        xaxis_title="Points Earned",
        yaxis=dict(autorange="reversed"),
        height=max(300, len(names) * 38),
        margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=COLOR_LIGHT)
    return fig


# ---------------------------------------------------------------------------
# Skill distribution donut
# ---------------------------------------------------------------------------

def skill_distribution_donut(analysis: SkillAnalysis) -> go.Figure:
    """Donut chart showing matching vs missing vs extra skills."""
    labels = ["Matching", "Missing", "Extra"]
    values = [
        len(analysis.matching_skills),
        len(analysis.missing_skills),
        len(analysis.extra_skills),
    ]
    colors = [COLOR_SUCCESS, COLOR_ERROR, COLOR_ACCENT]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+percent",
        textfont_size=14,
    ))

    fig.update_layout(
        title="Skill Distribution",
        title_font_size=18,
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        annotations=[dict(
            text=f"{values[0]}<br>Matching",
            x=0.5, y=0.5,
            font_size=16,
            font_color=COLOR_SUCCESS,
            showarrow=False,
        )],
    )
    return fig
