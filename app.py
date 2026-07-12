"""AI Resume Screener & Job Matcher — Streamlit application entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure the local src package is importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent))

from src.parsing import extract_text, clean_extracted_text
from src.preprocessing import ensure_nltk_resources
from src.matching import compute_match
from src.skills import extract_skills, skill_match_ratio
from src.ats import evaluate_ats
from src.visualization import (
    match_gauge, ats_gauge, skill_comparison_chart,
    top_terms_chart, ats_checks_chart, skill_distribution_donut,
)
from src.report import generate_report
from src.utils import compute_stats, stats_to_dict, rating_label, rating_color, truncate


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Resume Screener & Job Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS — modern ATS / recruiter dashboard styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ---------- Global ---------- */
.stApp {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #F8FAFC;
}

.main .block-container {
    padding-top: 1.5rem;
    max-width: 1200px;
}

/* ---------- Header ---------- */
.app-header {
    background: linear-gradient(135deg, #1E40AF 0%, #0EA5E9 100%);
    padding: 1.75rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(30, 64, 175, 0.15);
}
.app-header h1 {
    color: white;
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}
.app-header p {
    color: rgba(255, 255, 255, 0.85);
    font-size: 0.95rem;
    margin: 0.35rem 0 0 0;
}

/* ---------- Metric cards ---------- */
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: transform 0.2s, box-shadow 0.2s;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
div[data-testid="stMetric"] label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 1.85rem;
    font-weight: 700;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid #E2E8F0;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #1E40AF;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: white;
    padding: 6px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 0.9rem;
    color: #64748B;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: #1E40AF !important;
    color: white !important;
}

/* ---------- Skill badges ---------- */
.skill-badge {
    display: inline-block;
    padding: 4px 12px;
    margin: 3px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 500;
}
.skill-match {
    background: #DCFCE7;
    color: #166534;
    border: 1px solid #BBF7D0;
}
.skill-missing {
    background: #FEE2E2;
    color: #991B1B;
    border: 1px solid #FECACA;
}
.skill-extra {
    background: #E0F2FE;
    color: #075985;
    border: 1px solid #BAE6FD;
}

/* ---------- Suggestion cards ---------- */
.suggestion-card {
    background: white;
    border-left: 4px solid #F59E0B;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.suggestion-card span {
    color: #1E293B;
    font-size: 0.92rem;
    line-height: 1.5;
}

/* ---------- Check items ---------- */
.check-pass {
    color: #16A34A;
    font-weight: 600;
}
.check-fail {
    color: #DC2626;
    font-weight: 600;
}

/* ---------- Divider ---------- */
hr {
    border-color: #E2E8F0;
    margin: 1rem 0;
}

/* ---------- Spinner ---------- */
.stSpinner > div {
    border-color: #1E40AF !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached analysis functions
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_extract(uploaded_file_id: str, file_content: bytes, filename: str) -> str:
    """Extract and clean text from an uploaded file (cached by content hash)."""
    raw = extract_text(file_content, filename)
    return clean_extracted_text(raw)


@st.cache_data(show_spinner=False)
def cached_analysis(resume_text: str, job_text: str) -> dict:
    """Run the full analysis pipeline and cache the result."""
    match_result = compute_match(resume_text, job_text)
    skill_analysis = extract_skills(resume_text, job_text)
    ats_result = evaluate_ats(resume_text, job_text, skill_analysis)

    resume_stats = compute_stats(resume_text, skill_analysis.resume_skills)
    job_stats = compute_stats(job_text, skill_analysis.job_skills)

    return {
        "match": match_result,
        "skills": skill_analysis,
        "ats": ats_result,
        "resume_stats": resume_stats,
        "job_stats": job_stats,
    }


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.markdown("""
    <div class="app-header">
        <h1>AI Resume Screener &amp; Job Matcher</h1>
        <p>Upload a resume and job description to instantly compute match scores,
           skill gaps, ATS compatibility, and actionable suggestions.</p>
    </div>
    """, unsafe_allow_html=True)


def render_skill_badges(skills: list[str], css_class: str) -> str:
    """Return HTML string of skill badges."""
    if not skills:
        return '<span style="color:#94A3B8;font-size:0.9rem;">None detected.</span>'
    badges = "".join(
        f'<span class="skill-badge {css_class}">{skill}</span>' for skill in skills
    )
    return badges


def metric_card(label: str, value: str, delta: str | None = None) -> None:
    st.metric(label=label, value=value, delta=delta)


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str | None, str]:
    """Render the sidebar inputs and return (resume_text, job_text)."""
    with st.sidebar:
        st.markdown("### Upload Inputs")

        st.markdown("#### Resume")
        st.caption("Upload a PDF or DOCX file, or paste text below.")
        resume_file = st.file_uploader(
            "Resume file",
            type=["pdf", "docx"],
            label_visibility="collapsed",
            key="resume_file",
        )
        resume_pasted = st.text_area(
            "Or paste resume text",
            height=120,
            placeholder="Paste your resume text here…",
            key="resume_pasted",
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("#### Job Description")
        st.caption("Upload a PDF/DOCX or paste the job description.")
        job_file = st.file_uploader(
            "Job description file",
            type=["pdf", "docx"],
            label_visibility="collapsed",
            key="job_file",
        )
        job_pasted = st.text_area(
            "Or paste job description text",
            height=120,
            placeholder="Paste the job description here…",
            key="job_pasted",
            label_visibility="collapsed",
        )

        st.markdown("---")
        if st.button("Load Sample Data", use_container_width=True, type="secondary"):
            _load_sample_data()

        st.markdown("---")
        st.caption("Built with spaCy, NLTK, scikit-learn & Plotly.")

    # Resolve resume text.
    resume_text = None
    if resume_file is not None:
        content = resume_file.read()
        try:
            resume_text = cached_extract(f"{resume_file.name}_{resume_file.size}", content, resume_file.name)
        except Exception as e:
            st.sidebar.error(f"Failed to parse resume: {e}")
    elif resume_pasted.strip():
        resume_text = clean_extracted_text(resume_pasted)

    # Resolve job text.
    job_text = ""
    if job_file is not None:
        content = job_file.read()
        try:
            job_text = cached_extract(f"{job_file.name}_{job_file.size}", content, job_file.name)
        except Exception as e:
            st.sidebar.error(f"Failed to parse job description: {e}")
    elif job_pasted.strip():
        job_text = clean_extracted_text(job_pasted)

    return resume_text, job_text


def _load_sample_data() -> None:
    """Load sample resume and JD from the samples directory into session state."""
    base = Path(__file__).parent / "samples"
    resume_path = base / "sample_resume.txt"
    jd_path = base / "sample_job_description.txt"
    if resume_path.exists() and jd_path.exists():
        st.session_state["resume_pasted"] = resume_path.read_text()
        st.session_state["job_pasted"] = jd_path.read_text()
        st.rerun()
    else:
        st.sidebar.warning("Sample files not found.")


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

def tab_overview(result: dict) -> None:
    """Overview tab: scores, metric cards, quick summary."""
    match = result["match"]
    ats = result["ats"]
    skills = result["skills"]
    resume_stats = result["resume_stats"]
    job_stats = result["job_stats"]

    # Metric cards row.
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        mc = rating_color(match.score_percent)
        st.metric("Match Score", f"{match.score_percent:.1f}%",
                  delta=rating_label(match.score_percent),
                  delta_color="normal" if match.score_percent >= 50 else "inverse")
    with col2:
        st.metric("ATS Score", f"{ats.score:.1f}/100",
                  delta=rating_label(ats.score),
                  delta_color="normal" if ats.score >= 50 else "inverse")
    with col3:
        ratio = skill_match_ratio(skills) * 100
        st.metric("Skills Matched", f"{len(skills.matching_skills)}/{len(skills.job_skills)}",
                  delta=f"{ratio:.0f}% coverage",
                  delta_color="normal" if ratio >= 50 else "inverse")
    with col4:
        st.metric("Resume Words", f"{resume_stats.word_count:,}",
                  delta=resume_stats.reading_time)

    st.markdown("---")

    # Gauges side by side.
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        st.plotly_chart(match_gauge(match.score_percent), use_container_width=True)
    with gcol2:
        st.plotly_chart(ats_gauge(ats.score), use_container_width=True)

    # Quick summary.
    st.markdown("### Summary")
    ratio_pct = skill_match_ratio(skills) * 100
    if match.score_percent >= 75 and ats.score >= 75:
        st.success(
            f"Strong match! Your resume aligns well with this job description "
            f"({match.score_percent:.1f}% match, {ats.score:.0f} ATS score). "
            f"{ratio_pct:.0f}% of required skills are present."
        )
    elif match.score_percent >= 50:
        st.warning(
            f"Moderate match ({match.score_percent:.1f}%). Your resume covers "
            f"{ratio_pct:.0f}% of required skills. Review the Suggestions tab "
            f"to close the gap."
        )
    else:
        st.error(
            f"Low match ({match.score_percent:.1f}%). Significant gaps detected. "
            f"Only {ratio_pct:.0f}% of required skills are present. "
            f"See the Suggestions tab for targeted improvements."
        )

    # Document preview.
    with st.expander("View extracted resume text (preview)"):
        st.text(truncate(result.get("_resume_text", ""), 2000))
    with st.expander("View extracted job description (preview)"):
        st.text(truncate(result.get("_job_text", ""), 2000))


def tab_skills(result: dict) -> None:
    """Skills tab: matching, missing, extra skills with badges."""
    skills = result["skills"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Matching Skills")
        st.markdown(
            f'<div>{render_skill_badges(skills.matching_skills, "skill-match")}</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("#### Missing Skills")
        st.markdown(
            f'<div>{render_skill_badges(skills.missing_skills, "skill-missing")}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Additional Skills in Resume (not in JD)")
    st.markdown(
        f'<div>{render_skill_badges(skills.extra_skills, "skill-extra")}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Resume Skills", len(skills.resume_skills))
    c2.metric("Total JD Skills", len(skills.job_skills))
    c3.metric("Skill Gap", len(skills.missing_skills))


def tab_charts(result: dict) -> None:
    """Charts tab: all Plotly visualizations."""
    match = result["match"]
    skills = result["skills"]
    ats = result["ats"]

    st.plotly_chart(skill_comparison_chart(skills), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(top_terms_chart(match), use_container_width=True)
    with col2:
        st.plotly_chart(skill_distribution_donut(skills), use_container_width=True)

    st.markdown("---")
    checks_data = [
        {"name": c.name, "weight": c.weight, "passed": c.passed}
        for c in ats.checks
    ]
    st.plotly_chart(ats_checks_chart(checks_data), use_container_width=True)


def tab_suggestions(result: dict) -> None:
    """Suggestions tab: ATS checks + improvement suggestions."""
    ats = result["ats"]

    st.markdown("### ATS Compatibility Checks")
    for check in ats.checks:
        icon = "✅" if check.passed else "❌"
        css = "check-pass" if check.passed else "check-fail"
        st.markdown(
            f"**{check.name}** — "
            f"<span class='{css}'>{icon} {'PASS' if check.passed else 'FAIL'}</span> "
            f"<span style='color:#64748B;font-size:0.88rem;'>({check.weight} pts)</span>"
            f"<br><span style='color:#475569;font-size:0.9rem;'>{check.detail}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")

    st.markdown("---")
    st.markdown("### Improvement Suggestions")
    for suggestion in ats.suggestions:
        st.markdown(
            f'<div class="suggestion-card"><span>{suggestion}</span></div>',
            unsafe_allow_html=True,
        )


def tab_report(result: dict, resume_text: str, job_text: str) -> None:
    """Report tab: downloadable PDF + resume statistics."""
    resume_stats = result["resume_stats"]
    job_stats = result["job_stats"]

    st.markdown("### Resume Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Word Count", f"{resume_stats.word_count:,}")
    col2.metric("Sentences", resume_stats.sentence_count)
    col3.metric("Reading Time", resume_stats.reading_time)
    col4.metric("Detected Skills", resume_stats.skill_count)

    st.markdown("### Job Description Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Word Count", f"{job_stats.word_count:,}")
    col2.metric("Sentences", job_stats.sentence_count)
    col3.metric("Reading Time", job_stats.reading_time)
    col4.metric("Detected Skills", job_stats.skill_count)

    st.markdown("---")
    st.markdown("### Download PDF Report")
    st.write("Generate a comprehensive PDF report summarising the full analysis.")

    if st.button("Generate PDF Report", type="primary", use_container_width=True):
        with st.spinner("Generating report…"):
            try:
                pdf_bytes = generate_report(
                    match_result=result["match"],
                    skill_analysis=result["skills"],
                    ats_result=result["ats"],
                    resume_stats=stats_to_dict(resume_stats),
                    job_stats=stats_to_dict(job_stats),
                )
                st.success("Report generated successfully!")
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name="resume_analysis_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Failed to generate report: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_nltk_resources()
    render_header()
    resume_text, job_text = render_sidebar()

    # Input validation.
    if not resume_text:
        st.info("Upload a resume file or paste resume text in the sidebar to begin.")
        return
    if not job_text:
        st.info("Upload a job description or paste its text in the sidebar to begin.")
        return
    if len(resume_text.strip()) < 20:
        st.warning("Resume text is too short for meaningful analysis. Please provide a more complete resume.")
        return
    if len(job_text.strip()) < 20:
        st.warning("Job description is too short for meaningful analysis. Please provide a more complete job description.")
        return

    # Run analysis with a loading spinner.
    with st.spinner("Analyzing resume and job description…"):
        try:
            result = cached_analysis(resume_text, job_text)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.exception(e)
            return

    # Stash raw texts for preview.
    result["_resume_text"] = resume_text
    result["_job_text"] = job_text

    # Tabs.
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Overview", "🎯 Skills", "📈 Charts", "💡 Suggestions", "📄 Report"]
    )
    with tab1:
        tab_overview(result)
    with tab2:
        tab_skills(result)
    with tab3:
        tab_charts(result)
    with tab4:
        tab_suggestions(result)
    with tab5:
        tab_report(result, resume_text, job_text)


if __name__ == "__main__":
    main()
