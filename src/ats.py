"""ATS compatibility scoring and actionable improvement suggestions.

The ATS score is a composite of several heuristics that approximate what
Applicant Tracking Systems look for: keyword coverage, formatting
simplicity, section presence, contact info, and length appropriateness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .skills import SkillAnalysis, skill_match_ratio


@dataclass
class ATSCheck:
    """A single ATS evaluation criterion."""

    name: str
    passed: bool
    weight: float
    detail: str


@dataclass
class ATSResult:
    """Full ATS evaluation output."""

    score: float  # 0..100
    checks: list[ATSCheck] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"\b(experience|work history|employment|education|skills|summary|"
    r"objective|certifications|projects|publications|awards|contact|"
    r"profile|achievements|languages|interests)\b",
    re.IGNORECASE,
)

_ACTION_VERBS = {
    "developed", "designed", "implemented", "managed", "led", "created",
    "built", "launched", "improved", "optimized", "architected", "delivered",
    "achieved", "increased", "reduced", "streamlined", "automated", "coordinated",
    "established", "directed", "spearheaded", "engineered", "analyzed", "researched",
    "collaborated", "mentored", "negotiated", "executed", "deployed", "integrated",
    "migrated", "scaled", "transformed", "accelerated", "orchestrated",
}

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/[A-Za-z0-9_-]+", re.IGNORECASE)
_GITHUB_RE = re.compile(r"github\.com/[A-Za-z0-9_-]+", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def _has_contact_info(text: str) -> bool:
    return bool(_EMAIL_RE.search(text) or _PHONE_RE.search(text))


def _count_sections(text: str) -> int:
    return len(set(m.group(1).lower() for m in _SECTION_RE.finditer(text)))


def _has_quantified_achievements(text: str) -> bool:
    # Numbers / percentages near action verbs indicate quantified impact.
    return bool(re.search(r"\b\d+%|\$\d|\b\d{2,}\b", text))


def _action_verb_count(text: str) -> int:
    tokens = set(re.findall(r"\b[a-z]+\b", text.lower()))
    return len(tokens & _ACTION_VERBS)


def _has_problematic_elements(text: str) -> list[str]:
    """Return a list of detected elements that confuse ATS parsers."""
    problems: list[str] = []
    if re.search(r"\b(table|column|text box|header|footer|image|graphic)\b", text, re.IGNORECASE):
        problems.append("Detected references to tables/images/text boxes — ATS may not parse these.")
    if text.count("\f") > 3:
        problems.append("Multiple page breaks detected — keep the resume concise.")
    return problems


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _length_ok(word_count: int) -> bool:
    # Typical sweet spot for a 1-2 page resume.
    return 250 <= word_count <= 850


# ---------------------------------------------------------------------------
# Main ATS evaluation
# ---------------------------------------------------------------------------

def evaluate_ats(resume_text: str, job_text: str, skill_analysis: SkillAnalysis) -> ATSResult:
    """Compute a composite ATS score and generate improvement suggestions."""
    checks: list[ATSCheck] = []
    suggestions: list[str] = []

    wc = _word_count(resume_text)

    # 1. Keyword coverage (highest weight).
    ratio = skill_match_ratio(skill_analysis)
    coverage_pct = ratio * 100
    checks.append(ATSCheck(
        name="Keyword Coverage",
        passed=ratio >= 0.6,
        weight=30,
        detail=f"{coverage_pct:.0f}% of job-required skills found in resume.",
    ))
    if ratio < 0.6 and skill_analysis.missing_skills:
        top_missing = ", ".join(skill_analysis.missing_skills[:8])
        suggestions.append(
            f"Add missing keywords from the job description: {top_missing}."
        )

    # 2. Contact information.
    contact = _has_contact_info(resume_text)
    checks.append(ATSCheck(
        name="Contact Information",
        passed=contact,
        weight=10,
        detail="Email or phone number detected." if contact else "No email/phone found.",
    ))
    if not contact:
        suggestions.append("Include a professional email address and phone number at the top.")

    # 3. Section presence.
    sections = _count_sections(resume_text)
    sections_ok = sections >= 4
    checks.append(ATSCheck(
        name="Standard Sections",
        passed=sections_ok,
        weight=15,
        detail=f"{sections} standard sections detected (Experience, Education, Skills, etc.).",
    ))
    if not sections_ok:
        suggestions.append(
            "Add clearly labelled sections (Summary, Experience, Education, Skills, "
            "Projects) so ATS parsers can segment your resume."
        )

    # 4. Action verbs.
    verbs = _action_verb_count(resume_text)
    verbs_ok = verbs >= 8
    checks.append(ATSCheck(
        name="Action Verbs",
        passed=verbs_ok,
        weight=15,
        detail=f"{verbs} strong action verbs found (e.g. developed, led, optimized).",
    ))
    if not verbs_ok:
        suggestions.append(
            "Start bullet points with strong action verbs (developed, led, "
            "architected, optimized, delivered) to describe impact."
        )

    # 5. Quantified achievements.
    quantified = _has_quantified_achievements(resume_text)
    checks.append(ATSCheck(
        name="Quantified Achievements",
        passed=quantified,
        weight=15,
        detail="Numbers/metrics detected that quantify impact." if quantified else "No quantified metrics found.",
    ))
    if not quantified:
        suggestions.append(
            "Quantify achievements with metrics (e.g. 'reduced latency by 30%', "
            "'managed a team of 8') to demonstrate measurable impact."
        )

    # 6. Resume length.
    length_ok = _length_ok(wc)
    checks.append(ATSCheck(
        name="Resume Length",
        passed=length_ok,
        weight=10,
        detail=f"{wc} words — within the recommended 250-850 range." if length_ok
        else f"{wc} words — {'too short' if wc < 250 else 'too long'} for a standard resume.",
    ))
    if not length_ok:
        if wc < 250:
            suggestions.append("Expand your resume with more detail on projects and experience (aim for 250-850 words).")
        else:
            suggestions.append("Trim your resume to 1-2 pages (250-850 words) to keep it ATS-friendly.")

    # 7. Formatting problems.
    problems = _has_problematic_elements(resume_text)
    formatting_ok = len(problems) == 0
    checks.append(ATSCheck(
        name="ATS-Friendly Formatting",
        passed=formatting_ok,
        weight=5,
        detail="No problematic formatting detected." if formatting_ok
        else "; ".join(problems),
    ))
    if not formatting_ok:
        suggestions.append("Avoid tables, columns, images, and text boxes — use a simple single-column layout.")

    # LinkedIn / GitHub presence (bonus, not weighted in score).
    if not _LINKEDIN_RE.search(resume_text):
        suggestions.append("Add a LinkedIn profile URL to strengthen your professional presence.")
    if not _GITHUB_RE.search(resume_text) and any(
        s.lower() in {"python", "javascript", "java", "go", "typescript", "react"}
        for s in skill_analysis.resume_skills
    ):
        suggestions.append("Add a GitHub profile URL — recruiters look for portfolios for technical roles.")

    # Weighted score.
    total_weight = sum(c.weight for c in checks)
    earned = sum(c.weight for c in checks if c.passed)
    score = round((earned / total_weight) * 100, 1) if total_weight else 0.0

    if not suggestions:
        suggestions.append("Your resume is well-optimized for ATS. Keep tailoring keywords to each job application.")

    return ATSResult(score=score, checks=checks, suggestions=suggestions)
