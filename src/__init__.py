"""AI Resume Screener & Job Matcher — source package."""

from .parsing import extract_text, clean_extracted_text
from .preprocessing import preprocess, preprocess_to_string
from .matching import compute_match, keyword_overlap_score, MatchResult
from .skills import extract_skills, skill_match_ratio, SkillAnalysis
from .ats import evaluate_ats, ATSResult
from .visualization import (
    match_gauge, ats_gauge, skill_comparison_chart,
    top_terms_chart, ats_checks_chart, skill_distribution_donut,
)
from .report import generate_report
from .utils import compute_stats, stats_to_dict, rating_label, rating_color

__all__ = [
    "extract_text", "clean_extracted_text",
    "preprocess", "preprocess_to_string",
    "compute_match", "keyword_overlap_score", "MatchResult",
    "extract_skills", "skill_match_ratio", "SkillAnalysis",
    "evaluate_ats", "ATSResult",
    "match_gauge", "ats_gauge", "skill_comparison_chart",
    "top_terms_chart", "ats_checks_chart", "skill_distribution_donut",
    "generate_report",
    "compute_stats", "stats_to_dict", "rating_label", "rating_color",
]
