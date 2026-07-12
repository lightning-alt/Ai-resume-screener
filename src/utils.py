"""Utility functions: resume statistics, text metrics, and shared helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .skills import SkillAnalysis


@dataclass
class DocumentStats:
    """Lightweight statistics about a text document."""

    word_count: int
    char_count: int
    sentence_count: int
    reading_time: str
    skill_count: int
    detected_skills: list[str]


def compute_stats(text: str, skills: list[str] | None = None) -> DocumentStats:
    """Compute readability and structural statistics for a block of text."""
    if not text or not text.strip():
        return DocumentStats(
            word_count=0, char_count=0, sentence_count=0,
            reading_time="0 min", skill_count=0, detected_skills=[],
        )

    words = text.split()
    word_count = len(words)
    char_count = len(text)

    # Sentence count: split on . ! ? followed by whitespace.
    import re
    sentences = re.split(r"[.!?]+\s+", text)
    sentence_count = len([s for s in sentences if s.strip()])

    # Reading time: ~200 wpm average.
    minutes = word_count / 200
    if minutes < 1:
        reading_time = f"{math.ceil(minutes * 60)} sec"
    else:
        reading_time = f"{round(minutes, 1)} min"

    detected = skills or []
    return DocumentStats(
        word_count=word_count,
        char_count=char_count,
        sentence_count=sentence_count,
        reading_time=reading_time,
        skill_count=len(detected),
        detected_skills=detected,
    )


def stats_to_dict(stats: DocumentStats) -> dict:
    """Convert DocumentStats to a plain dict for the report module."""
    return {
        "word_count": stats.word_count,
        "char_count": stats.char_count,
        "sentence_count": stats.sentence_count,
        "reading_time": stats.reading_time,
        "skill_count": stats.skill_count,
        "detected_skills": stats.detected_skills,
    }


def rating_label(score: float) -> str:
    """Human-readable rating for a 0-100 score."""
    if score >= 75:
        return "Excellent"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Needs Work"
    return "Low"


def rating_color(score: float) -> str:
    """Hex color for a 0-100 score, matching the dashboard palette."""
    if score >= 75:
        return "#16A34A"
    if score >= 50:
        return "#F59E0B"
    return "#DC2626"


def truncate(text: str, max_chars: int = 500) -> str:
    """Truncate text for preview display, adding an ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"
