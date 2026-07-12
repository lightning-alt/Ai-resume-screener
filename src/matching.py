"""Resume-to-job matching using TF-IDF and cosine similarity.

Computes a semantic match score between a resume and a job description,
plus per-term contribution analysis for transparency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import preprocess_to_string


@dataclass
class MatchResult:
    """Container for all match-related outputs."""

    score: float  # 0..1 cosine similarity
    score_percent: float  # 0..100 rounded
    resume_vector: np.ndarray
    job_vector: np.ndarray
    feature_names: list[str]
    resume_terms: dict[str, float] = field(default_factory=dict)
    job_terms: dict[str, float] = field(default_factory=dict)
    overlap_terms: list[str] = field(default_factory=list)


def compute_match(resume_text: str, job_text: str) -> MatchResult:
    """Compute TF-IDF cosine similarity between resume and job description."""
    resume_clean = preprocess_to_string(resume_text)
    job_clean = preprocess_to_string(job_text)

    if not resume_clean or not job_clean:
        return MatchResult(
            score=0.0,
            score_percent=0.0,
            resume_vector=np.zeros(1),
            job_vector=np.zeros(1),
            feature_names=[],
        )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )

    try:
        tfidf_matrix = vectorizer.fit_transform([resume_clean, job_clean])
    except ValueError:
        return MatchResult(
            score=0.0,
            score_percent=0.0,
            resume_vector=np.zeros(1),
            job_vector=np.zeros(1),
            feature_names=[],
        )

    resume_vector = tfidf_matrix[0]
    job_vector = tfidf_matrix[1]
    cosine = cosine_similarity(resume_vector, job_vector)[0][0]
    cosine = float(max(0.0, min(1.0, cosine)))

    # Keyword coverage: of the top JD terms by TF-IDF weight, what fraction
    # appears in the resume? This mirrors what ATS systems actually measure
    # — whether the resume contains the important keywords from the JD —
    # and is far more meaningful than raw document similarity on short texts.
    feature_names_arr = vectorizer.get_feature_names_out()
    job_array_dense = job_vector.toarray().flatten()
    resume_array_dense = resume_vector.toarray().flatten()

    # Take top 30 JD terms by TF-IDF weight.
    top_jd_indices = np.argsort(job_array_dense)[::-1][:30]
    top_jd_indices = [i for i in top_jd_indices if job_array_dense[i] > 0]
    if top_jd_indices:
        covered = sum(
            1 for i in top_jd_indices if resume_array_dense[i] > 0
        )
        keyword_coverage = covered / len(top_jd_indices)
    else:
        keyword_coverage = 0.0

    # Blend: 40% cosine similarity + 60% keyword coverage.
    similarity = 0.4 * cosine + 0.6 * keyword_coverage
    similarity = float(max(0.0, min(1.0, similarity)))

    feature_names = feature_names_arr.tolist()

    # Dense term-score maps for downstream visualisation.
    resume_terms = {
        feature_names[i]: float(resume_array_dense[i])
        for i in np.argsort(resume_array_dense)[::-1][:50]
        if resume_array_dense[i] > 0
    }
    job_terms = {
        feature_names[i]: float(job_array_dense[i])
        for i in np.argsort(job_array_dense)[::-1][:50]
        if job_array_dense[i] > 0
    }
    overlap_terms = sorted(set(resume_terms.keys()) & set(job_terms.keys()))

    return MatchResult(
        score=similarity,
        score_percent=round(similarity * 100, 2),
        resume_vector=resume_vector,
        job_vector=job_vector,
        feature_names=feature_names,
        resume_terms=resume_terms,
        job_terms=job_terms,
        overlap_terms=overlap_terms,
    )


def keyword_overlap_score(resume_text: str, job_text: str) -> float:
    """Simple Jaccard-style keyword overlap as a secondary signal (0..1)."""
    resume_tokens = set(preprocess_to_string(resume_text).split())
    job_tokens = set(preprocess_to_string(job_text).split())
    if not resume_tokens or not job_tokens:
        return 0.0
    intersection = resume_tokens & job_tokens
    union = resume_tokens | job_tokens
    return len(intersection) / len(union) if union else 0.0
