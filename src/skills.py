"""Skill extraction and gap analysis.

Uses a curated taxonomy of technical / soft skills combined with
n-gram scanning over the raw text. The taxonomy is intentionally broad
but easy to extend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

# ---------------------------------------------------------------------------
# Skill taxonomy — extend freely; order does not matter.
# ---------------------------------------------------------------------------

TECHNICAL_SKILLS: list[str] = [
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "c", "go", "golang",
    "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "perl", "dart",
    "objective-c", "elixir", "haskell", "lua", "solidity", "bash", "shell", "powershell",
    # Frontend
    "html", "css", "sass", "scss", "tailwind", "bootstrap", "react", "redux",
    "angular", "vue", "svelte", "next.js", "nuxt", "jquery", "webpack", "vite",
    "responsive design", "material ui", "storybook",
    # Backend / frameworks
    "django", "flask", "fastapi", "spring", "spring boot", "express", "express.js",
    "node.js", "rails", "laravel", "asp.net", ".net", ".net core", "graphql", "rest",
    "rest api", "grpc", "microservices", "celery", "rabbitmq", "kafka",
    # Databases
    "sql", "mysql", "postgresql", "postgres", "mongodb", "redis", "cassandra",
    "elasticsearch", "dynamodb", "sqlite", "oracle", "snowflake", "bigquery",
    "databricks", "neo4j", "mariadb", "firebase",
    # Cloud / DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
    "terraform", "ansible", "jenkins", "ci/cd", "github actions", "gitlab ci",
    "circleci", "helm", "argocd", "prometheus", "grafana", "datadog",
    "linux", "unix", "nginx", "apache", "serverless", "lambda", "ec2", "s3",
    "cloudformation", "openshift",
    # Data / ML
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
    "sklearn", "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "tableau", "power bi", "looker", "spark", "hadoop", "airflow", "dbt",
    "spark sql", "etl", "data warehouse", "data lake", "data pipeline",
    "feature engineering", "model deployment", "mlops", "llm", "hugging face",
    "openai", "langchain", "xgboost", "lightgbm", "catboost",
    # Mobile
    "android", "ios", "react native", "flutter", "xamarin", "ionic",
    # Testing / QA
    "pytest", "unittest", "junit", "selenium", "cypress", "jest", "mocha",
    "testng", "playwright", "postman", "jmeter", "load testing",
    # Tools / collaboration
    "git", "github", "gitlab", "bitbucket", "jira", "confluence", "slack",
    "trello", "asana", "notion", "figma", "adobe xd", "sketch", "zeplin",
    # Security
    "oauth", "jwt", "sso", "penetration testing", "owasp", "encryption",
    "cybersecurity", "siem", "firewall", "iso 27001",
    # Architecture / patterns
    "agile", "scrum", "kanban", "tdd", "bdd", "ddd", "design patterns",
    "system design", "oop", "functional programming", "concurrency",
    # Other
    "excel", "vba", "sap", "salesforce", "hubspot", "quickbooks",
]

SOFT_SKILLS: list[str] = [
    "communication", "teamwork", "leadership", "problem solving", "critical thinking",
    "time management", "adaptability", "collaboration", "creativity",
    "interpersonal skills", "decision making", "conflict resolution",
    "negotiation", "presentation", "mentoring", "coaching", "project management",
    "stakeholder management", "cross-functional", "self-motivated", "detail oriented",
    "analytical", "strategic planning", "work ethic", "emotional intelligence",
    "public speaking", "writing", "active listening", "flexibility",
]

# Normalised lookup: lowercase alias -> canonical display name.
_ALL_SKILLS = {s.lower(): s for s in TECHNICAL_SKILLS + SOFT_SKILLS}


@dataclass
class SkillAnalysis:
    """Result of comparing resume skills against job-required skills."""

    resume_skills: list[str]
    job_skills: list[str]
    matching_skills: list[str]
    missing_skills: list[str]
    extra_skills: list[str]  # in resume but not in job


def _extract_skills_from_text(text: str) -> list[str]:
    """Return canonical skill names found in ``text`` (deduplicated, ordered)."""
    if not text:
        return []

    lowered = text.lower()
    found: list[str] = []
    seen: set[str] = set()

    # Sort longer aliases first so "spring boot" matches before "spring".
    for alias in sorted(_ALL_SKILLS.keys(), key=len, reverse=True):
        if alias in seen:
            continue
        # Word-boundary safe match for short tokens; substring for multiword.
        if " " in alias or any(ch in alias for ch in "+#.-"):
            if alias in lowered:
                found.append(_ALL_SKILLS[alias])
                seen.add(alias)
        else:
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, lowered):
                found.append(_ALL_SKILLS[alias])
                seen.add(alias)

    return found


def extract_skills(resume_text: str, job_text: str) -> SkillAnalysis:
    """Extract skills from both texts and compute the gap analysis."""
    resume_skills = _extract_skills_from_text(resume_text)
    job_skills = _extract_skills_from_text(job_text)

    resume_set = {s.lower() for s in resume_skills}
    job_set = {s.lower() for s in job_skills}

    matching = [s for s in resume_skills if s.lower() in job_set]
    missing = [s for s in job_skills if s.lower() not in resume_set]
    extra = [s for s in resume_skills if s.lower() not in job_set]

    # Deduplicate while preserving order.
    matching = list(dict.fromkeys(matching))
    missing = list(dict.fromkeys(missing))
    extra = list(dict.fromkeys(extra))

    return SkillAnalysis(
        resume_skills=list(dict.fromkeys(resume_skills)),
        job_skills=list(dict.fromkeys(job_skills)),
        matching_skills=matching,
        missing_skills=missing,
        extra_skills=extra,
    )


def skill_match_ratio(analysis: SkillAnalysis) -> float:
    """Fraction of job-required skills present in the resume (0..1)."""
    if not analysis.job_skills:
        return 0.0
    return len(analysis.matching_skills) / len(analysis.job_skills)
