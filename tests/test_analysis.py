"""
Unit tests for analyze.py
Run: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

import pytest
from analyze import (
    extract_skills_advanced,
    extract_experience,
    extract_contact_info,
    detect_sections,
    cluster_skills,
    detect_role,
    semantic_match,
    calculate_score,
    calculate_ats_score,
    match_resume_with_job,
    ALL_SKILLS,
)

SAMPLE_RESUME = """
John Doe
john.doe@email.com | +91 98765 43210 | github.com/johndoe | linkedin.com/in/johndoe

Summary
Experienced Python developer with 3 years of experience in backend development.

Skills
Python, Django, FastAPI, PostgreSQL, Docker, Git, REST API, Redis

Experience
Backend Developer — TechCorp (2021–2024)
- Built REST APIs using FastAPI and Django with 3 years of Python experience
- Managed PostgreSQL databases and Redis caching
- Deployed services using Docker and GitHub Actions

Education
B.Tech Computer Science — XYZ University, 2021

Projects
Resume Analyzer — Built with FastAPI, Python, spaCy, PostgreSQL
"""

SAMPLE_JD = """
We are looking for a Backend Developer with experience in:
- Python and Django or FastAPI
- PostgreSQL database management
- Docker and CI/CD pipelines
- REST API design
- Git version control
"""


class TestSkillExtraction:
    def test_extracts_known_skills(self):
        skills = extract_skills_advanced(SAMPLE_RESUME.lower())
        assert "python" in skills
        assert "django" in skills
        assert "fastapi" in skills
        assert "postgresql" in skills
        assert "docker" in skills

    def test_returns_sorted_list(self):
        skills = extract_skills_advanced(SAMPLE_RESUME.lower())
        assert skills == sorted(skills)

    def test_no_duplicates(self):
        skills = extract_skills_advanced(SAMPLE_RESUME.lower())
        assert len(skills) == len(set(skills))

    def test_empty_text(self):
        assert extract_skills_advanced("") == []


class TestContactInfo:
    def test_extracts_email(self):
        info = extract_contact_info(SAMPLE_RESUME)
        assert info["email"] == "john.doe@email.com"

    def test_extracts_phone(self):
        info = extract_contact_info(SAMPLE_RESUME)
        assert info["phone"] is not None

    def test_extracts_github(self):
        info = extract_contact_info(SAMPLE_RESUME)
        assert info["github"] is not None
        assert "johndoe" in info["github"]

    def test_extracts_linkedin(self):
        info = extract_contact_info(SAMPLE_RESUME)
        assert info["linkedin"] is not None


class TestSectionDetection:
    def test_detects_experience(self):
        sections = detect_sections(SAMPLE_RESUME)
        assert sections["experience"] is True

    def test_detects_education(self):
        sections = detect_sections(SAMPLE_RESUME)
        assert sections["education"] is True

    def test_detects_skills(self):
        sections = detect_sections(SAMPLE_RESUME)
        assert sections["skills"] is True

    def test_detects_projects(self):
        sections = detect_sections(SAMPLE_RESUME)
        assert sections["projects"] is True


class TestExperienceExtraction:
    def test_finds_years(self):
        exp = extract_experience(SAMPLE_RESUME.lower())
        assert len(exp) > 0

    def test_empty_text(self):
        exp = extract_experience("")
        assert exp == {}


class TestClustering:
    def test_returns_dict(self):
        skills = ["python", "django", "react", "docker", "pandas"]
        clusters = cluster_skills(skills, n_clusters=2)
        assert isinstance(clusters, dict)

    def test_all_skills_present(self):
        skills = ["python", "django", "react", "docker"]
        clusters = cluster_skills(skills, n_clusters=2)
        all_in_clusters = [s for group in clusters.values() for s in group]
        assert set(skills) == set(all_in_clusters)

    def test_single_skill(self):
        clusters = cluster_skills(["python"])
        assert clusters == {"General": ["python"]}


class TestRoleDetection:
    def test_detects_backend_role(self):
        skills = ["python", "django", "fastapi", "postgresql", "redis"]
        role = detect_role(skills)
        assert role["primary_role"] is not None
        assert role["confidence"] > 0

    def test_empty_skills(self):
        role = detect_role([])
        assert role["primary_role"] == "Unknown"
        assert role["confidence"] == 0.0

    def test_returns_all_scores(self):
        skills = ["python", "react", "docker"]
        role = detect_role(skills)
        assert "all_scores" in role
        assert len(role["all_scores"]) > 0


class TestSemanticMatch:
    def test_matches_identical(self):
        matched, missing = semantic_match(["python"], ["python"])
        assert "python" in matched

    def test_empty_resume_skills(self):
        matched, missing = semantic_match([], ["python", "django"])
        assert matched == []
        assert "python" in missing

    def test_empty_job_skills(self):
        matched, missing = semantic_match(["python"], [])
        assert matched == []
        assert missing == []


class TestScoring:
    def test_perfect_match(self):
        score = calculate_score(["python", "django"], ["python", "django"], {})
        assert score > 0
        assert score <= 100

    def test_no_match(self):
        score = calculate_score([], ["python", "django"], {})
        assert score == 0

    def test_no_job_skills(self):
        assert calculate_score(["python"], [], {}) == 0


class TestATS:
    def test_ats_score_range(self):
        sections = detect_sections(SAMPLE_RESUME)
        result = calculate_ats_score(SAMPLE_RESUME, SAMPLE_JD, sections)
        assert 0 <= result["ats_score"] <= 100

    def test_ats_has_required_keys(self):
        sections = detect_sections(SAMPLE_RESUME)
        result = calculate_ats_score(SAMPLE_RESUME, SAMPLE_JD, sections)
        assert "ats_score" in result
        assert "ats_level" in result
        assert "word_count" in result


class TestMainFunction:
    def test_full_analysis_returns_expected_keys(self):
        result = match_resume_with_job(SAMPLE_RESUME, SAMPLE_JD)
        expected = [
            "match_score", "confidence", "ats", "insight", "suggestions",
            "matched_skills", "missing_skills", "resume_skills", "job_skills",
            "matched_by_category", "skill_clusters", "role_detection",
            "experience", "fresher", "sections_detected", "contact_info",
            "total_skills_in_db",
        ]
        for key in expected:
            assert key in result, f"Missing key: {key}"

    def test_score_in_range(self):
        result = match_resume_with_job(SAMPLE_RESUME, SAMPLE_JD)
        assert 0 <= result["match_score"] <= 100

    def test_empty_resume(self):
        result = match_resume_with_job("", SAMPLE_JD)
        assert result["match_score"] == 0
        assert "error" in result

    def test_empty_jd(self):
        result = match_resume_with_job(SAMPLE_RESUME, "")
        assert result["match_score"] == 0
        assert "error" in result

    def test_all_skills_in_db_count(self):
        result = match_resume_with_job(SAMPLE_RESUME, SAMPLE_JD)
        assert result["total_skills_in_db"] == len(ALL_SKILLS)
