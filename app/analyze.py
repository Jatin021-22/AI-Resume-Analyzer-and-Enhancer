"""
analyze.py  —  Production-grade Resume Analysis Engine
• PDF extraction       (pdfplumber)
• Contact info         (regex + spaCy NER)
• Section detection    (regex heuristics)
• Skill extraction     (multi-pass: substring + spaCy)
• Skill clustering     (KMeans on sentence-transformer embeddings)
• Role/intent detect   (cosine sim to role profiles)
• Semantic matching    (sentence-transformers)
• ATS scoring          (keyword density + section + length)
• Match scoring        (weighted: skills + exp + diversity)
• Dynamic insights & suggestions
"""

import re
import pdfplumber
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from loguru import logger

# ── Load models once ─────────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("Run: python -m spacy download en_core_web_sm")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("NLP models loaded")

# ─────────────────────────────────────────────────────────────
# SKILLS DATABASE
# ─────────────────────────────────────────────────────────────
SKILLS_DB: dict = {
    "languages": [
        "python", "javascript", "typescript", "java", "go", "rust",
        "c++", "c#", "php", "ruby", "swift", "kotlin", "scala",
        "r", "bash", "shell", "perl", "matlab",
    ],
    "frameworks": [
        "django", "fastapi", "flask", "react", "vue", "angular",
        "nextjs", "nuxtjs", "spring", "express", "laravel", "rails",
        "nestjs", "svelte", "tailwind", "bootstrap", "asp.net",
        "starlette", "celery",
    ],
    "cloud_devops": [
        "aws", "gcp", "azure", "docker", "kubernetes", "terraform",
        "ansible", "github actions", "jenkins", "ci/cd", "heroku",
        "vercel", "netlify", "nginx", "linux", "ubuntu", "helm",
        "prometheus", "grafana", "cloudformation",
    ],
    "databases": [
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "sqlite", "oracle", "dynamodb", "firebase", "supabase",
        "cassandra", "mariadb", "neo4j", "influxdb", "clickhouse",
    ],
    "data_ai": [
        "pandas", "numpy", "tensorflow", "pytorch", "scikit-learn",
        "keras", "opencv", "matplotlib", "seaborn", "spark",
        "machine learning", "deep learning", "nlp", "computer vision",
        "data analysis", "power bi", "tableau", "airflow", "mlflow",
        "hugging face", "langchain", "rag", "llm", "bert",
    ],
    "tools": [
        "git", "github", "gitlab", "jira", "figma", "postman",
        "swagger", "graphql", "rest api", "microservices",
        "agile", "scrum", "kanban", "vs code", "webpack", "vite",
    ],
    "mobile": [
        "react native", "flutter", "android", "ios", "expo",
        "swift", "kotlin", "xamarin",
    ],
    "testing": [
        "pytest", "jest", "selenium", "unittest", "cypress",
        "tdd", "bdd", "playwright",
    ],
    "security": [
        "oauth", "jwt", "ssl", "tls", "encryption", "owasp",
        "penetration testing", "sso", "ldap",
    ],
}

ALL_SKILLS: list = [s for skills in SKILLS_DB.values() for s in skills]
SKILL_CATEGORY: dict = {
    skill: cat for cat, skills in SKILLS_DB.items() for skill in skills
}

# Role profiles for intent detection
ROLE_PROFILES: dict = {
    "Backend Developer": [
        "python", "django", "fastapi", "flask", "postgresql",
        "redis", "rest api", "docker", "jwt",
    ],
    "Frontend Developer": [
        "javascript", "typescript", "react", "vue", "angular",
        "nextjs", "tailwind", "css", "html", "webpack",
    ],
    "Full Stack Developer": [
        "react", "nodejs", "python", "postgresql", "docker",
        "rest api", "git", "aws",
    ],
    "Data Scientist": [
        "python", "pandas", "numpy", "scikit-learn", "tensorflow",
        "pytorch", "matplotlib", "machine learning",
    ],
    "DevOps Engineer": [
        "docker", "kubernetes", "terraform", "ansible", "aws",
        "ci/cd", "jenkins", "linux", "prometheus", "grafana",
    ],
    "Mobile Developer": [
        "react native", "flutter", "android", "ios", "swift", "kotlin",
    ],
    "ML Engineer": [
        "python", "tensorflow", "pytorch", "scikit-learn",
        "mlflow", "airflow", "docker", "aws", "bert",
    ],
    "Security Engineer": [
        "penetration testing", "owasp", "ssl", "encryption",
        "oauth", "jwt", "linux", "python",
    ],
}

# Pre-compute role embeddings once at startup
ROLE_EMBEDDINGS: dict = {
    role: embedder.encode(" ".join(skills), convert_to_numpy=True)
    for role, skills in ROLE_PROFILES.items()
}


# ─────────────────────────────────────────────────────────────
# PDF EXTRACTION
# ─────────────────────────────────────────────────────────────
def extract_text(file_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception as exc:
        logger.error(f"PDF extraction error: {exc}")
    return text.strip()


# ─────────────────────────────────────────────────────────────
# CONTACT INFO
# ─────────────────────────────────────────────────────────────
def extract_contact_info(text: str) -> dict:
    email    = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)
    phone    = re.search(r"(\+?\d[\d\s\-(). ]{7,14}\d)", text)
    linkedin = re.search(r"linkedin\.com/in/[\w\-]+", text, re.IGNORECASE)
    github   = re.search(r"github\.com/[\w\-]+", text, re.IGNORECASE)
    name_doc = nlp(text[:300])
    name     = next((e.text for e in name_doc.ents if e.label_ == "PERSON"), None)
    return {
        "name":     name,
        "email":    email.group(0)    if email    else None,
        "phone":    phone.group(0)    if phone    else None,
        "linkedin": linkedin.group(0) if linkedin else None,
        "github":   github.group(0)   if github   else None,
    }


# ─────────────────────────────────────────────────────────────
# SECTION DETECTION
# ─────────────────────────────────────────────────────────────
SECTION_PATTERNS = {
    "education":      r"\b(education|academic|qualification|degree|university|college)\b",
    "experience":     r"\b(experience|work history|employment|career|internship)\b",
    "skills":         r"\b(skills|technical skills|core competencies|expertise)\b",
    "projects":       r"\b(projects|portfolio|personal projects|open.?source)\b",
    "certifications": r"\b(certifications?|certificates?|courses?|training)\b",
    "summary":        r"\b(summary|objective|profile|about me|overview)\b",
}

def detect_sections(text: str) -> dict:
    lower = text.lower()
    return {
        sec: bool(re.search(pat, lower))
        for sec, pat in SECTION_PATTERNS.items()
    }


# ─────────────────────────────────────────────────────────────
# SKILL EXTRACTION  (multi-pass)
# ─────────────────────────────────────────────────────────────
def extract_skills_advanced(text: str) -> list:
    text_lower = text.lower()
    doc = nlp(text_lower)
    found: set = set()

    # Pass 1 — whole-word substring match (catches multi-word skills)
    for skill in ALL_SKILLS:
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
            found.add(skill)

    # Pass 2 — spaCy noun chunks
    for chunk in doc.noun_chunks:
        for skill in ALL_SKILLS:
            if skill in chunk.text:
                found.add(skill)

    # Pass 3 — single tokens
    for token in doc:
        if token.text in ALL_SKILLS:
            found.add(token.text)

    return sorted(found)


# ─────────────────────────────────────────────────────────────
# SKILL CLUSTERING  (KMeans on embeddings)
# ─────────────────────────────────────────────────────────────
def cluster_skills(skills: list, n_clusters: int = 4) -> dict:
    """
    Groups skills into semantic clusters using KMeans.
    Each cluster is named after the skill closest to the centroid.
    """
    if len(skills) < 2:
        return {"General": skills}

    n = min(n_clusters, len(skills))
    embs = embedder.encode(skills, convert_to_numpy=True)
    embs = normalize(embs)

    km = KMeans(n_clusters=n, random_state=42, n_init=10)
    labels = km.fit_predict(embs)

    raw_clusters: dict = {}
    for skill, label in zip(skills, labels.tolist()):
        raw_clusters.setdefault(label, []).append(skill)

    named: dict = {}
    for lid, cluster_list in raw_clusters.items():
        centroid   = km.cluster_centers_[lid]
        cembs      = embedder.encode(cluster_list, convert_to_numpy=True)
        distances  = np.linalg.norm(cembs - centroid, axis=1)
        rep_skill  = cluster_list[int(np.argmin(distances))]
        named[rep_skill] = cluster_list

    return named


# ─────────────────────────────────────────────────────────────
# ROLE / INTENT DETECTION
# ─────────────────────────────────────────────────────────────
def detect_role(resume_skills: list) -> dict:
    """
    Cosine similarity between resume skill set and role profiles.
    Returns primary + secondary role with confidence scores.
    """
    if not resume_skills:
        return {"primary_role": "Unknown", "confidence": 0.0, "all_scores": {}}

    resume_emb = embedder.encode(" ".join(resume_skills), convert_to_numpy=True)
    resume_emb = normalize(resume_emb.reshape(1, -1))[0]

    scores: dict = {}
    for role, role_emb in ROLE_EMBEDDINGS.items():
        normed       = normalize(role_emb.reshape(1, -1))[0]
        scores[role] = round(float(np.dot(resume_emb, normed)), 4)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary, top  = sorted_scores[0]
    secondary     = sorted_scores[1][0] if len(sorted_scores) > 1 else None

    return {
        "primary_role":   primary,
        "secondary_role": secondary,
        "confidence":     round(top * 100, 1),
        "all_scores":     {r: round(s * 100, 1) for r, s in sorted_scores},
    }


# ─────────────────────────────────────────────────────────────
# EXPERIENCE EXTRACTION
# ─────────────────────────────────────────────────────────────
def extract_experience(text: str) -> dict:
    doc = nlp(text.lower())
    data: dict = {}
    year_pat  = r"(\d+\.?\d*)\+?\s*(years?|yrs?)"
    month_pat = r"(\d+\.?\d*)\+?\s*(months?|mos?)"

    for sent in doc.sents:
        sentence = sent.text
        m = re.search(year_pat, sentence) or re.search(month_pat, sentence)
        if m:
            duration = m.group(1) + " " + m.group(2)
            for skill in ALL_SKILLS:
                if skill in sentence and skill not in data:
                    data[skill] = duration
    return data


# ─────────────────────────────────────────────────────────────
# SEMANTIC MATCHING
# ─────────────────────────────────────────────────────────────
def semantic_match(
    resume_skills: list,
    job_skills: list,
    threshold: float = 0.60,
) -> tuple:
    if not resume_skills or not job_skills:
        return [], list(job_skills)

    r_embs = embedder.encode(resume_skills, convert_to_tensor=True)
    j_embs = embedder.encode(job_skills,    convert_to_tensor=True)

    matched, missing = [], []
    for i, js in enumerate(job_skills):
        best = float(util.cos_sim(j_embs[i], r_embs)[0].max())
        (matched if best >= threshold else missing).append(js)

    return matched, missing


# ─────────────────────────────────────────────────────────────
# ATS SCORE
# ─────────────────────────────────────────────────────────────
def calculate_ats_score(text: str, job_description: str, sections: dict) -> dict:
    tl = text.lower()
    jl = job_description.lower()

    jd_words     = set(re.findall(r"\b\w{4,}\b", jl))
    resume_words = set(re.findall(r"\b\w{4,}\b", tl))
    overlap      = len(jd_words & resume_words) / max(len(jd_words), 1)
    keyword_score = min(overlap * 150, 40)

    critical      = ["experience", "skills", "education"]
    section_score = sum(10 for s in critical if sections.get(s))

    wc = len(text.split())
    if 400 <= wc <= 1000:
        length_score = 20
    elif wc < 200:
        length_score = 5
    elif wc < 400:
        length_score = 12
    else:
        length_score = 15

    contact_bonus = 10 if re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text) else 0
    total = max(0, min(100, int(keyword_score + section_score + length_score + contact_bonus)))

    return {
        "ats_score":    total,
        "keyword_score": int(keyword_score),
        "section_score": section_score,
        "length_score":  length_score,
        "word_count":    wc,
        "ats_level":     "Good" if total >= 70 else "Average" if total >= 45 else "Poor",
    }


# ─────────────────────────────────────────────────────────────
# MATCH SCORE
# ─────────────────────────────────────────────────────────────
def calculate_score(matched: list, job_skills: list, experience: dict) -> int:
    if not job_skills:
        return 0

    fresher      = len(experience) == 0
    skill_weight = 80 if fresher else 50
    skill_score  = (len(matched) / len(job_skills)) * skill_weight

    exp_score = 0
    if not fresher:
        for skill in matched:
            if skill in experience:
                val = experience[skill]
                nm  = re.search(r"\d+\.?\d*", val)
                if nm:
                    num   = float(nm.group())
                    years = num / 12 if "month" in val else num
                    exp_score += min(years * 5, 10)
        exp_score = min(exp_score, 30)

    cats           = {SKILL_CATEGORY.get(s) for s in matched if SKILL_CATEGORY.get(s)}
    diversity_bonus = min(len(cats) * 2, 10) if not fresher else 0

    return max(0, min(100, int(skill_score + exp_score + diversity_bonus)))


# ─────────────────────────────────────────────────────────────
# INSIGHT
# ─────────────────────────────────────────────────────────────
def generate_insight(
    score: int, matched: list, missing: list,
    experience: dict, fresher: bool, role_info: dict
) -> str:
    top_missing = ", ".join(missing[:3]) if missing else None
    top_matched = ", ".join(matched[:3]) if matched else None
    role        = role_info.get("primary_role", "this field")

    if fresher:
        if score >= 75:
            return (f"Strong fresher profile"
                + (f" with solid skills in {top_matched}" if top_matched else "")
                + f". You appear well-suited for a {role} role — "
                "build portfolio projects to stand out.")
        elif score >= 50:
            return (f"Decent match for a {role} fresher. "
                + (f"Strengthen {top_missing} " if top_missing else "")
                + "and add 1–2 real projects to improve chances.")
        else:
            return (f"Low match. Key gaps: {top_missing or 'multiple areas'}. "
                "Targeted upskilling before applying is recommended.")
    else:
        if score >= 80:
            return (f"Strong experienced candidate"
                + (f" with solid coverage in {top_matched}" if top_matched else "")
                + f". Well-aligned with {role} requirements.")
        elif score >= 55:
            return (f"Moderate match for {role}. "
                + (f"Bridging gaps in {top_missing} " if top_missing else "")
                + "would meaningfully improve your profile.")
        else:
            return (f"Significant skill gaps for {role}: {top_missing or 'review JD'}. "
                "Targeted upskilling is strongly recommended.")


# ─────────────────────────────────────────────────────────────
# SUGGESTIONS
# ─────────────────────────────────────────────────────────────
def generate_suggestions(
    missing: list, fresher: bool, matched: list,
    experience: dict, sections: dict, role_info: dict
) -> list:
    suggestions: list = []
    role = role_info.get("primary_role", "your target role")

    if fresher:
        suggestions.append(
            f"Build 2–3 portfolio projects tailored to {role} and host them on GitHub")
    else:
        suggestions.append(
            "Quantify achievements (e.g. 'reduced load time by 40%', 'scaled to 1M users')")

    if not sections.get("summary"):
        suggestions.append(
            "Add a professional summary — 3 sentences on who you are and your key strengths")
    if not sections.get("projects") and fresher:
        suggestions.append(
            "Add a Projects section with tech stack, your role, and measurable outcome")
    if not sections.get("certifications"):
        suggestions.append(
            f"Add relevant certifications — cloud, {role}-specific badges add credibility")

    for skill in missing[:5]:
        cat = SKILL_CATEGORY.get(skill, "general")
        if cat == "cloud_devops":
            suggestions.append(f"Get hands-on with {skill} using a free-tier cloud account")
        elif cat == "data_ai":
            suggestions.append(f"Build a practical {skill} project and publish on Kaggle/GitHub")
        elif cat == "testing":
            suggestions.append(f"Add {skill} tests to your existing projects")
        else:
            suggestions.append(f"Learn {skill} through a small hands-on project")

    if matched and not missing:
        suggestions.append(
            "All key skills matched — tailor your resume summary to the exact job title")

    return suggestions[:10]


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def match_resume_with_job(resume_text: str, job_description: str) -> dict:
    resume_lower = resume_text.lower().strip()
    jd_lower     = job_description.lower().strip()

    if not resume_lower:
        return {"match_score": 0, "error": "Could not extract text from resume"}
    if not jd_lower:
        return {"match_score": 0, "error": "Job description is empty"}

    contact_info  = extract_contact_info(resume_text)
    sections      = detect_sections(resume_text)
    resume_skills = extract_skills_advanced(resume_lower)
    job_skills    = [s for s in ALL_SKILLS
                     if re.search(r"\b" + re.escape(s) + r"\b", jd_lower)]
    experience    = extract_experience(resume_lower)
    fresher       = len(experience) == 0

    matched, missing = semantic_match(resume_skills, job_skills)
    role_info        = detect_role(resume_skills)
    skill_clusters   = cluster_skills(resume_skills) if len(resume_skills) >= 2 else {}

    score      = calculate_score(matched, job_skills, experience)
    ats_result = calculate_ats_score(resume_text, job_description, sections)
    confidence = "High" if score >= 75 else "Medium" if score >= 50 else "Low"

    matched_by_category: dict = {}
    for skill in matched:
        cat = SKILL_CATEGORY.get(skill, "other")
        matched_by_category.setdefault(cat, []).append(skill)

    insight     = generate_insight(score, matched, missing, experience, fresher, role_info)
    suggestions = generate_suggestions(missing, fresher, matched, experience, sections, role_info)

    return {
        "match_score":          score,
        "confidence":           confidence,
        "ats":                  ats_result,
        "insight":              insight,
        "suggestions":          suggestions,
        "matched_skills":       matched,
        "missing_skills":       missing,
        "resume_skills":        resume_skills,
        "job_skills":           job_skills,
        "matched_by_category":  matched_by_category,
        "skill_clusters":       skill_clusters,
        "role_detection":       role_info,
        "experience":           experience,
        "fresher":              fresher,
        "sections_detected":    sections,
        "contact_info":         contact_info,
        "total_skills_in_db":   len(ALL_SKILLS),
    }
