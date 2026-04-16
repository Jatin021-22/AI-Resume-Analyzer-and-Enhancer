import re
import pdfplumber
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from loguru import logger
import platform
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageFilter, ImageEnhance

pytesseract.pytesseract.tesseract_cmd = \
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("Run: python -m spacy download en_core_web_sm")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("NLP models loaded")


# ─────────────────────────────────────────────────────────────
# UNIVERSAL SKILLS DATABASE
# ─────────────────────────────────────────────────────────────
SKILLS_DB: dict = {
    "programming": [
        "python", "javascript", "typescript", "java", "go", "rust",
        "c++", "c#", "php", "ruby", "swift", "kotlin", "scala",
        "r", "bash", "shell", "perl", "matlab", "dart",
    ],
    "web_frameworks": [
        "django", "fastapi", "flask", "react", "vue", "angular",
        "nextjs", "nuxtjs", "spring", "express", "laravel", "rails",
        "nestjs", "svelte", "tailwind", "bootstrap", "asp.net",
        "starlette", "celery", "jquery",
    ],
    "cloud_devops": [
        "aws", "gcp", "azure", "docker", "kubernetes", "terraform",
        "ansible", "github actions", "jenkins", "ci/cd", "heroku",
        "vercel", "netlify", "nginx", "linux", "ubuntu", "helm",
        "prometheus", "grafana", "cloudformation", "git", "github",
        "gitlab", "bitbucket",
    ],
    "databases": [
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "sqlite", "oracle", "dynamodb", "firebase", "supabase",
        "cassandra", "mariadb", "neo4j", "influxdb", "clickhouse",
        "sql", "nosql",
    ],
    "data_ai": [
        "pandas", "numpy", "tensorflow", "pytorch", "scikit-learn",
        "keras", "opencv", "matplotlib", "seaborn", "spark",
        "machine learning", "deep learning", "nlp", "computer vision",
        "data analysis", "power bi", "tableau", "airflow", "mlflow",
        "hugging face", "langchain", "rag", "llm", "bert",
        "data science", "statistics", "data visualization",
        "big data", "hadoop", "data engineering",
    ],
    "tools_general": [
        "jira", "figma", "postman", "swagger", "graphql",
        "rest api", "microservices", "agile", "scrum", "kanban",
        "vs code", "webpack", "vite", "eslint", "slack",
        "microsoft office", "excel", "word", "powerpoint",
        "google workspace", "trello", "notion", "confluence",
    ],
    "mobile": [
        "react native", "flutter", "android", "ios", "expo",
        "swift", "kotlin", "xamarin", "mobile development",
    ],
    "testing": [
        "pytest", "jest", "selenium", "unittest", "cypress",
        "tdd", "bdd", "playwright", "qa testing",
        "manual testing", "automation testing",
    ],
    "security": [
        "oauth", "jwt", "ssl", "tls", "encryption", "owasp",
        "penetration testing", "sso", "ldap", "cybersecurity",
        "network security", "ethical hacking", "firewall",
        "vulnerability assessment",
    ],
    "design": [
        "figma", "adobe xd", "sketch", "photoshop", "illustrator",
        "invision", "zeplin", "wireframing", "prototyping",
        "ui design", "ux design", "ui/ux", "user research",
        "user experience", "graphic design", "canva",
        "adobe creative suite", "after effects", "premiere pro",
    ],
    "business_skills": [
        "project management", "product management", "business analysis",
        "strategic planning", "business development", "stakeholder management",
        "change management", "risk management", "process improvement",
        "operations management", "supply chain", "logistics",
        "vendor management", "contract management", "budgeting",
        "forecasting", "kpi", "okr", "six sigma", "lean",
    ],
    "finance_accounting": [
        "financial analysis", "financial modeling", "accounting",
        "bookkeeping", "taxation", "auditing", "budgeting",
        "financial reporting", "gaap", "ifrs", "quickbooks",
        "tally", "sap", "oracle financials", "cost accounting",
        "investment analysis", "cash flow", "payroll",
    ],
    "sales_marketing": [
        "sales", "business development", "lead generation",
        "crm", "salesforce", "digital marketing", "seo",
        "sem", "google analytics", "social media marketing",
        "content marketing", "email marketing", "copywriting",
        "brand management", "market research", "advertising",
        "hubspot", "mailchimp", "facebook ads", "google ads",
        "b2b sales", "b2c sales", "account management",
    ],
    "hr_recruitment": [
        "recruitment", "talent acquisition", "hr management",
        "performance management", "employee relations",
        "onboarding", "training and development", "payroll",
        "compensation and benefits", "labor law",
        "organizational development", "succession planning",
        "employer branding", "workforce planning",
    ],
    "healthcare": [
        "patient care", "clinical research", "medical records",
        "nursing", "pharmacy", "diagnosis", "treatment planning",
        "ehr", "emr", "hipaa", "medical coding", "icd-10",
        "healthcare management", "public health", "epidemiology",
        "clinical trials", "pharmacology", "anatomy", "physiology",
        "first aid", "bls", "acls", "medical billing",
    ],
    "education": [
        "curriculum development", "lesson planning", "teaching",
        "instructional design", "e-learning", "lms",
        "classroom management", "assessment", "student engagement",
        "special education", "adult learning", "training",
        "coaching", "mentoring", "educational technology",
    ],
    "legal": [
        "legal research", "contract drafting", "litigation",
        "corporate law", "intellectual property", "compliance",
        "regulatory affairs", "legal writing", "case management",
        "due diligence", "employment law", "legal analysis",
    ],
    "engineering": [
        "autocad", "solidworks", "catia", "ansys",
        "mechanical design", "civil engineering", "structural analysis",
        "electrical engineering", "plc programming", "scada",
        "quality control", "quality assurance", "iso standards",
        "project engineering", "product design", "3d modeling",
        "manufacturing", "cad", "cam",
    ],
    "content_communication": [
        "content writing", "technical writing", "copywriting",
        "blogging", "journalism", "editing", "proofreading",
        "public relations", "communications", "social media",
        "video production", "photography", "podcasting",
        "script writing", "storytelling", "documentation",
    ],
    "soft_skills": [
        "communication", "leadership", "teamwork", "problem solving",
        "critical thinking", "time management", "adaptability",
        "creativity", "attention to detail", "analytical skills",
        "decision making", "conflict resolution", "negotiation",
        "presentation skills", "interpersonal skills",
        "emotional intelligence", "customer service", "multitasking",
    ],
    "languages_spoken": [
        "english", "hindi", "gujarati", "marathi", "tamil",
        "telugu", "french", "german", "spanish", "arabic",
        "mandarin", "japanese",
    ],
    "certifications": [
        "aws certified", "google cloud certified", "azure certified",
        "pmp", "scrum master", "cissp", "comptia", "ccna",
        "cpa", "cfa", "ca", "cma", "six sigma green belt",
        "six sigma black belt", "itil", "prince2",
    ],
}

ALL_SKILLS: list = [s for skills in SKILLS_DB.values() for s in skills]
SKILL_CATEGORY: dict = {
    skill: cat for cat, skills in SKILLS_DB.items() for skill in skills
}


# ─────────────────────────────────────────────────────────────
# ROLE TITLE → SKILLS MAP
# Fixes the "Backend Developer → 0 score" problem
# ─────────────────────────────────────────────────────────────
ROLE_SKILL_MAP: dict = {
    "backend developer":       ["python", "django", "fastapi", "flask", "postgresql", "mysql", "redis", "rest api", "docker", "git", "sql"],
    "backend engineer":        ["python", "django", "fastapi", "flask", "postgresql", "mysql", "redis", "rest api", "docker", "git"],
    "frontend developer":      ["javascript", "typescript", "react", "vue", "angular", "tailwind", "webpack", "git", "css", "html"],
    "frontend engineer":       ["javascript", "typescript", "react", "vue", "angular", "tailwind", "webpack", "git"],
    "full stack developer":    ["python", "javascript", "react", "django", "fastapi", "postgresql", "docker", "git", "rest api"],
    "full stack engineer":     ["python", "javascript", "react", "django", "fastapi", "postgresql", "docker", "git"],
    "software developer":      ["python", "java", "javascript", "git", "sql", "rest api", "problem solving", "agile"],
    "software engineer":       ["python", "java", "javascript", "git", "sql", "rest api", "problem solving", "agile"],
    "python developer":        ["python", "django", "fastapi", "flask", "postgresql", "rest api", "git", "docker"],
    "java developer":          ["java", "spring", "postgresql", "mysql", "rest api", "git", "maven"],
    "react developer":         ["react", "javascript", "typescript", "tailwind", "git", "rest api"],
    "node developer":          ["javascript", "typescript", "nodejs", "express", "mongodb", "rest api", "git"],
    "django developer":        ["python", "django", "postgresql", "rest api", "git", "docker"],
    "php developer":           ["php", "laravel", "mysql", "javascript", "rest api", "git"],
    "flutter developer":       ["flutter", "dart", "android", "ios", "rest api", "git"],
    "android developer":       ["android", "kotlin", "java", "git", "rest api", "sqlite"],
    "ios developer":           ["ios", "swift", "xcode", "git", "rest api"],
    "mobile developer":        ["react native", "flutter", "android", "ios", "swift", "kotlin", "git"],
    "devops engineer":         ["docker", "kubernetes", "terraform", "ansible", "aws", "ci/cd", "jenkins", "linux", "prometheus"],
    "cloud engineer":          ["aws", "gcp", "azure", "docker", "kubernetes", "terraform", "linux"],
    "data scientist":          ["python", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "machine learning", "statistics", "sql"],
    "data analyst":            ["python", "sql", "excel", "power bi", "tableau", "data analysis", "statistics", "pandas"],
    "data engineer":           ["python", "sql", "spark", "airflow", "postgresql", "aws", "docker"],
    "ml engineer":             ["python", "tensorflow", "pytorch", "scikit-learn", "mlflow", "docker", "aws"],
    "machine learning engineer": ["python", "tensorflow", "pytorch", "scikit-learn", "mlflow", "docker"],
    "ai engineer":             ["python", "tensorflow", "pytorch", "llm", "langchain", "bert", "nlp", "docker"],
    "security engineer":       ["penetration testing", "owasp", "ssl", "encryption", "oauth", "jwt", "linux", "cybersecurity"],
    "qa engineer":             ["selenium", "pytest", "jest", "automation testing", "manual testing", "jira", "agile"],
    "product manager":         ["product management", "agile", "scrum", "stakeholder management", "jira", "user research", "kpi"],
    "ui ux designer":          ["figma", "adobe xd", "wireframing", "prototyping", "user research", "ui design", "ux design"],
    "ui/ux designer":          ["figma", "adobe xd", "wireframing", "prototyping", "user research", "ui design"],
    "graphic designer":        ["photoshop", "illustrator", "canva", "figma", "graphic design", "adobe creative suite"],
    "business analyst":        ["business analysis", "sql", "excel", "stakeholder management", "agile", "jira", "requirements gathering"],
    "project manager":         ["project management", "agile", "scrum", "stakeholder management", "risk management", "jira"],
    "marketing manager":       ["digital marketing", "seo", "content marketing", "google analytics", "social media marketing", "crm"],
    "digital marketer":        ["digital marketing", "seo", "sem", "google analytics", "social media marketing", "email marketing"],
    "content writer":          ["content writing", "copywriting", "seo", "research", "editing", "communication"],
    "sales executive":         ["sales", "crm", "lead generation", "negotiation", "communication", "b2b sales"],
    "hr manager":              ["recruitment", "hr management", "employee relations", "performance management", "payroll", "labor law"],
    "hr executive":            ["recruitment", "talent acquisition", "onboarding", "hr management", "communication"],
    "financial analyst":       ["financial analysis", "financial modeling", "excel", "sql", "accounting", "budgeting"],
    "accountant":              ["accounting", "tally", "excel", "taxation", "bookkeeping", "financial reporting", "auditing"],
    "operations manager":      ["operations management", "process improvement", "supply chain", "logistics", "kpi", "leadership"],
    "teacher":                 ["teaching", "lesson planning", "curriculum development", "classroom management", "communication"],
    "trainer":                 ["training", "instructional design", "presentation skills", "communication", "coaching"],
    "nurse":                   ["patient care", "nursing", "first aid", "bls", "ehr", "anatomy", "clinical research"],
    "doctor":                  ["diagnosis", "treatment planning", "patient care", "clinical research", "pharmacology"],
    "pharmacist":              ["pharmacy", "pharmacology", "patient care", "medical records"],
    "mechanical engineer":     ["autocad", "solidworks", "mechanical design", "quality control", "manufacturing", "cad"],
    "civil engineer":          ["autocad", "civil engineering", "structural analysis", "cad", "project management"],
    "electrical engineer":     ["electrical engineering", "autocad", "plc programming", "scada"],
}

ROLE_ALIASES: dict = {
    "developer":       "software developer",
    "engineer":        "software engineer",
    "sde":             "software developer",
    "swe":             "software engineer",
    "be developer":    "backend developer",
    "fe developer":    "frontend developer",
    "be engineer":     "backend developer",
    "fe engineer":     "frontend developer",
    "web developer":   "full stack developer",
    "web designer":    "ui/ux designer",
    "data analyst":    "data analyst",
}

ROLE_PROFILES: dict = {
    "Backend Developer":    ["python", "django", "fastapi", "postgresql", "redis", "rest api", "docker"],
    "Frontend Developer":   ["javascript", "typescript", "react", "vue", "tailwind", "css", "webpack"],
    "Full Stack Developer": ["react", "python", "postgresql", "docker", "rest api", "git", "aws"],
    "Data Scientist":       ["python", "pandas", "scikit-learn", "tensorflow", "statistics", "machine learning"],
    "DevOps Engineer":      ["docker", "kubernetes", "terraform", "aws", "ci/cd", "jenkins", "linux"],
    "Mobile Developer":     ["react native", "flutter", "android", "ios", "swift", "kotlin"],
    "ML Engineer":          ["tensorflow", "pytorch", "scikit-learn", "mlflow", "docker", "bert"],
    "Business Analyst":     ["business analysis", "sql", "excel", "stakeholder management", "agile"],
    "Marketing Manager":    ["digital marketing", "seo", "google analytics", "social media marketing", "crm"],
    "HR Professional":      ["recruitment", "hr management", "employee relations", "performance management"],
    "Financial Analyst":    ["financial analysis", "excel", "accounting", "budgeting", "sql"],
    "Product Manager":      ["product management", "agile", "scrum", "user research", "stakeholder management"],
    "UI/UX Designer":       ["figma", "wireframing", "user research", "prototyping", "ui design"],
    "Content Writer":       ["content writing", "seo", "copywriting", "research", "editing"],
    "Security Engineer":    ["penetration testing", "owasp", "encryption", "oauth", "cybersecurity"],
}

ROLE_EMBEDDINGS: dict = {
    role: embedder.encode(" ".join(skills), convert_to_numpy=True)
    for role, skills in ROLE_PROFILES.items()
}


# ─────────────────────────────────────────────────────────────
# PDF EXTRACTION
# ─────────────────────────────────────────────────────────────
def extract_text(file_path: str) -> str:
    text = _extract_pdfplumber(file_path)
    if len(text.strip()) < 50:
        logger.info(f"pdfplumber got {len(text.strip())} chars — switching to OCR")
        text = _extract_ocr(file_path)
    return text.strip()


def _extract_pdfplumber(file_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception as exc:
        logger.error(f"pdfplumber error: {exc}")
    return text


def _extract_ocr(file_path: str) -> str:
    text = ""
    try:
        import os
        kwargs = {"dpi": 300}
        poppler_path = r"C:\poppler\Library\bin"
        if os.path.exists(poppler_path):
            kwargs["poppler_path"] = poppler_path
        else:
            logger.warning("Poppler not found — OCR skipped")
            return ""

        images = convert_from_path(file_path, **kwargs)
        for i, image in enumerate(images):
            processed = _preprocess_for_ocr(image)
            page_text = pytesseract.image_to_string(
                processed,
                lang="eng",
                config="--psm 6 --oem 3"
            )
            text += page_text + "\n"
            logger.info(f"OCR page {i+1}: {len(page_text)} chars")
    except Exception as exc:
        logger.error(f"OCR failed: {exc}")
    return text


def _preprocess_for_ocr(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    image = ImageEnhance.Contrast(image).enhance(2.0)
    image = image.filter(ImageFilter.SHARPEN)
    w, h  = image.size
    image = image.resize((int(w * 1.2), int(h * 1.2)), Image.LANCZOS)
    return image

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
    "experience":     r"\b(experience|work history|employment|career|internship|worked at)\b",
    "skills":         r"\b(skills|technical skills|core competencies|expertise)\b",
    "projects":       r"\b(projects|portfolio|personal projects|open.?source)\b",
    "certifications": r"\b(certifications?|certificates?|courses?|training|achievements?)\b",
    "summary":        r"\b(summary|objective|profile|about me|overview)\b",
}

def detect_sections(text: str) -> dict:
    lower = text.lower()
    return {sec: bool(re.search(pat, lower)) for sec, pat in SECTION_PATTERNS.items()}


# ─────────────────────────────────────────────────────────────
# SMART JD PARSER — THE KEY FIX
# ─────────────────────────────────────────────────────────────
def extract_job_skills(job_description: str) -> list:
    """
    Extracts required skills from JD using 4 passes:
    1. Direct skill matching
    2. Role title expansion (Backend Developer → python, fastapi...)
    3. Alias matching (developer → software developer)
    4. Semantic fallback (when JD has no obvious keywords)
    """
    jd_lower  = job_description.lower().strip()
    found: set = set()

    # Pass 1 — direct skill match
    for skill in ALL_SKILLS:
        if re.search(r"\b" + re.escape(skill) + r"\b", jd_lower):
            found.add(skill)

    # Pass 2 — role title → skill expansion
    for role_title, skills in ROLE_SKILL_MAP.items():
        if re.search(r"\b" + re.escape(role_title) + r"\b", jd_lower):
            logger.info(f"Role title '{role_title}' found in JD → expanding to {len(skills)} skills")
            found.update(skills)

    # Pass 3 — alias matching
    for alias, canonical in ROLE_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", jd_lower):
            if canonical in ROLE_SKILL_MAP:
                found.update(ROLE_SKILL_MAP[canonical])

    # Pass 4 — semantic fallback when too few skills found
    if len(found) < 3 and len(jd_lower.split()) > 8:
        logger.info(f"Only {len(found)} skills found directly — running semantic fallback")
        found.update(_semantic_jd_expansion(jd_lower))

    result = sorted(found)
    logger.info(f"Total JD skills extracted: {len(result)}")
    return result


def _semantic_jd_expansion(jd_text: str) -> list:
    """Embed JD text and find top semantically similar skills."""
    try:
        jd_emb      = embedder.encode(jd_text, convert_to_numpy=True)
        skill_embs  = embedder.encode(ALL_SKILLS, convert_to_numpy=True)
        jd_norm     = jd_emb / (np.linalg.norm(jd_emb) + 1e-9)
        skill_norms = skill_embs / (np.linalg.norm(skill_embs, axis=1, keepdims=True) + 1e-9)
        sims        = skill_norms @ jd_norm
        top_idx     = np.argsort(sims)[::-1][:15]
        return [ALL_SKILLS[i] for i in top_idx if sims[i] > 0.25]
    except Exception as e:
        logger.error(f"Semantic JD expansion failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# RESUME SKILL EXTRACTION
# ─────────────────────────────────────────────────────────────
def extract_skills_advanced(text: str) -> list:
    text_lower = text.lower()
    doc = nlp(text_lower)
    found: set = set()

    for skill in ALL_SKILLS:
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
            found.add(skill)

    for chunk in doc.noun_chunks:
        for skill in ALL_SKILLS:
            if skill in chunk.text:
                found.add(skill)

    for token in doc:
        if token.text in ALL_SKILLS:
            found.add(token.text)

    return sorted(found)


# ─────────────────────────────────────────────────────────────
# SKILL CLUSTERING
# ─────────────────────────────────────────────────────────────
def cluster_skills(skills: list, n_clusters: int = 4) -> dict:
    if len(skills) < 2:
        return {"General": skills}

    n    = min(n_clusters, len(skills))
    embs = embedder.encode(skills, convert_to_numpy=True)
    embs = normalize(embs)

    km     = KMeans(n_clusters=n, random_state=42, n_init=10)
    labels = km.fit_predict(embs)

    raw: dict = {}
    for skill, label in zip(skills, labels.tolist()):
        raw.setdefault(label, []).append(skill)

    named: dict = {}
    for lid, cluster_list in raw.items():
        centroid  = km.cluster_centers_[lid]
        cembs     = embedder.encode(cluster_list, convert_to_numpy=True)
        distances = np.linalg.norm(cembs - centroid, axis=1)
        rep       = cluster_list[int(np.argmin(distances))]
        named[rep] = cluster_list

    return named


# ─────────────────────────────────────────────────────────────
# ROLE DETECTION
# ─────────────────────────────────────────────────────────────
def detect_role(resume_skills: list) -> dict:
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
    doc  = nlp(text.lower())
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
    threshold: float = 0.55,
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
    if 400 <= wc <= 1200:
        length_score = 20
    elif wc < 150:
        length_score = 5
    elif wc < 400:
        length_score = 12
    else:
        length_score = 14

    contact_bonus = 10 if re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text) else 0
    total = max(0, min(100, int(keyword_score + section_score + length_score + contact_bonus)))

    return {
        "ats_score":     total,
        "keyword_score": int(keyword_score),
        "section_score": section_score,
        "length_score":  length_score,
        "word_count":    wc,
        "ats_level":     "Good" if total >= 70 else "Average" if total >= 45 else "Poor",
    }


# ─────────────────────────────────────────────────────────────
# SCORING
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

    cats            = {SKILL_CATEGORY.get(s) for s in matched if SKILL_CATEGORY.get(s)}
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
                + (f" with good skills in {top_matched}" if top_matched else "")
                + f". Well-suited for a {role} role — build portfolio projects to stand out.")
        elif score >= 50:
            return (f"Decent match for {role}. "
                + (f"Strengthen {top_missing} " if top_missing else "")
                + "and add 1–2 real projects to improve your chances.")
        else:
            return (f"Low match. Key gaps: {top_missing or 'multiple areas'}. "
                "Targeted upskilling before applying is recommended.")
    else:
        if score >= 80:
            return (f"Strong experienced candidate"
                + (f" with good coverage in {top_matched}" if top_matched else "")
                + f". Well-aligned with {role} requirements.")
        elif score >= 55:
            return (f"Moderate match for {role}. "
                + (f"Bridging gaps in {top_missing} " if top_missing else "")
                + "would meaningfully improve your profile.")
        else:
            return (f"Significant skill gaps for {role}: {top_missing or 'review the JD'}. "
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
        suggestions.append(f"Build 2–3 portfolio projects tailored to {role} and host them on GitHub")
    else:
        suggestions.append("Quantify achievements — e.g. 'reduced load time by 40%' or 'managed team of 8'")

    if not sections.get("summary"):
        suggestions.append("Add a professional summary — 3 sentences about who you are and key strengths")
    if not sections.get("projects") and fresher:
        suggestions.append("Add a Projects section with tech stack, your role, and measurable outcome")
    if not sections.get("certifications"):
        suggestions.append(f"Add relevant certifications for {role} — they are highly valued by recruiters")

    for skill in missing[:5]:
        cat = SKILL_CATEGORY.get(skill, "general")
        if cat == "cloud_devops":
            suggestions.append(f"Get hands-on with {skill} — free-tier cloud accounts are a great start")
        elif cat == "data_ai":
            suggestions.append(f"Build a practical {skill} project and publish on Kaggle or GitHub")
        elif cat == "testing":
            suggestions.append(f"Add {skill} tests to your existing projects to show quality focus")
        elif cat in ("business_skills", "soft_skills"):
            suggestions.append(f"Highlight {skill} with a specific example from your experience")
        else:
            suggestions.append(f"Learn {skill} and add it to your resume with a project example")

    if matched and not missing:
        suggestions.append("All key skills matched — tailor your resume summary to the exact job title")

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
    job_skills    = extract_job_skills(jd_lower)       # ← smart parser
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