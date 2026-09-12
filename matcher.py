"""
matcher.py
Core logic for the Smart Shortlisting Engine.

This file contains reusable FUNCTIONS ONLY (no code runs automatically when
this file is imported by app.py). To test this file directly, run:
    python matcher.py
at the bottom of the file, under the "if __name__ == '__main__':" block.
"""

import os
import re
import pdfplumber
from docx import Document
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# STEP 1: Extract raw text from a resume/JD file, whatever format it's in
# ---------------------------------------------------------------------------

def extract_text(filepath):
    """Reads a .pdf, .docx, .html/.htm, or .txt file and returns its text."""
    text = ""
    lower = filepath.lower()

    if lower.endswith(".pdf"):
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"

    elif lower.endswith(".docx"):
        doc = Document(filepath)
        for para in doc.paragraphs:
            text += para.text + "\n"
        # also grab text inside tables, some resumes use tables for layout
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    text += cell.text + " "

    elif lower.endswith((".html", ".htm")):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
            text = soup.get_text(separator="\n")

    elif lower.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    return text


def load_resumes_from_folder(folder_path):
    """Returns a dict {filename: extracted_text} for every supported file in a folder."""
    resumes = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".pdf", ".docx", ".html", ".htm", ".txt")):
            filepath = os.path.join(folder_path, filename)
            resumes[filename] = extract_text(filepath)
    return resumes


# ---------------------------------------------------------------------------
# STEP 2: Figure out which skills the JD actually cares about
# ---------------------------------------------------------------------------

SKILL_VOCABULARY = [
    "javascript", "typescript", "python", "java", "c++", "c#", "node.js", "nodejs",
    "express", "express.js", "react", "react.js", "angular", "vue", "next.js",
    "redux", "html", "css", "sass", "tailwind", "bootstrap",
    "mongodb", "postgresql", "postgres", "mysql", "sqlite", "redis", "firebase",
    "sql", "nosql", "graphql", "rest", "rest api", "restful",
    "docker", "kubernetes", "aws", "azure", "gcp", "ci/cd", "github actions",
    "jenkins", "git", "github", "gitlab", "linux", "bash",
    "jest", "mocha", "cypress", "selenium", "unit testing", "tdd",
    "agile", "scrum", "jira", "figma", "postman", "swagger",
    "microservices", "websocket", "socket.io", "jwt", "oauth",
    "machine learning", "data structures", "algorithms", "system design",
    "leetcode", "hackerrank", "tryhackme", "hackthebox", "codeforces",
    "codechef", "kaggle", "topcoder", "ctf", "capture the flag",
    "competitive programming", "open source", "codewars",
]


def extract_dynamic_terms(jd_text, company_name=None):
    exclude = {
        "THE", "AND", "FOR", "YOU", "ARE", "OUR", "WITH", "WILL", "THIS",
        "API", "CI", "CD", "JD",
    }
    if company_name:
        exclude.add(company_name.upper())

    camel_case = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", jd_text)
    acronyms = re.findall(r"\b[A-Z]{3,5}\b", jd_text)

    found = set()
    for term in camel_case + acronyms:
        if term.upper() not in exclude:
            found.add(term.lower())
    return sorted(found)


def extract_required_skills(jd_text):
    jd_lower = jd_text.lower()
    found = []
    for skill in SKILL_VOCABULARY:
        if skill in jd_lower:
            found.append(skill)

    company_match = re.search(r"Company:\s*(.+)", jd_text)
    company_name = company_match.group(1).split()[0] if company_match else None

    for term in extract_dynamic_terms(jd_text, company_name=company_name):
        if term not in found:
            found.append(term)

    return found


# ---------------------------------------------------------------------------
# STEP 3: Keyword matching -- does the resume mention the JD's required skills?
# ---------------------------------------------------------------------------

def keyword_match(resume_text, required_skills, fuzzy_threshold=85):
    resume_lower = resume_text.lower()
    resume_words = re.findall(r"[a-zA-Z0-9\.\+#]+", resume_lower)
    resume_joined = " ".join(resume_words)

    found, missing = [], []
    for skill in required_skills:
        if skill in resume_lower:
            found.append(skill)
            continue
        match_score = fuzz.partial_ratio(skill, resume_joined)
        if match_score >= fuzzy_threshold:
            found.append(skill)
        else:
            missing.append(skill)

    score = len(found) / len(required_skills) if required_skills else 0.0
    return found, missing, score


# ---------------------------------------------------------------------------
# STEP 3.5: Extra criteria beyond tech skills
# ---------------------------------------------------------------------------

DEGREE_KEYWORDS = [
    "computer science", "information technology", "software engineering",
    "electronics", "electrical engineering", "data science", "artificial intelligence",
    "bachelor", "master", "b.tech", "m.tech", "b.e.", "m.e.", "bsc", "msc", "b.sc", "m.sc",
]


def extract_education_requirement(jd_text):
    jd_lower = jd_text.lower()
    found = [kw for kw in DEGREE_KEYWORDS if kw in jd_lower]
    return found


def check_education_match(resume_text, required_degrees):
    if not required_degrees:
        return [], 1.0
    resume_lower = resume_text.lower()
    matched = [d for d in required_degrees if d in resume_lower]
    score = 1.0 if matched else 0.0
    return matched, score


def extract_experience_requirement(jd_text):
    match = re.search(r"(\d+)\+?\s*year", jd_text.lower())
    return int(match.group(1)) if match else None


MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"

def estimate_experience_years(resume_text):
    pattern = rf"({MONTHS}[a-z]*\.?\s+\d{{4}})\s*[-–—]\s*(present|{MONTHS}[a-z]*\.?\s+\d{{4}})"
    matches = re.findall(pattern, resume_text.lower())

    total_months = 0
    for start_str, end_str in matches:
        start_match = re.search(r"\d{4}", start_str)
        if not start_match:
            continue
        start_year = int(start_match.group())

        if "present" in end_str:
            end_year = 2026
            end_month = 9
        else:
            end_match = re.search(r"\d{4}", end_str)
            if not end_match:
                continue
            end_year = int(end_match.group())
            end_month = 6

        start_month = 6
        total_months += max(0, (end_year - start_year) * 12 + (end_month - start_month))

    return round(total_months / 12, 1)


def check_experience_match(resume_text, min_years_required):
    if min_years_required is None:
        return None, 1.0
    candidate_years = estimate_experience_years(resume_text)
    score = 1.0 if candidate_years >= min_years_required else candidate_years / min_years_required
    return candidate_years, min(1.0, score)


def extract_cgpa_requirement(jd_text):
    match = re.search(r"(?:cgpa|gpa)[^\d]{0,10}(\d\.\d+)", jd_text.lower())
    return float(match.group(1)) if match else None


def check_cgpa_match(resume_text, min_cgpa_required):
    if min_cgpa_required is None:
        return None, 1.0
    match = re.search(r"(?:cgpa|gpa)[^\d]{0,10}(\d\.\d+)", resume_text.lower())
    if not match:
        return None, 0.5
    candidate_cgpa = float(match.group(1))
    score = 1.0 if candidate_cgpa >= min_cgpa_required else candidate_cgpa / min_cgpa_required
    return candidate_cgpa, min(1.0, score)


def extract_location_requirement(jd_text):
    jd_lower = jd_text.lower()
    if "remote" in jd_lower:
        return "remote"
    match = re.search(r"(?:based in|location:)\s*([a-zA-Z\s]+?)(?:[\.\n,]|$)", jd_lower)
    return match.group(1).strip() if match else None


def check_location_match(resume_text, required_location):
    if not required_location or required_location == "remote":
        return True, 1.0
    resume_lower = resume_text.lower()
    matched = required_location in resume_lower
    return matched, (1.0 if matched else 0.3)


def extract_employment_type(jd_text):
    jd_lower = jd_text.lower()
    for word in ["internship", "intern", "full-time", "part-time", "contract"]:
        if word in jd_lower:
            return word
    return None


CERT_VOCABULARY = [
    "aws certified", "azure certified", "google cloud certified", "gcp certified",
    "pmp", "scrum master", "certified scrum master", "csm",
    "comptia", "ccna", "cissp", "oracle certified", "microsoft certified",
    "certified kubernetes", "cka", "ckad",
]


def extract_certification_requirement(jd_text):
    jd_lower = jd_text.lower()
    return [c for c in CERT_VOCABULARY if c in jd_lower]


def check_certification_match(resume_text, required_certs):
    if not required_certs:
        return [], 1.0
    resume_lower = resume_text.lower()
    matched = [c for c in required_certs if c in resume_lower]
    score = len(matched) / len(required_certs)
    return matched, score


def extract_extracurricular_mentions(resume_text):
    resume_lower = resume_text.lower()
    keywords = ["club", "society", "volunteer", "hackathon", "captain", "organizer", "extracurricular"]
    return [k for k in keywords if k in resume_lower]


# ---------------------------------------------------------------------------
# STEP 4: Semantic matching (LSA)
# ---------------------------------------------------------------------------

def semantic_scores_for_batch(jd_text, resume_texts):
    documents = [jd_text] + resume_texts

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(documents)

    n_components = min(100, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
    n_components = max(n_components, 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    latent_matrix = svd.fit_transform(tfidf_matrix)

    jd_vector = latent_matrix[0].reshape(1, -1)
    resume_vectors = latent_matrix[1:]

    similarities = cosine_similarity(jd_vector, resume_vectors)[0]
    return [max(0.0, min(1.0, float(s))) for s in similarities]


# ---------------------------------------------------------------------------
# STEP 5: Combine both scores, rank everyone, and explain the top 3
# ---------------------------------------------------------------------------

def rank_resumes(
    jd_text, resumes,
    keyword_weight=0.4, semantic_weight=0.4, criteria_weight=0.2,
):
    required_skills = extract_required_skills(jd_text)

    required_degrees = extract_education_requirement(jd_text)
    min_years = extract_experience_requirement(jd_text)
    min_cgpa = extract_cgpa_requirement(jd_text)
    required_location = extract_location_requirement(jd_text)
    employment_type = extract_employment_type(jd_text)
    required_certs = extract_certification_requirement(jd_text)

    filenames = list(resumes.keys())
    texts = [resumes[f] for f in filenames]

    raw_semantic_scores = semantic_scores_for_batch(jd_text, texts)

    results = []
    for filename, text, sem_score in zip(filenames, texts, raw_semantic_scores):
        found, missing, kw_score = keyword_match(text, required_skills)

        matched_degrees, edu_score = check_education_match(text, required_degrees)
        candidate_years, exp_score = check_experience_match(text, min_years)
        candidate_cgpa, cgpa_score = check_cgpa_match(text, min_cgpa)
        location_matched, loc_score = check_location_match(text, required_location)
        matched_certs, cert_score = check_certification_match(text, required_certs)
        extracurriculars = extract_extracurricular_mentions(text)

        active_scores = []
        if required_degrees: active_scores.append(edu_score)
        if min_years is not None: active_scores.append(exp_score)
        if min_cgpa is not None: active_scores.append(cgpa_score)
        if required_location: active_scores.append(loc_score)
        if required_certs: active_scores.append(cert_score)
        criteria_score = sum(active_scores) / len(active_scores) if active_scores else 1.0

        results.append({
            "filename": filename,
            "keyword_score": kw_score,
            "semantic_score_raw": sem_score,
            "criteria_score": criteria_score,
            "found": found,
            "missing": missing,
            "criteria": {
                "required_degrees": required_degrees,
                "matched_degrees": matched_degrees,
                "min_years": min_years,
                "candidate_years": candidate_years,
                "min_cgpa": min_cgpa,
                "candidate_cgpa": candidate_cgpa,
                "required_location": required_location,
                "location_matched": location_matched,
                "employment_type": employment_type,
                "required_certs": required_certs,
                "matched_certs": matched_certs,
                "extracurriculars": extracurriculars,
            },
        })

    sem_values = [r["semantic_score_raw"] for r in results]
    sem_min, sem_max = min(sem_values), max(sem_values)
    sem_range = (sem_max - sem_min) or 1e-6

    for r in results:
        sem_norm = (r["semantic_score_raw"] - sem_min) / sem_range
        r["semantic_score"] = round(sem_norm, 3)
        r["final_score"] = round(
            keyword_weight * r["keyword_score"]
            + semantic_weight * sem_norm
            + criteria_weight * r["criteria_score"],
            3,
        )

    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results, required_skills


def explain_candidate(result, required_skills):
    name = result["filename"]
    found = result["found"]
    missing = result["missing"]
    c = result["criteria"]

    lines = [f"**{name}** -- final score: {result['final_score']:.2f}"]
    if found:
        lines.append(f"- Matched skills: {', '.join(found)}")
    if missing:
        lines.append(f"- Missing skills: {', '.join(missing)}")
    lines.append(
        f"- Semantic fit to JD: {result['semantic_score']:.2f} "
        f"(how closely the resume's overall experience aligns with the JD's intent, "
        f"beyond just keyword overlap)"
    )

    if c["required_degrees"]:
        status = f"matched ({', '.join(c['matched_degrees'])})" if c["matched_degrees"] else "no matching degree found"
        lines.append(f"- Education: JD wants {', '.join(c['required_degrees'])} -- {status}")
    if c["min_years"] is not None:
        lines.append(f"- Experience: JD wants {c['min_years']}+ years -- resume shows ~{c['candidate_years']} years")
    if c["min_cgpa"] is not None:
        cgpa_str = c["candidate_cgpa"] if c["candidate_cgpa"] is not None else "not found in resume"
        lines.append(f"- CGPA: JD wants {c['min_cgpa']}+ -- resume shows {cgpa_str}")
    if c["required_location"]:
        status = "matches" if c["location_matched"] else "no match found (may still be open to relocating)"
        lines.append(f"- Location: JD wants {c['required_location']} -- {status}")
    if c["employment_type"]:
        lines.append(f"- Note: JD is for a {c['employment_type']} position")
    if c["required_certs"]:
        status = f"matched ({', '.join(c['matched_certs'])})" if c["matched_certs"] else "no matching certifications found"
        lines.append(f"- Certifications: JD wants {', '.join(c['required_certs'])} -- {status}")
    if c["extracurriculars"]:
        lines.append(f"- Extracurricular signals noted: {', '.join(c['extracurriculars'])} (informational only)")

    return "\n".join(lines)


if __name__ == "__main__":
    JD_PATH = "Sample_JD.pdf"
    RESUME_FOLDER = "resumes"

    if not os.path.exists(JD_PATH):
        print(f"Couldn't find '{JD_PATH}'. Update JD_PATH in matcher.py to match your file.")
    else:
        jd_text = extract_text(JD_PATH)
        resumes = load_resumes_from_folder(RESUME_FOLDER)
        print(f"Loaded JD ({len(jd_text)} chars) and {len(resumes)} resumes.\n")

        results, required_skills = rank_resumes(jd_text, resumes)
        print(f"Required skills detected in JD: {required_skills}\n")

        print("=== FULL RANKING ===")
        for i, r in enumerate(results, start=1):
            print(f"{i}. {r['filename']} — score: {r['final_score']:.2f}")

        print("\n=== TOP 3 EXPLANATIONS ===")
        for r in results[:3]:
            print(explain_candidate(r, required_skills))
            print()
