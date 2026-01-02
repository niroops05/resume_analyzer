import streamlit as st
import pdfplumber
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="AI Resume ATS Analyzer",
    layout="centered"
)

# ================== GLOBAL CSS ==================
st.markdown("""
<style>

/* Global page */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    background-color: #f6f8fa;
}

/* Main container */
.main {
    background-color: #ffffff;
    padding: 2rem;
    border-radius: 12px;
}

/* Headings */
h1 {
    color: #1f2937;
    font-weight: 700;
}
h2, h3 {
    color: #374151;
}

/* Button */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #1e40af;
}

/* File uploader */
.stFileUploader {
    border: 2px dashed #c7d2fe;
    border-radius: 10px;
    padding: 15px;
    background-color: #f8fafc;
}

/* Text area */
textarea {
    border-radius: 8px !important;
    border: 1px solid #cbd5e1 !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-size: 28px;
    font-weight: 700;
    color: #2563eb;
}

/* Alerts */
.stAlert {
    border-radius: 8px;
}

/* Remove Streamlit footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ================== NLP SETUP ==================
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

# ================== FUNCTIONS ==================

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(words)


def calculate_similarity(resume, jd):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume, jd])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]


def ats_engine(resume_text, jd_text):
    resume_words = set(resume_text.split())
    jd_words = set(jd_text.split())

    matched = resume_words.intersection(jd_words)
    missing = jd_words - resume_words

    keyword_score = len(matched) / len(jd_words) if jd_words else 0
    similarity_score = calculate_similarity(resume_text, jd_text)

    final_score = (0.5 * keyword_score + 0.3 * similarity_score) * 100
    return round(final_score, 2), matched, missing, similarity_score


def categorize_keywords(keywords):
    technical = {
        "python","java","sql","machine","learning","ai","ml","nlp",
        "data","analysis","algorithm","model","api"
    }
    tools = {
        "docker","aws","cloud","git","linux","kubernetes",
        "streamlit","tensorflow","pytorch"
    }

    categories = {
        "Technical Skills": [],
        "Tools & Platforms": [],
        "Other Keywords": []
    }

    for k in keywords:
        if k in technical:
            categories["Technical Skills"].append(k)
        elif k in tools:
            categories["Tools & Platforms"].append(k)
        else:
            categories["Other Keywords"].append(k)

    return categories


def render_skill_cards(title, keywords, bg_color):
    st.markdown(f"### {title}")
    cols = st.columns(4)
    for i, word in enumerate(list(keywords)[:16]):
        cols[i % 4].markdown(
            f"""
            <div style="
                background:{bg_color};
                color:#1f2937;
                padding:10px;
                margin:6px;
                border-radius:20px;
                text-align:center;
                font-size:14px;
                font-weight:500;
                box-shadow:0 2px 6px rgba(0,0,0,0.08);
            ">
                {word}
            </div>
            """,
            unsafe_allow_html=True
        )


def generate_suggestions(missing):
    suggestions = []
    for skill in list(missing)[:8]:
        suggestions.append(
            f"Add a resume bullet demonstrating hands-on experience with **{skill}**, aligned to the job description."
        )
    return suggestions

# ================== UI ==================

st.title("AI Resume Analyzer & ATS Scorer")
st.subheader("Evaluate resume compatibility with a job description")

st.markdown("---")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description", height=220)

if st.button("Analyze Resume"):
    if resume_file is None or not job_description.strip():
        st.error("Please upload a resume and paste a job description.")
    else:
        with st.spinner("Running ATS analysis..."):
            resume_text = extract_text_from_pdf(resume_file)
            resume_clean = clean_text(resume_text)
            jd_clean = clean_text(job_description)

            score, matched, missing, similarity = ats_engine(
                resume_clean, jd_clean
            )

        st.success("Analysis Complete")

        # ATS Score
        st.metric("Final ATS Score", f"{score} / 100")

        # Score Breakdown
        st.markdown("### Score Breakdown")
        col1, col2 = st.columns(2)
        col1.metric(
            "Keyword Match %",
            f"{round(len(matched)/(len(matched)+len(missing))*100,1) if matched or missing else 0}%"
        )
        col2.metric(
            "JD Similarity %",
            f"{round(similarity*100,1)}%"
        )

        st.markdown("---")

        # Skill Cards
        render_skill_cards("Matched Skills", matched, "#5eff97")
        render_skill_cards("Missing Skills", missing, "#fd5656")

        st.markdown("---")

        # Category Analysis
        st.markdown("### Skill Category Analysis")
        matched_cat = categorize_keywords(matched)
        missing_cat = categorize_keywords(missing)

        for category in matched_cat:
            st.markdown(f"**{category}**")
            st.write("Matched:", matched_cat[category] if matched_cat[category] else "—")
            st.write("Missing:", missing_cat[category] if missing_cat[category] else "—")
            st.markdown("---")

        # Suggestions
        st.markdown("### Resume Improvement Suggestions")
        for s in generate_suggestions(missing):
            st.info(s)
# ------------------- FOOTER -------------------
st.markdown(
    """
    ---
    **• Streamlit • NLP • TF-IDF • Scikit-learn • NLTK • PDFPlumber**  
    Developed by ***Naga Niroop***
    """
)