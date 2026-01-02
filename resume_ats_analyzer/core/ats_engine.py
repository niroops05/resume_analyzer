from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(resume, jd):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume, jd])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

def calculate_ats_score(resume_text, jd_text):
    resume_words = set(resume_text.split())
    jd_words = set(jd_text.split())

    matched = resume_words.intersection(jd_words)
    missing = jd_words - resume_words

    keyword_score = len(matched) / len(jd_words) if jd_words else 0
    similarity_score = calculate_similarity(resume_text, jd_text)

    final_score = (0.5 * keyword_score + 0.3 * similarity_score) * 100

    return round(final_score, 2), matched, missing
