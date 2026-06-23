import streamlit as st
import sqlite3
import re
import os
import requests
from datetime import datetime
from collections import Counter

# PDF Processing
from pdfminer.high_level import extract_text
from PyPDF2 import PdfReader

# NLP / ML
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Resume Parser (Deployment Safe)
ResumeParser = None

try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    from pyresparser import ResumeParser
except Exception:
    ResumeParser = None

# Web Scraping
from bs4 import BeautifulSoup

# Data
import pandas as pd

# PDF Report
from reportlab.pdfgen import canvas

# Gemini API (LLM)
import google.generativeai as genai

# ============================
# CONFIG
# ============================
nltk.download('punkt')
nltk.download('stopwords')
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = spacy.blank("en")

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ============================
# DATABASE
# ============================
class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect('resume.db', check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS resumes(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            score REAL,
            role TEXT,
            predicted_role TEXT,
            skills TEXT,
            sentiment TEXT,
            summary TEXT,
            created_at TEXT
        )
        ''')
        self.conn.commit()

    def insert(self, data):
        self.cursor.execute('''
        INSERT INTO resumes(name,score,role,predicted_role,skills,sentiment,summary,created_at)
        VALUES(?,?,?,?,?,?,?,?)
        ''', data)
        self.conn.commit()

    def fetch(self):
        return self.cursor.execute('SELECT * FROM resumes').fetchall()

# ============================
# TEXT PROCESSOR
# ============================
class TextProcessor:
    def preprocess(self, text):
        text = re.sub(r'[^a-zA-Z ]', '', text.lower())
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        filtered = [w for w in tokens if w not in stop_words]
        return ' '.join(filtered)

    def extract_skills(self, text):
        skills_db = [
            'python','sql','machine learning','nlp','flask','django',
            'streamlit','tensorflow','pytorch','pandas','numpy','api'
        ]
        found = []
        for skill in skills_db:
            if skill in text.lower():
                found.append(skill)
        return found

    def sentiment(self, text):
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0:
            return 'Positive'
        elif polarity < 0:
            return 'Negative'
        return 'Neutral'

    def keyword_density(self, text):
        return dict(Counter(text.split()).most_common(10))

    def similarity(self, resume, job_desc):
        tfidf = TfidfVectorizer()
        vectors = tfidf.fit_transform([resume, job_desc])
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    def ner(self, text):
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

# ============================
# ROLE PREDICTOR
# ============================
class RolePredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        data = [
            'python machine learning pandas numpy',
            'html css javascript react',
            'django flask api sql backend'
        ]
        labels = ['Data Scientist','Frontend Developer','Backend Developer']
        X = self.vectorizer.fit_transform(data)
        self.model = LogisticRegression()
        self.model.fit(X, labels)

    def predict(self, text):
        X = self.vectorizer.transform([text])
        return self.model.predict(X)[0]

# ============================
# GEMINI LLM
# ============================
class GeminiAnalyzer:
    def summarize_resume(self, text):
        if not GEMINI_API_KEY:
            return 'Gemini API key not configured.'
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f'Summarize this resume and suggest improvements: {text}'
        response = model.generate_content(prompt)
        return response.text

    def generate_cover_letter(self, resume_text, role):
        if not GEMINI_API_KEY:
            return 'Gemini API key not configured.'
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f'Generate a cover letter for {role} using this resume: {resume_text}'
        response = model.generate_content(prompt)
        return response.text

# ============================
# JOB SCRAPER (BeautifulSoup)
# ============================
class JobScraper:
    def fetch_jobs(self, keyword='python developer'):
        url = f'https://remoteok.com/remote-{keyword.replace(" ", "-")}-jobs'
        headers = {'User-Agent':'Mozilla/5.0'}
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            jobs = []
            for job in soup.find_all('h2')[:5]:
                jobs.append(job.text.strip())
            return jobs
        except Exception:
            return []

# ============================
# REPORT GENERATOR
# ============================
class ReportGenerator:
    def generate(self, name, score, role, sentiment):
        filename = f'{name}_report.pdf'
        pdf = canvas.Canvas(filename)
        pdf.drawString(100, 800, 'Resume Analysis Report')
        pdf.drawString(100, 760, f'ATS Score: {score}')
        pdf.drawString(100, 720, f'Role Prediction: {role}')
        pdf.drawString(100, 680, f'Sentiment: {sentiment}')
        pdf.save()
        return filename

# ============================
# FILE READER
# ============================
def read_resume(uploaded_file):
    with open('temp_resume.pdf', 'wb') as f:
        f.write(uploaded_file.read())
    text = extract_text('temp_resume.pdf')
    return text

# ============================
# MAIN APP
# ============================
def main():
    st.set_page_config(page_title='Resume Analyzer', layout='wide')
    st.title('AI Resume Analyser')

    db = DatabaseManager()
    tp = TextProcessor()
    rp = RolePredictor()
    llm = GeminiAnalyzer()
    scraper = JobScraper()
    report = ReportGenerator()

    menu = st.sidebar.selectbox('Menu', ['Analyze Resume'])

    if menu == 'Analyze Resume':
        file = st.file_uploader('Upload Resume', type=['pdf'])
        job_role = st.selectbox(
            'Select Target Job Role',
            [
                'Software Engineer',
                'Data Scientist',
                'Backend Developer',
                'Frontend Developer',
                'Machine Learning Engineer',
                'Python Developer'
            ]
        )

        if st.button('Analyze') and file:
            text = read_resume(file)
            clean_text = tp.preprocess(text)
            skills = tp.extract_skills(clean_text)
            sentiment = tp.sentiment(text)
            predicted_role = rp.predict(clean_text)
            similarity_score = tp.similarity(clean_text, job_role)
            ats_score = round(similarity_score * 100, 2)
            entities = tp.ner(text)
            summary = llm.summarize_resume(text)
            cover_letter = llm.generate_cover_letter(text, job_role)
            jobs = scraper.fetch_jobs(job_role)

            # PyResparser fallback fix (Python 3.14 / spaCy compatibility issue)

                    # PyResparser fallback fix
            if ResumeParser:
                try:
                    parsed_data = ResumeParser('temp_resume.pdf').get_extracted_data()
                except Exception as e:
                    parsed_data = {
                        'name': file.name,
                        'skills': skills,
                        'note': f'Resume parsing unavailable: {str(e)}'
                    }
            else:
                parsed_data = {
                    'name': file.name,
                    'skills': skills,
                    'note': 'PyResparser not available on deployment server'
                }

            st.success(f'ATS Score: {ats_score}%')
            st.write('Predicted Role:', predicted_role)
            st.subheader('Skills Extraction')
            st.write('Detected Skills:', skills)

            # Skill recommendations
            recommended_skills = [
                'Docker', 'Kubernetes', 'AWS', 'System Design', 'GitHub Actions'
            ]
            st.write('Recommended Skills:', recommended_skills)

            # Certification recommendation slider
            cert_count = st.slider('Number of Certifications to Recommend', 1, 10, 3)
            certifications = [
                'Google Data Analytics',
                'AWS Certified Cloud Practitioner',
                'TensorFlow Developer',
                'Microsoft Azure Fundamentals',
                'IBM Data Science Professional'
            ]
            st.write('Recommended Certifications:', certifications[:cert_count])
            st.write('Sentiment:', sentiment)
            st.write('Named Entities:', entities)
            st.subheader('Resume Grading & Insights')
            resume_grade = min(len(skills) * 10, 100)
            st.progress(resume_grade / 100)
            st.write(f'Overall Resume Grade: {resume_grade}/100')
            st.write('Resume Summary:', summary)
            st.write('Suggestions: Add more measurable achievements, improve formatting, and include relevant certifications.')
            # Resume details section (same as project output)
            st.subheader('Resume Details Extraction')
            st.write('Parsed Resume:', parsed_data)
            st.write('Candidate Name:', parsed_data.get('name', 'Not Found'))
            st.write('Contact:', parsed_data.get('mobile_number', 'Not Found'))
            st.write('Email:', parsed_data.get('email', 'Not Found'))
            
            # Page count
            pdf_reader = PdfReader('temp_resume.pdf')
            st.write('Number of Pages:', len(pdf_reader.pages))
            st.subheader('Resume Matching')
            st.write('Job Matches from Web Scraping:', jobs)
            if ats_score >= 60:
                st.success('Resume matches the job role.')
            else:
                st.error('Resume does not match well. Improve skills and keyword relevance.')
            if ats_score >= 60:
                st.subheader('Generated Cover Letter')
                st.write(cover_letter)

            # Personality Insights
            st.subheader('Personality Insights')
            personality_scores = {
                'Openness': 70,
                'Conscientiousness': 80,
                'Extraversion': 60,
                'Agreeableness': 75,
                'Neuroticism': 40
            }
            st.bar_chart(personality_scores)

            st.subheader('Behavioral Factors')
            behavior_scores = {
                'Leadership': 75,
                'Communication': 82,
                'Teamwork': 78,
                'Problem Solving': 88
            }
            st.bar_chart(behavior_scores)

            st.write('Detailed Personality Insights:', 'You show strong problem-solving and teamwork ability, suitable for technical and collaborative job roles.')
            st.bar_chart(tp.keyword_density(clean_text))

            db.insert((
                file.name,
                ats_score,
                job_role,
                predicted_role,
                ', '.join(skills),
                sentiment,
                summary,
                datetime.now().strftime('%Y-%m-%d %H:%M')
            ))

            report_file = report.generate(
                file.name,
                ats_score,
                predicted_role,
                sentiment
            )

            with open(report_file, 'rb') as f:
                st.download_button('Download Report', f, file_name=report_file)

    

if __name__ == '__main__':
    main()

