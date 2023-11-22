import PyPDF2
import re
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import nltk
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
import os
import random
import fnmatch
from flask import send_from_directory


from langdetect import detect

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load the skill dataset (Technology Skills.xlsx)
skill_dataset = pd.read_excel("Technology Skills.xlsx")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Create an "uploads" directory to store uploaded files
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in range(len(pdf_reader.pages)):  
        text += pdf_reader.pages[page].extract_text()
    return text

# Function to preprocess text
def preprocess_text(text, remove_stopwords=True, use_lemmatization=True):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    if remove_stopwords:
        stop_words = set(nltk.corpus.stopwords.words('english'))
        words = nltk.word_tokenize(text)
        words = [word for word in words if word not in stop_words]

    # Lemmatize words using spaCy
    if use_lemmatization:
        doc = nlp(" ".join(words))
        words = [token.lemma_ for token in doc]

    text = " ".join(words)

    return text

# Function to extract skills from a CV
def extract_skills_from_cv(cv_text, skill_dataset):
    matched_skills = set()  # Use a set to avoid duplicates
    for skill in skill_dataset['Example']:
        if skill.lower() in cv_text.lower():
            matched_skills.add(skill)
    return list(matched_skills)

# Function to calculate cosine similarity
def calculate_cosine_similarity(job_description_text, cv_text):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([job_description_text, cv_text])

    job_description_tfidf = tfidf_matrix[0]
    cv_tfidf = tfidf_matrix[1]

    cosine_similarity_score = cosine_similarity(job_description_tfidf, cv_tfidf)
    similarity_percentage = cosine_similarity_score[0][0] * 100
    return similarity_percentage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/view_cv/<candidate_name>')
def view_cv(candidate_name):
    cv_file_path = os.path.join("uploads", f"uploaded_{candidate_name}_*.pdf")

    # Mengambil path file CV yang sesuai
    matching_files = [file for file in os.listdir("uploads") if fnmatch.fnmatch(file, f"uploaded_{candidate_name}_*.pdf")]
    if matching_files:
        cv_file_path = os.path.join("uploads", matching_files[0])
        return send_from_directory(os.path.dirname(cv_file_path), os.path.basename(cv_file_path))
    else:
        return render_template('error.html', error_message='CV tidak ditemukan')

@app.route('/process', methods=['POST'])
def process():
    # Access the job description PDF file
    job_description_file = request.files['job_description_file']
    job_description_text = ""
    if job_description_file:
        unique_filename = f"uploaded_{random.randint(1, 1000000)}.pdf"
        job_description_file.save(os.path.join("uploads", unique_filename))
        job_description_text = extract_text_from_pdf(os.path.join("uploads", unique_filename))
        job_description_text = preprocess_text(job_description_text)

        # Language validation based on user selection
        user_language_selection = request.form['language_selection']
        detected_language = detect(job_description_text)
        if (user_language_selection == 'Indonesia' and detected_language == 'en') or \
           (user_language_selection == 'English' and detected_language == 'id'):
            error_message = "Language selection does not match the language of the uploaded job description."
            return render_template('error.html', error_message=error_message)

    num_candidates_to_select = int(request.form['num_candidates_to_select'])
    candidates = []

    for i in range(num_candidates_to_select):
        candidate_name = request.form[f'candidate_name_{i + 1}']
        cv_file = request.files[f'cv_file_{i + 1}']
        cv_text = ""
        if cv_file:
            unique_filename = f"uploaded_{candidate_name}_{random.randint(1, 1000000)}.pdf"
            cv_file.save(os.path.join("uploads", unique_filename))
            cv_text = extract_text_from_pdf(os.path.join("uploads", unique_filename))
            cv_text = preprocess_text(cv_text)

            # Language validation for each candidate
            detected_language = detect(cv_text)
            if (user_language_selection == 'Indonesia' and detected_language == 'en') or \
               (user_language_selection == 'English' and detected_language == 'id'):
                error_message = f"Language selection does not match the language of the CV for candidate {candidate_name}."
                return render_template('error.html', error_message=error_message)

        candidates.append((candidate_name, cv_text))

    selected_candidates = []

    for candidate_name, cv_text in candidates:
        matched_skills = extract_skills_from_cv(cv_text, skill_dataset)
        similarity_percentage = calculate_cosine_similarity(job_description_text, cv_text)
        selected_candidates.append((candidate_name, similarity_percentage, matched_skills))

    selected_candidates.sort(key=lambda x: x[1], reverse=True)

    return render_template('results.html', selected_candidates=selected_candidates)

if __name__ == '__main__':
    app.run(debug=True)
