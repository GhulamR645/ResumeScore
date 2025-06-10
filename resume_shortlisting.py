
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO
import pdfminer.high_level
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import Dataset
import os
import json

# Download necessary NLTK data 
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 1. Text Preprocessing Functions
def clean_text(text):
    """Cleans the input text by removing punctuation, converting to lowercase, and removing numbers."""
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text

def tokenize_text(text):
    """Tokenizes the input text into individual words."""
    tokenizer = RegexpTokenizer(r'\w+')  # Tokenize alphanumeric words
    return tokenizer.tokenize(text)

def remove_stopwords(tokens):
    """Removes common English stopwords from the list of tokens."""
    stop_words = set(stopwords.words('english'))
    #Custom stop words
    custom_stop_words = ['the', 'and', 'a', 'an', 'in', 'of', 'to', 'for', 'with', 'on', 'is', 'are', 'be']
    return [token for token in tokens if token not in stop_words and token not in custom_stop_words]

def preprocess_text(text):
    """Combines all text preprocessing steps."""
    cleaned_text = clean_text(text)
    tokens = tokenize_text(cleaned_text)
    tokens = remove_stopwords(tokens)
    return tokens

# 2. Custom Scoring Mechanism
def calculate_resume_score(jd_tokens, resume_tokens):
    """Calculates a custom score (out of 100) for a resume based on the job description."""
    jd_set = set(jd_tokens)
    resume_set = set(resume_tokens)

    common_words = jd_set.intersection(resume_set)
    num_common_words = len(common_words)

    if not jd_set or not resume_set:
        return 0  # Handle empty sets

    # Calculate score based on the proportion of common words to the average length of JD and resume
    average_length = (len(jd_set) + len(resume_set)) / 2
    epsilon = 1e-9  # Small value to prevent division by zero
    score = (num_common_words / (average_length + epsilon)) * 100
    return score

# 3. Top-K Resume Ranking
def rank_resumes(jd_tokens, resumes):  # Removed k here
    """Ranks resumes based on their calculated scores and returns the ranked resumes."""
    resume_scores = []
    for resume_text in resumes:
        resume_tokens = preprocess_text(resume_text)
        score = calculate_resume_score(jd_tokens, resume_tokens)
        resume_scores.append(score)

    # Sort resumes by score in descending order
    ranked_resumes = sorted(zip(resumes, resume_scores), key=lambda x: x[1], reverse=True)
    return ranked_resumes

# 4. Candidate Summary
def generate_candidate_summary(jd_tokens, resume_text):
    """Generates a summary of a candidate's resume, highlighting matched skills and relevant terms."""
    resume_tokens = preprocess_text(resume_text)
    jd_set = set(jd_tokens)
    resume_set = set(resume_tokens)
    matched_skills = jd_set.intersection(resume_set)

    summary = f"Matched Skills: {', '.join(matched_skills)}\n"
    summary += f"Relevant Terms: {', '.join(resume_set)}\n"

    return summary

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    try:
        text = pdfminer.high_level.extract_text(file)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

# 5. User Interface (Streamlit)
def main():
    st.title("Resume Shortlisting System")

    # Define the path to the directory containing the JSON files
    dataset_path = "./resume-score-details"  # Adjust if the directory is different

    # Load the dataset from the local directory
    try:
        # Check if the directory exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Directory not found: {dataset_path}")

        # Load the JSON files and extract the data
        data = []
        json_files = [f for f in os.listdir(dataset_path) if f.endswith(".json")]
        if not json_files:
            raise FileNotFoundError("No JSON files found in the directory.")

        for json_file in json_files:
            file_path = os.path.join(dataset_path, json_file)
            with open(file_path, 'r', encoding='utf-8') as f:  # Specify encoding
                try:
                    json_data = json.load(f)
                    jd_text = json_data['input']['job_description']
                    resume_text = json_data['input']['resume']
                   #personal info 
                    personal_info = json_data['output'].get('personal_info', {}) 
                    data.append({'jd_text': jd_text, 'resume_text': resume_text, 'personal_info': personal_info})
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {json_file}: {e}")
                except KeyError as e:
                    print(f"Missing key in file {json_file}: {e}")

        # Create a datasets Dataset from the extracted data
        dataset = Dataset.from_pandas(pd.DataFrame(data))
        st.success("Dataset loaded successfully from JSON files!")
        print("Dataset loaded successfully from JSON files!")

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()  # Stop if dataset loading failed
        print(f"Error loading dataset: {e}")  # Jupyter Notebook error message
        return

    # Job Description Input
    jd_source = st.radio("Job Description Source:", ("Upload PDF", "Write Text"))
    jd_text = ""

    if jd_source == "Upload PDF":
        uploaded_file = st.file_uploader("Upload Job Description PDF", type="pdf")
        if uploaded_file is not None:
            jd_text = extract_text_from_pdf(uploaded_file)
    else:
        jd_text = st.text_area("Enter Job Description Text")

    if jd_text:
        jd_tokens = preprocess_text(jd_text)

        # Rank Resumes
        ranked_resumes = rank_resumes(jd_tokens, dataset['resume_text'])

        # K Value Input
        num_resumes_to_use = len(dataset)
        k = st.slider("Select Top K Resumes", min_value=1, max_value=num_resumes_to_use, value=min(5, num_resumes_to_use))

        # Display Results
        st.header("Top Ranked Resumes")
        for i, (resume_text, score) in enumerate(ranked_resumes[:k]):
            st.subheader(f"Rank {i+1}: Score = {score:.2f}")
            summary = generate_candidate_summary(jd_tokens, resume_text)
            st.write(summary)

            # Find the corresponding personal_info
            resume_index = dataset['resume_text'].index(resume_text)
            personal_info = dataset[resume_index]['personal_info']

            # Display Contact Information
            st.write("**Contact Information:**")
            st.write(f"Name: {personal_info.get('name', 'N/A')}")
            st.write(f"Email: {personal_info.get('email', 'N/A')}")
            st.write(f"Phone: {personal_info.get('phone', 'N/A')}")
            st.write("---")
    else:
        st.info("Please provide a job description.")

if __name__ == "__main__":
    main()






