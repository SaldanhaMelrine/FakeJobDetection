import pandas as pd
import numpy as np
import re
import html
from bs4 import BeautifulSoup
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the final dataset
df = pd.read_csv("/content/company_feature_augmented.csv")

# Binary field conversion
binary_fields = ['telecommuting', 'has_company_logo', 'has_questions', 'has_website']
for col in binary_fields:
    if df[col].dtype == 'object':
        df[col] = df[col].map({'t': 1, 'f': 0})
    df[col] = df[col].astype('Int64')

df['fraudulent'] = df['fraudulent'].astype(int)

# Type casting
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['review'] = pd.to_numeric(df['review'], errors='coerce')

# Imputation: has_website (class-wise mode)
for label in [0, 1]:
    mode_val = df[df['fraudulent'] == label]['has_website'].mode(dropna=True)
    if not mode_val.empty:
        df.loc[(df['fraudulent'] == label) & (df['has_website'].isna()), 'has_website'] = mode_val[0]

# Imputation: review (class-wise median)
for label in [0, 1]:
    median_val = df[df['fraudulent'] == label]['review'].median()
    df.loc[(df['fraudulent'] == label) & (df['review'].isna()), 'review'] = median_val

# Map company_size to numeric midpoint
size_map = {
    '1-10': 5, '2-10 employees': 6, '11-50': 30, '51-200': 125, '201-500': 350,
    '501-1000': 750, '1001-5000': 3000, '5001-10000': 7500, '10000+': 15000,
    '1,001 to 5,000 employees': 3000, '5,001 to 10,000 employees': 7500,
    '10,001+ employees': 15000
}
df['company_size_avg'] = df['company_size'].map(size_map)

# Impute company_size_avg using class-wise mode
for label in [0, 1]:
    mode_val = df[df['fraudulent'] == label]['company_size_avg'].mode(dropna=True)
    if not mode_val.empty:
        df.loc[(df['fraudulent'] == label) & (df['company_size_avg'].isna()), 'company_size_avg'] = mode_val[0]

# Replace missing text fields with placeholder
text_fields = ['title', 'description', 'company_profile', 'requirements', 'benefits']
for col in text_fields:
    df[col] = df[col].fillna("Missing")

# Fill categorical fields 
cat_placeholder_fields = [
    'employment_type', 'industry', 'function', 'department',
    'location', 'required_education', 'required_experience', 'company'
]
for col in cat_placeholder_fields:
    df[col] = df[col].fillna("Missing")

df['salary_range'] = df['salary_range'].fillna("Missing")

# Function to clean and normalize text
def clean_and_normalize_text(text):
    if pd.isna(text):
        return text, 0
    text = html.unescape(text)  # Decode HTML entities
    text = BeautifulSoup(text, "html.parser").get_text()  # Strip HTML tags
    special_count = count_special_chars(text)  # Count special chars
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
    text = text.lower()  # Lowercase
    text = re.sub(r'\s+', ' ', text)  # Whitespace normalization
    return text.strip(), special_count

# Stemming and stopword removal
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def remove_stopwords_and_stem(text):
    tokens = text.split()
    filtered = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered)

# Apply to each text field
for field in text_fields:
    clean_col = f"{field}_clean"
    count_col = f"{field}_special_char_count"
    
    # Apply structural cleaning and count
    df[[clean_col, count_col]] = df[field].apply(lambda x: pd.Series(clean_and_normalize_text(x)))
    
    # Apply stopword removal and stemming
    df[clean_col] = df[clean_col].apply(remove_stopwords_and_stem)

# Save the fully preprocessed dataset
df.to_csv("Preprocessed_Job_Postings.csv", index=False)
