import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import textstat
import spacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate

df = pd.read_csv(/content/Final_Job_Postings.csv")

# Combine text fields
df['text_all'] = df[['title', 'company_profile', 'description', 'requirements', 'benefits']].fillna('').agg(' '.join, axis=1)

#Linguistic Features
df['readability'] = df['text_all'].apply(lambda x: textstat.flesch_kincaid_grade(x) if len(x.strip()) > 0 else 0)
df['sentiment_polarity'] = df['text_all'].apply(lambda x: TextBlob(x).sentiment.polarity if len(x.strip()) > 0 else 0)
df['sentiment_subjectivity'] = df['text_all'].apply(lambda x: TextBlob(x).sentiment.subjectivity if len(x.strip()) > 0 else 0)

def capitalization_ratio(text):
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return 0
    caps = [w for w in words if w.isupper() and len(w) > 1]
    return len(caps) / len(words)

def punctuation_count(text):
    return len(re.findall(r'[.!?,;:]', text))

def special_char_count(text):
    return len(re.findall(r'[^a-zA-Z0-9\s]', text))

df['capitalization_ratio'] = df['text_all'].apply(capitalization_ratio)
df['punctuation_count'] = df['text_all'].apply(punctuation_count)
df['special_char_count'] = df['text_all'].apply(special_char_count)

#Bin company size 
def bin_company_size(val):
    if pd.isnull(val) or str(val).lower() == 'missing':
        return 'Missing'
    val = str(val)
    if '-' in val:
        try:
            low, high = val.split('-')
            avg = (int(low.replace('+', '').strip()) + int(high.replace('+', '').strip())) // 2
        except Exception:
            return 'Missing'
    else:
        try:
            avg = int(val.replace('+', '').strip())
        except Exception:
            return 'Missing'
    if avg < 50:
        return '<50'
    elif avg <= 200:
        return '51–200'
    elif avg <= 500:
        return '201–500'
    elif avg <= 1000:
        return '501–1000'
    elif avg <= 5000:
        return '1001–5000'
    else:
        return '5001+'

if 'company_size' in df.columns:
    df['company_size_bin'] = df['company_size'].apply(bin_company_size)

#Structural Features
def clean_wordcount(text):
    if pd.isnull(text) or text.strip().lower() == 'missing':
        return 0
    text = re.sub('<.*?>', '', str(text))
    text = re.sub(r'\s+', ' ', text)
    return len(text.strip().split())

df['has_company_profile'] = df['company_profile'].apply(lambda x: int(pd.notnull(x) and str(x).strip() and str(x).strip().lower() != 'missing'))
df['has_description'] = df['description'].apply(lambda x: int(pd.notnull(x) and str(x).strip() and str(x).strip().lower() != 'missing'))
df['has_requirements'] = df['requirements'].apply(lambda x: int(pd.notnull(x) and str(x).strip() and str(x).strip().lower() != 'missing'))
df['has_benefits'] = df['benefits'].apply(lambda x: int(pd.notnull(x) and str(x).strip() and str(x).strip().lower() != 'missing'))
df['has_salary'] = df.apply(lambda r: int(r.get('min_salary', -1) not in [-1, -2] and r.get('max_salary', -1) not in [-1, -2]), axis=1)
df['has_location'] = df['location'].apply(lambda x: int(pd.notnull(x) and str(x).strip() and str(x).strip().lower() != 'missing'))

df['company_profile_wordcount'] = df['company_profile'].apply(clean_wordcount)
df['description_wordcount'] = df['description'].apply(clean_wordcount)
df['requirements_wordcount'] = df['requirements'].apply(clean_wordcount)
df['benefits_wordcount'] = df['benefits'].apply(clean_wordcount)

meta_fields = ['industry', 'function', 'employment_type', 'required_experience', 'required_education', 'location']
def missing_count(row):
    count = 0
    for field in meta_fields:
        val = row.get(field, None)
        if pd.isnull(val) or str(val).strip().lower() == 'missing':
            count += 1
    if row.get('has_salary', 1) == 0:
        count += 1
    return count

df['missing_metadata_fields'] = df.apply(missing_count, axis=1)

def compute_salary_avg(row):
    if row['min_salary'] >= 0 and row['max_salary'] >= 0:
        return (row['min_salary'] + row['max_salary']) / 2
    elif row['min_salary'] >= 0:
        return row['min_salary']
    elif row['max_salary'] >= 0:
        return row['max_salary']
    else:
        return np.nan

def compute_salary_deviation(row):
    if row['min_salary'] >= 0 and row['max_salary'] >= 0:
        return row['max_salary'] - row['min_salary']
    else:
        return np.nan

df['salary_avg'] = df.apply(compute_salary_avg, axis=1)
df['salary_deviation'] = df.apply(compute_salary_deviation, axis=1)

# Impute median for missing values
salary_avg_median = df['salary_avg'].median()
salary_deviation_median = df['salary_deviation'].median()

df['salary_avg'] = df['salary_avg'].fillna(salary_avg_median)
df['salary_deviation'] = df['salary_deviation'].fillna(salary_deviation_median)

# Feature selection 
categorical = [
    'required_experience', 'required_education', 'company', 'company_size_bin'
]
numeric_binary = [
    'telecommuting', 'has_company_logo', 'has_questions', 'has_website',
    'year', 'Review', 'salary_avg', 'salary_deviation'
]
linguistic_features = [
    'readability', 'sentiment_polarity', 'sentiment_subjectivity',
    'capitalization_ratio', 'punctuation_count', 'special_char_count'
]
structural_features = [
    'has_company_profile', 'has_description', 'has_requirements', 'has_benefits',
    'has_salary', 'has_location', 'company_profile_wordcount', 'description_wordcount',
    'requirements_wordcount', 'benefits_wordcount', 'missing_metadata_fields'
]

features = categorical + numeric_binary + linguistic_features + structural_features
features = [col for col in features if col in df.columns]

# Encode categoricals
for col in categorical:
    df[col] = df[col].astype(str).fillna('Missing')
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Fill missing values
for col in numeric_binary + linguistic_features + structural_features:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

X = df[features]
y = df['fraudulent'].astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

#Resampling
resampling_strategy = "smote"  # "smote", "undersample", or "none"

if resampling_strategy == "smote":
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print("SMOTE applied. Resampled training shape:", X_train_res.shape)
else:
    X_train_res, y_train_res = X_train, y_train
    print("No resampling applied.")

# ML Models
models = {
    'Logistic Regression': (LogisticRegression(max_iter=300, n_jobs=-1), {'C': [0.01, 0.1, 1, 10]}),
    'Random Forest': (RandomForestClassifier(random_state=42), {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_leaf': [1, 5, 10]}),
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), {'n_estimators': [100, 200], 'max_depth': [3, 6, 10], 'learning_rate': [0.01, 0.1, 0.2]})
}


scoring = ['recall', 'f1', 'roc_auc']
top_n = 10

scoring = ['recall', 'f1', 'roc_auc']

for name, (model, params) in models.items():
    print(f"\\n--- {name} ---")
    grid = GridSearchCV(model, params, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train_res, y_train_res)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:,1]
    print("Best Parameters:", grid.best_params_)
    print("Classification Report:\\n", classification_report(y_test, y_pred, digits=3))
    print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    # Top 10 Feature Importances/Scores
    print(f"\nTop {top_n} Feature Importances/Coefs:")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        for idx in indices:
            print(f"{X.columns[idx]}: {importances[idx]:.4f}")
    elif hasattr(model, "coef_"):
        coefs = model.coef_[0]
        indices = np.argsort(np.abs(coefs))[::-1][:top_n]
        for idx in indices:
            print(f"{X.columns[idx]}: {coefs[idx]:.4f}")
    
    #Cross-Validation Scores
    cv_results = cross_validate(model, X_train_res, y_train_res, scoring=scoring, cv=5)
    print(f"\n5-fold CV scores ({name}):")
    print("Recall scores per fold: ", cv_results['test_recall'])
    print("F1 scores per fold: ", cv_results['test_f1'])
    print("ROC-AUC scores per fold: ", cv_results['test_roc_auc'])
    print("Mean Recall: {:.3f} ± {:.3f}".format(np.mean(cv_results['test_recall']), np.std(cv_results['test_recall'])))
    print("Mean F1: {:.3f} ± {:.3f}".format(np.mean(cv_results['test_f1']), np.std(cv_results['test_f1'])))
    print("Mean ROC-AUC: {:.3f} ± {:.3f}".format(np.mean(cv_results['test_roc_auc']), np.std(cv_results['test_roc_auc'])))


