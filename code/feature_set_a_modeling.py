import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

df = pd.read_csv("/content/Final_Job_Postings.csv")

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

#Metafeature Set
categorical = [
    'required_experience', 'Country','department','industry','function'
    'required_education', 'company', 'company_size_bin'
]
numeric_binary = [
    'telecommuting', 'has_company_logo', 'has_questions', 'has_website',
    'year', 'Review', 'min_salary', 'max_salary'
]
categorical = [col for col in categorical if col in df.columns]
numeric_binary = [col for col in numeric_binary if col in df.columns]
features = categorical + numeric_binary

for col in categorical:
    if col in df.columns:
        df[col] = df[col].astype(str).fillna('Missing')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

for col in numeric_binary:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

X = df[features]
y = df['fraudulent'].astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

# Resampling
resampling_strategy = "smote"  # or "undersample" or "smote"
if resampling_strategy == "smote":
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print("SMOTE applied. Resampled training shape:", X_train_res.shape)
elif resampling_strategy == "undersample":
    rus = RandomUnderSampler(random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    print("Random undersampling applied. Resampled training shape:", X_train_res.shape)
else:
    X_train_res, y_train_res = X_train, y_train
    print("No resampling applied.")

# Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=300, n_jobs=-1),
    'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42)
}

from sklearn.model_selection import cross_validate

scoring = ['recall', 'f1', 'roc_auc']
top_n = 10

for name, model in models.items():
    print(f"\n--- {name} ({resampling_strategy}) ---")
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except:
        y_prob = y_pred
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC-AUC: ", roc_auc_score(y_test, y_prob))
    
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
    
    # Cross-Validation Scores
    cv_results = cross_validate(model, X_train_res, y_train_res, scoring=scoring, cv=5)
    print(f"\n5-fold CV scores ({name}):")
    print("Recall scores per fold: ", cv_results['test_recall'])
    print("F1 scores per fold: ", cv_results['test_f1'])
    print("ROC-AUC scores per fold: ", cv_results['test_roc_auc'])
    print("Mean Recall: {:.3f} ± {:.3f}".format(np.mean(cv_results['test_recall']), np.std(cv_results['test_recall'])))
    print("Mean F1: {:.3f} ± {:.3f}".format(np.mean(cv_results['test_f1']), np.std(cv_results['test_f1'])))
    print("Mean ROC-AUC: {:.3f} ± {:.3f}".format(np.mean(cv_results['test_roc_auc']), np.std(cv_results['test_roc_auc'])))
