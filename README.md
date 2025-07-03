# Fake Job Detection using AI and NLP

This project investigates the detection of fake job postings using machine learning and natural language processing (NLP) techniques. It integrates structured metadata, linguistic cues, and credibility signals to build interpretable models capable of identifying fraudulent job listings.

## Research Focus

The primary goal is to explore AI and NLP in fake job detection and how different feature types—such as linguistic, structural, metadata, and credibility-based indicators—affect the performance and explainability of fake job detection models. The study also examines the impact of class imbalance and representation bias on prediction outcomes.

---

## Datasets

The analysis combines two complementary datasets:

1. **EMSCAD** (Employment Scam Aegean Dataset)  
   - Public dataset with job ads collected globally (2012–2014)
   - Includes structured and unstructured job fields

2. **Nigerian Job Dataset**  
   - Locally curated dataset from Nigerian job boards (2023)
   - More recent fraud tactics and context-specific postings

Both datasets are harmonized into a common schema before training.

---

##  Preprocessing Pipeline

The preprocessing pipeline performs the following steps:

### Column Harmonization
- Unified schema across EMSCAD and Nigerian records
- Company names extracted from unstructured fields using a BERT-based NER model

### Missing Value Handling
- Categorical fields imputed with `"Missing"` placeholder (e.g., industry, function)
- Class-wise strategies for fields like `review`, `company_size`, `age`
- Salary extracted from both structured ranges and unstructured text

###  Type Conversion
- Binary fields mapped to 0/1
- Salary fields parsed and split into `min_salary`, `max_salary`
- `age`, `company_size_avg`, and other derived numerics added

###  Feature Augmentation
- `tagged_gdp` assigned using GDP per capita: 2023 for Nigerian records, 2012–2014 average for others
- `salary_avg` normalized using GDP to form `expected_salary_gdp`
- Role-based expected salary assigned using fuzzy match to a benchmark table
- Final features include credibility signals and text-based structure

---

##  Feature Engineering

Features are grouped into:

- **Meta Features**: location, employment type, department, etc.
- **Linguistic Features**: text length, punctuation count, sentiment, readability
- **Structural Features**: presence of requirements, description formatting, etc.
- **Credibility Features**: company size, website presence, review score, year founded
- **TF-IDF Features**: extracted from job descriptions and requirements

---

##  Modeling

Models evaluated include:

- Logistic Regression
- Random Forest
- XGBoost

Both baseline and resampled datasets were tested using SMOTE to address class imbalance.

---

## Evaluation

Models were evaluated using:

- **Precision, Recall, F1-score** (with special focus on fraud recall)
- **ROC-AUC Score**

---

##  Key Findings

- Structural and linguistic features greatly improved fraud recall
- Company and salary credibility signals contributed strongly to fraud prediction
- Overrepresented categorical values introduced bias, mitigated through feature selection and resampling
- XGBoost and Random Forest showed strong performance even on imbalanced data when engineered features were used


