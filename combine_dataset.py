import pandas as pd
import numpy as np
# Load datasets
emscad = pd.read_csv("/mnt/data/EMSCAD_with_Company.csv") #updated EMSCAD With company
nigerian = pd.read_csv("/mnt/data/CompiledjobListNigeria.csv")
# Standardize Nigerian column names
nigerian = nigerian.rename(columns={
    'job_title': 'title',
    'company_desc': 'company_profile',
    'job_desc': 'description',
    'job_requirement': 'requirements',
    'salary': 'salary_range',
    'company_name': 'company',
    'label': 'fraudulent'
})
# label to binary
nigerian['fraudulent'] = nigerian['fraudulent'].map({'t': 1, 'f': 0})
# Final schema (aligned to EMSCAD structure)
common_cols = [
    'title', 'company', 'location', 'department', 'salary_range',
    'company_profile', 'description', 'requirements', 'benefits',
    'telecommuting', 'has_company_logo', 'has_questions', 'employment_type',
    'required_experience', 'required_education', 'industry', 'function',
    'fraudulent'
]
# Add missing columns to Nigerian dataset
for col in common_cols:
    if col not in nigerian.columns:
        nigerian[col] = np.nan
# Add missing columns to EMSCAD if any
for col in common_cols:
    if col not in emscad.columns:
        emscad[col] = np.nan
# Reorder both datasets
emscad = emscad[common_cols]
nigerian = nigerian[common_cols]
# Concatenate datasets
combined_df = pd.concat([emscad, nigerian], ignore_index=True)
combined_df.to_csv("/content/Integrated_Job_Postings.csv", index=False)

print(f"Integrated dataset shape: {combined_df.shape}")
