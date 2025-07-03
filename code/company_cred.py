import pandas as pd

# Load the integrated dataset
combined_df = pd.read_csv("/content/Integrated_Job_Postings.csv") #Integrated Dataset

# Load the company-level indicators
augmented = pd.read_csv("/content/Company_Cred_Data.csv") #Company cred dataset (manually collected)
augmented = augmented.drop(columns=['Unnamed: 0'], errors='ignore').rename(columns={
    'extracted_company': 'company',
    'Review': 'review',
    'employees': 'company_size'
})

emscad_only = combined_df[combined_df['company'].notna()].copy()
nigerian_only = combined_df[combined_df['company'].isna()].copy()

# Merge company-level metadata (review, website, year, size) into EMSCAD
emscad_augmented = emscad_only.merge(
    augmented[['company', 'fraudulent', 'review', 'has_website', 'year', 'company_size']],
    on=['company', 'fraudulent'], how='left'
)

# Combine the updated EMSCAD portion with the original Nigerian records
company_feature = pd.concat([emscad_augmented, nigerian_only], ignore_index=True)

# Save the output
company_feature.to_csv("/content/company_feature_augmented.csv", index=False)

