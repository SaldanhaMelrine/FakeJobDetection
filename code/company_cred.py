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

has_company = combined_df[combined_df['company'].notna()].copy()
no_company = combined_df[combined_df['company'].isna()].copy() 

# Merge company-level metadata (review, website, year, size)
has_company_augmented = has_company.merge(
    augmented[['company', 'fraudulent', 'review', 'has_website', 'year', 'company_size']],
    on=['company', 'fraudulent'],
    how='left'
)

# Combine the augmented rows with the rows that have no company
company_feature = pd.concat([has_company_augmented, no_company], ignore_index=True)

# Save the output
company_feature.to_csv("/content/company_feature_augmented.csv", index=False)

