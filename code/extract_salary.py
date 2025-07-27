import pandas as pd
import re
import numpy as np

# Load preprocessed dataset
df = pd.read_csv("/content/company_feature_augmented.csv")

# Combine relevant text fields
text_fields = ['description', 'requirements', 'benefits']
df['text_all'] = df[text_fields].fillna('').agg(' '.join, axis=1)

# Regular expression to detect salary patterns
salary_pattern = re.compile(
    r'(?i)\$?\s?(\d{2,3}(?:[,\.\s]?\d{3})*)(?:\s?[-to–]\s?\$?\s?(\d{2,3}(?:[,\.\s]?\d{3})*))?\s?(per\s(month|year|week|hour))?',
    re.IGNORECASE
)

# Function to convert string salary to float
def clean_salary_string(s):
    return float(s.replace(',', '').replace('.', '').strip())

# Function to extract min and max salary
def extract_salary(text):
    matches = salary_pattern.findall(text)
    if matches:
        raw_min, raw_max, _, period = matches[0]
        min_sal = clean_salary_string(raw_min)
        max_sal = clean_salary_string(raw_max) if raw_max else min_sal

        # Normalize monthly or weekly or hourly to annual if possible
        if period:
            period = period.lower()
            if period == 'month':
                min_sal *= 12
                max_sal *= 12
            elif period == 'week':
                min_sal *= 52
                max_sal *= 52
            elif period == 'hour':
                min_sal *= 2080  # assuming 40 hours/week * 52 weeks

        return pd.Series([min_sal, max_sal])
    return pd.Series([-1, -1])  # Placeholder for missing

# Apply to text_all column
df[['min_salary', 'max_salary']] = df['text_all'].apply(extract_salary)

# Tag missing extracted salary rows
df['extracted_salary'] = df.apply(
    lambda x: 'Missing' if x['min_salary'] == -1 and x['max_salary'] == -1 else 'Extracted', axis=1
)

# Save the result
df.to_csv("/content/Job_Postings_with_Salary_Extracted.csv", index=False)
