import pandas as pd
import re
import numpy as np

# Load dataset
df = pd.read_csv("/content/company_feature_augmented.csv")

# Combine relevant text fields
text_fields = ['description', 'requirements', 'benefits']
df['text_all'] = df[text_fields].fillna('').agg(' '.join, axis=1)

# Regex for salary
salary_pattern = re.compile(
    r'(?i)\$?\s?(\d{2,3}(?:[,\.\s]?\d{3})*)(?:\s?[-to–]\s?\$?\s?(\d{2,3}(?:[,\.\s]?\d{3})*))?\s?(per\s(month|year|week|hour))?',
    re.IGNORECASE
)

# Clean salary string
def clean_salary_string(s):
    try:
        return float(re.sub(r'[,\.\s]', '', s.strip()))
    except:
        return None

# Main extraction logic
def extract_salary(row):
    salary_range = str(row['salary_range']).strip().lower()

    # If salary_range is valid, use it directly
    if pd.notna(row['salary_range']) and salary_range != 'missing' and salary_range != '':
        df.at[row.name, 'extracted_salary'] = row['salary_range']
        try:
            s = str(row['salary_range']).replace("$", "").replace(",", "").replace("–", "-").replace("to", "-")
            parts = [p.strip() for p in s.split("-")]
            if len(parts) == 1:
                val = clean_salary_string(parts[0])
                if val is not None:
                    return pd.Series([val, val])
            elif len(parts) == 2:
                min_val = clean_salary_string(parts[0])
                max_val = clean_salary_string(parts[1])
                if min_val is not None and max_val is not None:
                    return pd.Series([min_val, max_val])
            return pd.Series([-2, -2])
        except:
            return pd.Series([-2, -2])

    # Extract from text_all
    matches = salary_pattern.findall(row['text_all'])
    if matches:
        raw_min, raw_max, _, period = matches[0]
        min_sal = clean_salary_string(raw_min)
        max_sal = clean_salary_string(raw_max) if raw_max else min_sal

        if min_sal is None or max_sal is None:
            df.at[row.name, 'extracted_salary'] = 'Missing'
            return pd.Series([-2, -2])

        if period:
            period = period.lower()
            if period == 'month':
                min_sal *= 12
                max_sal *= 12
            elif period == 'week':
                min_sal *= 52
                max_sal *= 52
            elif period == 'hour':
                min_sal *= 2080
                max_sal *= 2080

        if min_sal < 10000 or max_sal < 10000:
            df.at[row.name, 'extracted_salary'] = 'Missing'
            return pd.Series([-2, -2])

        df.at[row.name, 'extracted_salary'] = f"{min_sal:.0f} - {max_sal:.0f}" if min_sal != max_sal else f"{min_sal:.0f}"
        return pd.Series([min_sal, max_sal])

    # If nothing is found
    df.at[row.name, 'extracted_salary'] = 'Missing'
    return pd.Series([-1, -1])

# Apply to DataFrame
df[['min_salary', 'max_salary']] = df.apply(extract_salary, axis=1)

# Salary average
df['salary_avg'] = df[['min_salary', 'max_salary']].mean(axis=1)

# Save result
df.to_csv("/content/Job_Postings_with_Final_Extracted_Salary.csv", index=False)
