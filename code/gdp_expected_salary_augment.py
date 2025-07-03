import pandas as pd
from fuzzywuzzy import process

# Load the job postings dataset
df = pd.read_csv("Job_Postings_with_Salary_Extracted.csv")

# Extract country and assign GDP values
df['country'] = df['location'].apply(lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'Unknown')

# Mapping to ISO codes
country_code_map = {
    'US': 'USA', 'GB': 'GBR', 'GR': 'GRC', 'NG': 'NGA', 'CA': 'CAN', 'IN': 'IND',
    'PK': 'PAK', 'DE': 'DEU', 'FR': 'FRA', 'KE': 'KEN', 'AE': 'ARE', 'Unknown': 'UNK'
}
df['country_code'] = df['country'].map(country_code_map).fillna('UNK')

# Load GDP data and compute 2012–2014 average
gdp = pd.read_csv("API_NY.GDP.PCAP.CD_DS2_en_csv_v2_26433.csv", skiprows=4)
gdp = gdp[['Country Code', '2012', '2013', '2014', '2023']]
gdp[['2012', '2013', '2014', '2023']] = gdp[['2012', '2013', '2014', '2023']].apply(pd.to_numeric, errors='coerce')
gdp['GDP_2012_2014_Avg'] = gdp[['2012', '2013', '2014']].mean(axis=1)
gdp.rename(columns={'2023': 'GDP_2023'}, inplace=True)

# Merge GDP info to main data
df = df.merge(gdp[['Country Code', 'GDP_2012_2014_Avg', 'GDP_2023']],
              how='left', left_on='country_code', right_on='Country Code')

# Assign tagged GDP based on dataset source
df['tagged_gdp'] = df.apply(
    lambda x: x['GDP_2023'] if x['country_code'] == 'NGA' else x['GDP_2012_2014_Avg'],
    axis=1
)

# Compute salary average and normalize using GDP 

df['salary_avg'] = df[['min_salary', 'max_salary']].mean(axis=1)

df['expected_salary_gdp'] = df.apply(
    lambda x: round(x['salary_avg'] / x['tagged_gdp'], 2)
    if x['salary_avg'] > 0 and x['tagged_gdp'] > 0 else -1,
    axis=1
)

# Fuzzy match job titles to standardized roles and assign expected salary 

# Load role-based salary benchmark
ref = pd.read_csv("Salary_Data_Based_country_and_race.csv")

# Compute average salary per role
salary_map = (
    ref.groupby('Job Title')['Salary']
    .mean().round(2)
    .reset_index()
    .rename(columns={'Job Title': 'role', 'Salary': 'expected_salary'})
)

# Match titles to closest role label
roles = salary_map['role'].dropna().unique().tolist()

def map_role_fuzzy(title):
    if pd.isna(title) or not title.strip():
        return "Unknown"
    best_match, score = process.extractOne(title, roles)
    return best_match if score >= 80 else "Unknown"

df['mapped_role'] = df['title'].apply(map_role_fuzzy)

# Merge expected salary from role benchmark
df = df.merge(salary_map, how='left', left_on='mapped_role', right_on='role')
df.drop(columns=['role'], inplace=True)
df.rename(columns={'expected_salary': 'role_expected_salary'}, inplace=True)

df.to_csv("Final_Job_Postings.csv", index=False)
