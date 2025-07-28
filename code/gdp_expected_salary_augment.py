import pandas as pd
from fuzzywuzzy import process

# Load the job postings dataset
df = pd.read_csv("/content/Job_Postings_with_Final_Extracted_Salary.csv")

# Extract country and assign GDP values
df['country'] = df['location'].apply(lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'Unknown')

# Mapping to ISO codes
country_code_map = {
    'US': 'USA', 'GB': 'GBR', 'GR': 'GRC', 'NG': 'NGA', 'CA': 'CAN', 'IN': 'IND',
    'PK': 'PAK', 'DE': 'DEU', 'FR': 'FRA', 'KE': 'KEN', 'AE': 'ARE', 'Unknown': 'UNK'
}
df['country_code'] = df['country'].map(country_code_map).fillna('UNK')

# Load GDP data and compute 2012–2014 average
gdp = pd.read_excel("/content/API_NY.GDP.PCAP.CD_DS2_en_excel_v2_26418.xls", skiprows=4)
gdp_subset = gdp.iloc[:, [1, 56, 57, 58, 67]].copy()
gdp_subset.columns = ['Country Code', '2012', '2013', '2014', '2023']
gdp_subset[['2012', '2013', '2014', '2023']] = gdp_subset[['2012', '2013', '2014', '2023']].apply(pd.to_numeric, errors='coerce')
gdp_subset['GDP_2012_2014_Avg'] = gdp_subset[['2012', '2013', '2014']].mean(axis=1)
gdp_subset.rename(columns={'2023': 'GDP_2023'}, inplace=True)

# Merge GDP info to main data
df = df.merge(gdp_subset[['Country Code', 'GDP_2012_2014_Avg', 'GDP_2023']],
              how='left', left_on='country_code', right_on='Country Code')

# Assign tagged GDP
df['tagged_gdp'] = df.apply(
    lambda x: x['GDP_2023'] if x['country_code'] == 'NGA' else x['GDP_2012_2014_Avg'],
    axis=1
)

# Fuzzy match job titles to standardized roles
ref = pd.read_csv("/content/Salary_Data_Based_country_and_race.csv")
salary_map = (
    ref.groupby('Job Title')['Salary']
    .mean().round(2)
    .reset_index()
    .rename(columns={'Job Title': 'role', 'Salary': 'base_expected_salary'})
)
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

# Adjust expected salary using GDP
df['expected_salary'] = df.apply(
    lambda x: round(x['base_expected_salary'] * x['tagged_gdp'], 2)
    if pd.notna(x['base_expected_salary']) and pd.notna(x['tagged_gdp']) else -1,
    axis=1
)

# Compute salary deviation (absolute difference)
df['salary_avg'] = df[['min_salary', 'max_salary']].mean(axis=1)
df['salary_deviation'] = df.apply( 
    lambda x: abs(x['salary_avg'] - x['expected_salary'])
    if x['expected_salary'] > 0 and x['salary_avg'] > 0 else -1,
    axis=1
)

# Normalize salary deviation (z-score), ignoring -1 placeholders
valid_deviations = df[df['salary_deviation'] != -1]['salary_deviation']
mean_dev = valid_deviations.mean()
std_dev = valid_deviations.std()

df['salary_deviation_norm'] = df['salary_deviation'].apply(
    lambda x: (x - mean_dev) / std_dev if x != -1 else -1
)

# Save output
df.to_csv("Updated_Job_Postings.csv", index=False)
