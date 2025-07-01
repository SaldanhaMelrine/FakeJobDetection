import pandas as pd
from transformers import pipeline

# Load dataset
df = pd.read_csv("/content/DataSet.csv") #Update path

# Load BERT-based NER model
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# Company name extraction function
def extract_company_bert(text):
    if pd.isna(text): return None
    try:
        entities = ner(text)
        for ent in entities:
            if ent['entity_group'] == 'ORG':
                return ent['word']
    except:
        return None
    return None
    
df.loc['company'] = df.loc['company_profile'].apply(extract_company_bert)

# Save the updated DataFrame
df.to_csv("/content/EMSCAD_with_Company.csv", index=False) #Update path

