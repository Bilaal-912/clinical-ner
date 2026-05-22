import pandas as pd
import re
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'\n',' ',text)
    text = re.sub(r'\s+',' ',text)
    text = re.sub(r'[^\w\s\.\,\-]', '', text)
    text = text.strip()
    return text

def preprocess_dataset(input_path,output_path):
    df=load_data(input_path)
    df=df.dropna(subset=['transcription'])
    df['clean_transcription']=df['transcription'].apply(clean_text)
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    df.to_csv(output_path,index=False)
    print(f"Saved cleaned data to {output_path}")
    return df

if __name__ == "__main__":
    input_path = "data/raw/mtsamples.csv"
    output_path = "data/processed/mtsamples_cleaned.csv"
    preprocess_dataset(input_path, output_path)