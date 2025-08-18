import json
import os
import pandas as pd
from datasets import load_dataset
import textstat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# --------- Step 1: Load Human Wikipedia Data ---------
def load_human_wiki_samples(sample_size=5000):
    print("Loading Wikipedia human-written samples...")
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split=f"train[:{sample_size}]")
    texts = [article["text"] for article in wiki]
    df_human = pd.DataFrame({"text": texts, "label": 0})
    print(f"Loaded {len(df_human)} human samples")
    return df_human

# --------- Step 2: Load Machine-Generated Wikipedia Data ---------
def load_machine_generated_jsonl(filepath):
    print(f"Loading machine-generated data from {filepath} ...")
    data = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                # Use 'machine_text' field
                data.append(entry['machine_text'])
    df_machine = pd.DataFrame({"text": data, "label": 1})
    print(f"Loaded {len(df_machine)} machine samples")
    return df_machine

# --------- Step 3: Feature Extraction ---------
def extract_stylometric_features(df):
    print("Extracting stylometric features...")
    features = []
    
    for text in df['text']:
        try:
            flesch = textstat.flesch_reading_ease(text)
        except:
            flesch = 0.0
        try:
            syllables = textstat.syllable_count(text)
        except:
            syllables = 0
        words = text.split()
        lex_div = len(set(words)) / (len(words) + 1e-6) if len(words) > 0 else 0
        
        features.append([flesch, syllables, lex_div])
    feature_df = pd.DataFrame(features, columns=['flesch_reading_ease', 'syllable_count', 'lexical_diversity'])
    return feature_df

# --------- Main Pipeline ---------
def main():
    # Paths -- Adjust path to your actual jsonl machine-generated Wikipedia file
    machine_jsonl_path = "data/wikipedia_davinci.jsonl"
    
    # 1. Load datasets
    df_human = load_human_wiki_samples(sample_size=5000)   # Get 5k human Wikipedia articles
    df_machine = load_machine_generated_jsonl(machine_jsonl_path)
    
    # 2. Combine and shuffle
    df_all = pd.concat([df_human, df_machine], ignore_index=True)
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Total samples combined: {len(df_all)}")
    print(df_all['label'].value_counts())
    
    # 3. Extract features
    feature_df = extract_stylometric_features(df_all)
    
    # 4. Prepare train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, df_all['label'], test_size=0.2, random_state=42, stratify=df_all['label'])
    
    # 5. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Train logistic regression classifier
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    # 7. Evaluate
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Human", "Machine-Generated"]))
    
    # Optional: Show feature coefficients
    print("\nFeature Coefficients:")
    for feature, coef in zip(feature_df.columns, clf.coef_[0]):
        print(f"  {feature}: {coef:.4f}")

if __name__ == "__main__":
    main()
