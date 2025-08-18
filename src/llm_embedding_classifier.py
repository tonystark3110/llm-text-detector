# src/llm_embedding_classifier.py

import os
import json
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def load_human_wikipedia(sample_size=5000):
    print("Loading human-written Wikipedia articles...")
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split=f"train[:{sample_size}]")
    human_texts = [article["text"] for article in wiki]
    df_human = pd.DataFrame({"text": human_texts, "label": 0})
    print(f"Loaded {len(df_human)} human samples.")
    return df_human

def load_machine_jsonl(jsonl_path):
    print(f"Loading machine-generated Wikipedia samples from {jsonl_path} ...")
    machine_texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                machine_texts.append(rec["machine_text"])
    df_machine = pd.DataFrame({"text": machine_texts, "label": 1})
    print(f"Loaded {len(df_machine)} machine samples.")
    return df_machine

def main():
    # SET THIS to your actual file path
    machine_jsonl_path = "data/wikipedia_davinci.jsonl"

    # --- STEP 1: Load data ---
    df_human = load_human_wikipedia(sample_size=5000)
    df_machine = load_machine_jsonl(machine_jsonl_path)

    # Combine and shuffle
    df_all = pd.concat([df_human, df_machine], ignore_index=True)
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nCombined dataframe, total samples: {len(df_all)}")
    print(df_all['label'].value_counts())

    # --- STEP 2: Embedding extraction ---
    print("Generating sentence embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(
        df_all["text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    # --- STEP 3: Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, df_all["label"], test_size=0.2, stratify=df_all["label"], random_state=42
    )

    # --- STEP 4: Train classifier ---
    print("\nTraining Logistic Regression classifier...")
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # --- STEP 5: Evaluation ---
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Human", "Machine-Generated"]))

    # --- Optional: Save predictions ---
    pred_df = df_all.loc[y_test.index].copy()

    pred_df['y_true'] = y_test
    pred_df['y_pred'] = y_pred
    pred_df.to_csv("data/wiki_llm_classifier_predictions.csv", index=False)
    print("\nClassification predictions saved to data/wiki_llm_classifier_predictions.csv")
    import joblib
    joblib.dump(clf, "data/logreg_model.pkl")
    joblib.dump(model, "data/embedding_model.pkl")  # saves the SentenceTransformer for reuse


if __name__ == "__main__":
    main()
