import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments

def load_and_prepare_dataset(csv_path):
    df = pd.read_csv(csv_path)
    # Sample a smaller subset for faster training
    df_small = df.sample(n=2000, random_state=42)
    train_df = df_small.sample(frac=0.8, random_state=42)
    val_df = df_small.drop(train_df.index)

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    return DatasetDict({'train': train_ds, 'validation': val_ds})

def tokenize_data(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)
    return dataset.map(tokenize, batched=True)

def main():
    csv_path = "data/combined_dataset_balanced.csv"
    print("Loading dataset...")
    dataset = load_and_prepare_dataset(csv_path)

    print("Loading tokenizer and model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    print("Tokenizing dataset...")
    tokenized_dataset = tokenize_data(dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir="model_output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")

    print("Saving model...")
    trainer.save_model("model_output")

if __name__ == "__main__":
    main()
