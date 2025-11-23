"""
Intent classifier using Hugging Face transformers + Trainer.
Place this file at: src/nlu/intent_classifier.py

Requirements: transformers, torch, scikit-learn, pandas
You already have these installed in your venv.

How to run (example):
> python src/nlu/intent_classifier.py --train
> python src/nlu/intent_classifier.py --predict "How much money do I have?"
"""

import os
import json
import argparse
from typing import List, Dict

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# -----------------------
# Config
# -----------------------
DEFAULT_MODEL_NAME = "distilbert-base-uncased"  # lightweight, fast to fine-tune
OUTPUT_DIR = "models/intent_classifier"
INTENTS_PATH = "data/intents/intents.json"
RANDOM_SEED = 42

# -----------------------
# Dataset helper
# -----------------------
class SimpleTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        txt = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            txt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

# -----------------------
# Data loading / formatting
# -----------------------
def load_intent_data(intents_json_path: str = INTENTS_PATH) -> pd.DataFrame:
    with open(intents_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for intent_name, obj in data.items():
        examples = obj.get("examples", [])
        for ex in examples:
            rows.append({"text": ex, "intent": intent_name})
    df = pd.DataFrame(rows)
    return df

def prepare_datasets(tokenizer, test_size: float = 0.15, val_size: float = 0.15, random_state: int = RANDOM_SEED):
    df = load_intent_data()
    if df.empty:
        raise ValueError(f"No examples found in {INTENTS_PATH}")

    # label encoding
    label2id = {label: i for i, label in enumerate(sorted(df["intent"].unique()))}
    id2label = {i: label for label, i in label2id.items()}
    df["label"] = df["intent"].map(label2id)

    # train / temp (temp -> val + test)
    train_df, temp_df = train_test_split(df, test_size=(test_size + val_size), stratify=df["label"], random_state=random_state)
    # split temp into val & test with ratio based on sizes
    val_ratio = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
    temp_df, 
    test_size=(1 - val_ratio), 
    random_state=random_state,
    stratify=None  # disable stratification for tiny dataset
    )

    def df_to_dataset(dframe):
        return SimpleTextDataset(dframe["text"].tolist(), dframe["label"].tolist(), tokenizer)

    train_dataset = df_to_dataset(train_df)
    val_dataset = df_to_dataset(val_df)
    test_dataset = df_to_dataset(test_df)

    return train_dataset, val_dataset, test_dataset, label2id, id2label, df

# -----------------------
# Metrics
# -----------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# -----------------------
# Training routine
# -----------------------
def train(model_name: str = DEFAULT_MODEL_NAME,
          output_dir: str = OUTPUT_DIR,
          epochs: int = 3,
          batch_size: int = 8,
          learning_rate: float = 2e-5,
          save_strategy: str = "epoch"):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds, val_ds, test_ds, label2id, id2label, full_df = prepare_datasets(tokenizer)

    num_labels = len(label2id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy=save_strategy,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_strategy="epoch",
        load_best_model_at_end=True if save_strategy == "epoch" else False,
        metric_for_best_model="accuracy",
        seed=RANDOM_SEED,
        fp16=torch.cuda.is_available(),  # use mixed precision if available
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # Evaluate on test set
    print("Evaluating on test set...")
    preds_output = trainer.predict(test_ds)
    preds = np.argmax(preds_output.predictions, axis=1)
    labels = preds_output.label_ids
    acc = accuracy_score(labels, preds)
    print(f"Test accuracy: {acc:.4f}")
    print("\nClassification report:")
    try:
        print(classification_report(labels, preds, target_names=[id2label[i] for i in sorted(id2label.keys())]))
    except Exception:
        pass

    # Save model + tokenizer + label mapping
    print(f"Saving model & tokenizer to {output_dir} ...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "label2id.json"), "w", encoding="utf-8") as fh:
        json.dump(label2id, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "id2label.json"), "w", encoding="utf-8") as fh:
        json.dump(id2label, fh, indent=2, ensure_ascii=False)

    print("Artifacts saved.")
    return output_dir

# -----------------------
# Inference utilities
# -----------------------
def load_trained_model(model_dir: str = OUTPUT_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    with open(os.path.join(model_dir, "label2id.json"), "r", encoding="utf-8") as fh:
        label2id = json.load(fh)
    with open(os.path.join(model_dir, "id2label.json"), "r", encoding="utf-8") as fh:
        id2label = json.load(fh)
    return model, tokenizer, label2id, id2label

def predict_intent(text: str, model=None, tokenizer=None, id2label=None, device=None, top_k: int = 1):
    if model is None or tokenizer is None or id2label is None:
        model, tokenizer, _, id2label = load_trained_model()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
        topk_idx = np.argsort(probs)[-top_k:][::-1]
        results = [(id2label[str(i)] if isinstance(list(id2label.keys())[0], str) else id2label[i], float(probs[i])) for i in topk_idx]
    return results

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--predict", type=str, default=None, help="Run a sample prediction")
    args = parser.parse_args()

    if args.train:
        train(model_name=args.model_name, epochs=args.epochs, batch_size=args.batch_size)

    if args.predict:
        # load model and run prediction
        print("Predicting:", args.predict)
        model, tokenizer, label2id, id2label = load_trained_model()
        res = predict_intent(args.predict, model=model, tokenizer=tokenizer, id2label=id2label)
        print("Predictions:", res)

if __name__ == "__main__":
    main()
