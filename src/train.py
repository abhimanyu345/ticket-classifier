%%writefile src/train.py
import os
import numpy as np
import pandas as pd
import wandb
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME    = "distilbert-base-uncased"
MAX_LENGTH    = 128
BATCH_SIZE    = 32
EPOCHS        = 4
LR            = 3e-5
OUTPUT_DIR    = "models/distilbert-ticket-classifier"
PROCESSED_DIR = "data/processed"

# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    train = pd.read_csv(f"{PROCESSED_DIR}/train.csv")
    val   = pd.read_csv(f"{PROCESSED_DIR}/val.csv")
    test  = pd.read_csv(f"{PROCESSED_DIR}/test.csv")
    return train, val, test

# ── Encode labels ─────────────────────────────────────────────────────────────
def encode_labels(train, val, test):
    le = LabelEncoder()
    train["label_id"] = le.fit_transform(train["label"])
    val["label_id"]   = le.transform(val["label"])
    test["label_id"]  = le.transform(test["label"])
    return train, val, test, le

# ── Tokenize ──────────────────────────────────────────────────────────────────
def tokenize(df, tokenizer):
    dataset = Dataset.from_pandas(
        df[["text", "label_id"]].rename(columns={"label_id": "labels"})
    )
    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )
    return dataset.map(_tokenize, batched=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    wandb.init(
        project="ticket-classifier",
        config={
            "model":      MODEL_NAME,
            "epochs":     EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr":         LR,
            "max_length": MAX_LENGTH,
            "dataset":    "twitter-customer-support",
        }
    )

    print("Loading data...")
    train_df, val_df, test_df = load_data()
    train_df, val_df, test_df, le = encode_labels(train_df, val_df, test_df)

    num_labels = len(le.classes_)
    print(f"Classes ({num_labels}): {list(le.classes_)}")

    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds  = tokenize(train_df, tokenizer)
    val_ds    = tokenize(val_df,   tokenizer)
    test_ds   = tokenize(test_df,  tokenizer)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label={i: l for i, l in enumerate(le.classes_)},
        label2id={l: i for i, l in enumerate(le.classes_)},
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="wandb",
        run_name=wandb.run.name,
        logging_steps=50,
        weight_decay=0.01,
        warmup_steps=200,
        fp16=True,              # faster training on T4 GPU
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    # ── Test set evaluation ───────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    preds_output = trainer.predict(test_ds)
    preds  = np.argmax(preds_output.predictions, axis=-1)
    labels = test_ds["labels"]

    report = classification_report(labels, preds, target_names=le.classes_)
    print(report)

    wandb.log({
        "test_accuracy": accuracy_score(labels, preds),
        "test_f1_macro": f1_score(labels, preds, average="macro"),
        "classification_report": wandb.Table(
            columns=["report"],
            data=[[report]]
        )
    })

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")

    # ── Log artifact to WandB ─────────────────────────────────────────────────
    artifact = wandb.Artifact(
        name="ticket-classifier-model",
        type="model",
        description="DistilBERT fine-tuned on Twitter customer support",
        metadata={
            "f1_macro":  f1_score(labels, preds, average="macro"),
            "accuracy":  accuracy_score(labels, preds),
            "dataset":   "twitter-customer-support",
            "classes":   list(le.classes_),
        }
    )
    artifact.add_dir(OUTPUT_DIR)
    wandb.log_artifact(artifact)
    print("Model artifact logged to WandB")

    wandb.finish()

if __name__ == "__main__":
    main()