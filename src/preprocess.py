import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os

RAW_PATH     = "data/raw/twcs.csv"
PROCESSED_DIR = "data/processed"

# ── Keyword rules to assign labels ───────────────────────────────────────────
# Order matters — first match wins
LABEL_RULES = [
    ("Billing inquiry", [
        "bill", "charge", "charged", "invoice", "payment", "pay",
        "overcharged", "fee", "price", "cost", "debit", "credit",
        "transaction", "statement", "amount", "money", "subscription"
    ]),
    ("Refund request", [
        "refund", "reimburse", "reimbursement", "money back",
        "return", "cashback", "compensation", "repay"
    ]),
    ("Cancellation request", [
        "cancel", "cancellation", "unsubscribe", "terminate",
        "close account", "end subscription", "stop service",
        "discontinue", "opt out"
    ]),
    ("Technical issue", [
        "not working", "broken", "error", "bug", "crash", "issue",
        "problem", "fix", "slow", "freeze", "won't load", "cant login",
        "can't login", "unable", "failed", "failure", "down", "outage",
        "glitch", "lag", "update", "install", "connection", "login"
    ]),
    ("Product inquiry", [
        "how do i", "how to", "where can", "what is", "feature",
        "available", "support", "compatible", "plan", "upgrade",
        "difference", "option", "offer", "service", "question", "info",
        "information", "help", "wondering", "want to know"
    ]),
]

def assign_label(text):
    text_lower = text.lower()
    for label, keywords in LABEL_RULES:
        if any(kw in text_lower for kw in keywords):
            return label
    return None  # no match → drop this row

def clean_text(text):
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove HTML entities
    text = re.sub(r'&amp;|&lt;|&gt;', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading data...")
    # Use chunking since the file is 2.8M rows
    chunks = []
    for chunk in pd.read_csv(RAW_PATH, chunksize=100_000,
                              usecols=["inbound", "text"]):
        # Keep only customer tweets
        chunk = chunk[chunk["inbound"] == True].copy()
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    print(f"Customer tweets loaded: {len(df)}")

    # Clean text
    df["text"] = df["text"].astype(str).apply(clean_text)

    # Drop short tweets (less than 20 chars after cleaning)
    df = df[df["text"].str.len() >= 20]

    # Assign labels
    df["label"] = df["text"].apply(assign_label)

    # Drop unmatched rows
    df = df.dropna(subset=["label"])
    df = df[["text", "label"]]
    df = df.drop_duplicates(subset=["text"])

    print(f"After labeling: {len(df)} rows")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # Balance classes — cap at 5000 per class so training is fast on Colab
    df = pd.concat([
        group.sample(min(len(group), 5000), random_state=42)
        for _, group in df.groupby("label")
    ]).reset_index(drop=True)
    print(f"After balancing: {len(df)} rows")
    print(df["label"].value_counts())

    # Train / val / test split 70/15/15
    train, temp = train_test_split(df, test_size=0.30,
                                   random_state=42, stratify=df["label"])
    val, test   = train_test_split(temp, test_size=0.50,
                                   random_state=42, stratify=temp["label"])

    train.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)
    val.to_csv(f"{PROCESSED_DIR}/val.csv",     index=False)
    test.to_csv(f"{PROCESSED_DIR}/test.csv",   index=False)

    print(f"\nSaved → train: {len(train)}, val: {len(val)}, test: {len(test)}")
    print(f"\nSample texts:")
    for label in df["label"].unique():
        sample = df[df["label"] == label]["text"].iloc[0]
        print(f"\n[{label}]: {sample[:100]}")

if __name__ == "__main__":
    preprocess()