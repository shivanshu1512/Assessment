# -*- coding: utf-8 -*-
"""
Support Ticket Classifier
=========================
Classifies support tickets into: Billing, Technical, HR, General

Pipeline:
  Text Cleaning -> TF-IDF Vectorization -> Naive Bayes / Logistic Regression
  -> Confidence Score -> Priority Tagging -> Human-Review Threshold

Author: Shivanshu Shukla
"""

import re
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Windows
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# -----------------------------------------------------------
#  Constants
# -----------------------------------------------------------

CATEGORIES = ["Billing", "Technical", "HR", "General"]
CONFIDENCE_THRESHOLD = 0.60   # below this -> flag for human review

URGENT_KEYWORDS = [
    "urgent", "immediately", "critical", "asap", "down",
    "not working", "broken", "emergency", "cannot", "failed",
]

DATA_PATH = "data/tickets.csv"


# -----------------------------------------------------------
#  1. Text Preprocessing
# -----------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Lowercase, strip punctuation/numbers, collapse whitespace.

    Why minimal stopword removal?
    -> TF-IDF already down-weights high-frequency words via IDF.
    -> For short ticket texts, aggressive removal loses context
       (e.g. "not working" becomes "working" which flips meaning).
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)   # remove non-alpha
    text = re.sub(r"\s+", " ", text).strip() # collapse spaces
    return text


def load_and_preprocess(path: str) -> pd.DataFrame:
    """Load CSV, merge subject + body, clean text."""
    df = pd.read_csv(path)
    # Combine subject and body -- both carry signal
    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    df["clean_text"] = df["text"].apply(clean_text)
    df["category"] = df["category"].str.strip()
    print(f"[OK] Loaded {len(df)} tickets across {df['category'].nunique()} categories")
    print(f"     Category distribution:")
    for cat, cnt in df['category'].value_counts().items():
        print(f"       {cat:<12} : {cnt}")
    print()
    return df


# -----------------------------------------------------------
#  2. Feature Engineering -- TF-IDF
# -----------------------------------------------------------

def build_vectorizer() -> TfidfVectorizer:
    """
    TF-IDF with unigrams + bigrams.

    Why TF-IDF over simple Bag-of-Words?
    -> Penalises common words that appear in every ticket (IDF part)
    -> Rewards words specific to a ticket/category (TF part)
    -> Bigrams capture phrases like 'not working', 'payment failed'
    -> sublinear_tf: log(1+tf) dampens very high-frequency terms
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
    )


# -----------------------------------------------------------
#  3. Model Training
# -----------------------------------------------------------

def train_models(X_train, y_train):
    """
    Train both Naive Bayes and Logistic Regression.

    Why these two?
    -> Naive Bayes: fast, works well on sparse TF-IDF, great baseline.
    -> Logistic Regression: models feature correlations better, usually
       edges out NB on small labelled datasets.
    Both are trained -- we compare and pick the best.
    """
    nb = MultinomialNB(alpha=0.5)   # alpha = Laplace smoothing
    nb.fit(X_train, y_train)

    lr = LogisticRegression(
        max_iter=1000,
        C=5.0,           # inverse regularisation strength
        solver="lbfgs",
    )
    lr.fit(X_train, y_train)

    return nb, lr


# -----------------------------------------------------------
#  4. Evaluation
# -----------------------------------------------------------

def evaluate(model, X_test, y_test, model_name: str):
    """Print accuracy, classification report, and save confusion matrix."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc:.2%}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=sorted(set(y_test))))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=CATEGORIES)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax,
    )
    ax.set_title(f"Confusion Matrix -- {model_name}", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    filename = f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"  [OK] Confusion matrix saved: {filename}\n")
    return acc


# -----------------------------------------------------------
#  5. Priority Tagging
# -----------------------------------------------------------

def get_priority(text: str) -> str:
    """
    Keyword-based priority tagger. Returns 'URGENT' or 'NORMAL'.
    Rule-based tagging is transparent and requires no training data.
    """
    text_lower = text.lower()
    for kw in URGENT_KEYWORDS:
        if kw in text_lower:
            return "URGENT"
    return "NORMAL"


# -----------------------------------------------------------
#  6. Prediction with Confidence + Human Review Flag
# -----------------------------------------------------------

def predict_ticket(model, vectorizer, ticket_text: str) -> dict:
    """
    Predict category with confidence score.
    If confidence < CONFIDENCE_THRESHOLD -> route to human review queue.
    """
    clean = clean_text(ticket_text)
    X = vectorizer.transform([clean])
    proba = model.predict_proba(X)[0]
    classes = model.classes_

    predicted_idx = int(np.argmax(proba))
    predicted_category = classes[predicted_idx]
    confidence = proba[predicted_idx]
    needs_review = confidence < CONFIDENCE_THRESHOLD
    priority = get_priority(ticket_text)

    return {
        "predicted_category": predicted_category,
        "confidence": round(float(confidence) * 100, 2),
        "needs_human_review": needs_review,
        "priority": priority,
        "all_probabilities": {
            cat: round(float(p) * 100, 2)
            for cat, p in zip(classes, proba)
        },
    }


def display_prediction(result: dict, ticket_text: str):
    """Pretty-print a single prediction result."""
    flag = "!! NEEDS HUMAN REVIEW" if result["needs_human_review"] else ">> AUTO-ASSIGNED"
    priority_icon = "[URGENT]" if result["priority"] == "URGENT" else "[NORMAL]"

    print(f"\n{'─'*60}")
    print(f"  Ticket  : {ticket_text[:80]}{'...' if len(ticket_text) > 80 else ''}")
    print(f"  Category: {result['predicted_category']}  ({result['confidence']}% confidence)")
    print(f"  Priority: {priority_icon}")
    print(f"  Status  : {flag}")
    print(f"  Score breakdown:")
    for cat, prob in sorted(result["all_probabilities"].items(), key=lambda x: -x[1]):
        bar = "#" * int(prob / 5)
        print(f"    {cat:<12} {prob:5.1f}%  {bar}")
    print(f"{'─'*60}")


# -----------------------------------------------------------
#  7. Sample Test Tickets (5+ new tickets not in training data)
# -----------------------------------------------------------

SAMPLE_TICKETS = [
    "I was charged twice this month and need an immediate refund.",
    "The app keeps crashing whenever I try to upload a file. Please fix this.",
    "I need to apply for paternity leave next month. What is the process?",
    "Can you tell me more about your enterprise pricing and discount options?",
    "Our webhook is not firing after your latest deployment. This is urgent.",
    "My salary increment letter has not been issued even after 3 months.",
    "Why am I getting a 500 server error every time I hit the checkout API?",
]


# -----------------------------------------------------------
#  Main
# -----------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("   SUPPORT TICKET CLASSIFIER -- Shivanshu Shukla")
    print("=" * 60 + "\n")

    # Step 1: Load & Preprocess
    df = load_and_preprocess(DATA_PATH)

    # Step 2: Train / Test Split
    X_raw = df["clean_text"]
    y = df["category"]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 3: TF-IDF Vectorisation
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    print(f"[OK] TF-IDF vocabulary size: {len(vectorizer.vocabulary_)} features")
    print(f"     Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples\n")

    # Step 4: Train Models
    print("[*] Training Naive Bayes and Logistic Regression...")
    nb_model, lr_model = train_models(X_train, y_train)

    # Step 5: Evaluate
    acc_nb = evaluate(nb_model, X_test, y_test, "Naive Bayes")
    acc_lr = evaluate(lr_model, X_test, y_test, "Logistic Regression")

    # Step 6: Choose Best Model
    best_model = lr_model if acc_lr >= acc_nb else nb_model
    best_name = "Logistic Regression" if acc_lr >= acc_nb else "Naive Bayes"
    print(f"\n[BEST] Model selected: {best_name} (Accuracy: {max(acc_nb, acc_lr):.2%})\n")

    # Step 7: Predict on 5+ New Sample Tickets
    print("=" * 60)
    print("   PREDICTIONS ON NEW SAMPLE TICKETS")
    print("=" * 60)

    for ticket in SAMPLE_TICKETS:
        result = predict_ticket(best_model, vectorizer, ticket)
        display_prediction(result, ticket)

    # Step 8: Interactive CLI Demo
    print("\n" + "=" * 60)
    print("   LIVE DEMO -- Type a ticket below (or 'exit' to quit)")
    print("=" * 60)
    while True:
        try:
            user_input = input("\n  Enter ticket text: ").strip()
            if user_input.lower() in ("exit", "quit", "q"):
                print("\n[OK] Exiting. Thank you!\n")
                break
            if not user_input:
                print("  [!] Please enter some text.")
                continue
            result = predict_ticket(best_model, vectorizer, user_input)
            display_prediction(result, user_input)
        except (KeyboardInterrupt, EOFError):
            print("\n[OK] Exiting.\n")
            break


if __name__ == "__main__":
    main()
