"""
LexisNexis NLP & Document Classification Solution

Dataset: 10,000 legal paragraphs/news articles
Labels: Compliance Risk, Contract Breach, Neutral
Challenges: Severe class imbalance, noisy metadata
"""

import re
import html
import numpy as np
import pandas as pd
from typing import Optional

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, f1_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# Optional: sentence-transformers for embedding-based approach
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


# ---------------------------------------------------------------------------
# 1. Wrangle and Clean
# ---------------------------------------------------------------------------

# Noise patterns common in system-generated legal docs
_NOISE_PATTERNS = [
    r'\x00-\x08\x0b\x0c\x0e-\x1f\x7f',   # control characters
    r'[�﻿]',                       # unicode replacement / BOM
    r'\b(?:NULL|NULLVALUE|N/A|undefined)\b', # placeholder tokens
    r'\|\|+',                                # pipe separators (metadata noise)
    r'={3,}|-{3,}|\*{3,}',                  # decorative dividers
    r'\[\s*\]|\(\s*\)',                      # empty brackets
]
_NOISE_RE = re.compile(
    '|'.join(_NOISE_PATTERNS), re.IGNORECASE | re.UNICODE
)
_WHITESPACE_RE = re.compile(r'\s+')


def clean_text(text: Optional[str]) -> str:
    """
    Clean a single document:
    1. Coerce non-strings to empty string (never drops records)
    2. Unescape HTML entities (&amp; → &)
    3. Strip HTML/XML tags
    4. Remove system-generated noise characters and tokens
    5. Normalize whitespace
    """
    if not isinstance(text, str):
        text = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)

    # unescape HTML entities before stripping tags
    text = html.unescape(text)

    # strip HTML/XML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # remove noise
    text = _NOISE_RE.sub(' ', text)

    # normalize whitespace
    text = _WHITESPACE_RE.sub(' ', text).strip()

    return text


def wrangle_dataframe(df: pd.DataFrame, text_col: str = 'text', label_col: str = 'label') -> pd.DataFrame:
    """
    Wrangle the raw dataset:
    - Clean text column in-place (no rows dropped)
    - Strip whitespace from labels
    - Return copy with added 'clean_text' column
    """
    df = df.copy()
    df['clean_text'] = df[text_col].apply(clean_text)
    if label_col in df.columns:
        df[label_col] = df[label_col].astype(str).str.strip()
    return df


# ---------------------------------------------------------------------------
# 2. Feature Extraction — TF-IDF Baseline vs Sentence Embeddings
# ---------------------------------------------------------------------------

LABEL_NAMES = ['Compliance Risk', 'Contract Breach', 'Neutral']


def build_tfidf_pipeline(class_weight='balanced') -> Pipeline:
    """
    Baseline: TF-IDF (unigrams + bigrams) → Logistic Regression.
    class_weight='balanced' compensates for class imbalance automatically.
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50_000,
            sublinear_tf=True,       # log(1+tf) dampens frequency dominance
            min_df=2,
            strip_accents='unicode',
            analyzer='word',
        )),
        ('clf', LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            C=1.0,
            solver='lbfgs',
        )),
    ])


def get_sentence_embeddings(texts: list[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Dense embeddings via SentenceTransformers.
    Falls back to None if library not installed.
    """
    if not HAS_SENTENCE_TRANSFORMERS:
        return None
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)


def build_embedding_classifier(class_weight='balanced') -> LogisticRegression:
    """Classifier head for sentence embedding features."""
    return LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
    )


# ---------------------------------------------------------------------------
# 3. Training & Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred, label_names=LABEL_NAMES) -> dict:
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(classification_report(y_true, y_pred, target_names=label_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    return {'macro_f1': macro_f1, 'report': report}


def run_experiment(df: pd.DataFrame, text_col: str = 'clean_text', label_col: str = 'label'):
    """
    Full experiment:
    1. Train/test split (stratified to preserve class ratios)
    2. Fit TF-IDF baseline and evaluate
    3. Fit embedding model (if available) and compare
    """
    X = df[text_col].tolist()
    y = df[label_col].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    # --- Baseline: TF-IDF + LogReg ---
    print("\n=== TF-IDF Baseline ===")
    tfidf_pipe = build_tfidf_pipeline()
    tfidf_pipe.fit(X_train, y_train)
    y_pred_tfidf = tfidf_pipe.predict(X_test)
    results['tfidf'] = evaluate(y_test, y_pred_tfidf)

    # --- Embedding-based model ---
    if HAS_SENTENCE_TRANSFORMERS:
        print("\n=== Sentence Embeddings (all-MiniLM-L6-v2) ===")
        X_train_emb = get_sentence_embeddings(X_train)
        X_test_emb = get_sentence_embeddings(X_test)
        emb_clf = build_embedding_classifier()
        emb_clf.fit(X_train_emb, y_train)
        y_pred_emb = emb_clf.predict(X_test_emb)
        results['embeddings'] = evaluate(y_test, y_pred_emb)
    else:
        print("\n[INFO] sentence-transformers not installed; skipping embedding model.")
        print("       Install with: pip install sentence-transformers")

    # --- Summary ---
    print("\n=== Summary ===")
    for name, res in results.items():
        print(f"  {name:15s} macro-F1: {res['macro_f1']:.4f}")

    return results


# ---------------------------------------------------------------------------
# 4. Data Generation (synthetic demo — replace with real dataset)
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Produces a synthetic imbalanced dataset that mirrors the real task.
    Class distribution: Neutral 80%, Compliance Risk 13%, Contract Breach 7%
    Includes HTML noise and system-generated artifacts in raw text.
    """
    rng = np.random.default_rng(seed)
    labels = rng.choice(
        LABEL_NAMES,
        size=n,
        p=[0.13, 0.07, 0.80]
    )

    templates = {
        'Compliance Risk': [
            "<p>The entity failed to adhere to &amp; regulatory requirements under Section {s}.</p> ||NULL||",
            "ALERT: Compliance breach detected in fiscal period {s}. ===",
            "<div>Non-compliance with AML directives noted. Record ID: {s} �</div>",
        ],
        'Contract Breach': [
            "<b>Party A has materially breached clause {s} of the agreement.</b>",
            "CONTRACT VIOLATION\x00: delivery obligation unmet as of period {s}. ***",
            "Breach of warranty under &lt;Section {s}&gt; confirmed by legal review.",
        ],
        'Neutral': [
            "The quarterly report for period {s} has been filed with the registrar.",
            "<p>Standard operating procedure {s} reviewed and approved. N/A</p>",
            "Document {s} archived. No further action required. []",
        ],
    }

    records = []
    for i, label in enumerate(labels):
        tmpl = templates[label][i % len(templates[label])]
        records.append({'text': tmpl.format(s=rng.integers(100, 999)), 'label': label})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Generating synthetic dataset...")
    raw_df = generate_synthetic_dataset(n=10_000)

    print(f"Raw class distribution:\n{raw_df['label'].value_counts()}\n")

    print("Wrangling and cleaning...")
    clean_df = wrangle_dataframe(raw_df)

    print(f"Sample cleaned text:\n{clean_df['clean_text'].iloc[0]}\n")

    run_experiment(clean_df)
