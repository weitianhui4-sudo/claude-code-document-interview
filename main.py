"""
Entry point for the LexisNexis NLP Document Classification task.
"""

from pathlib import Path
from sklearn.model_selection import train_test_split

from src.classifier import (
    SyntheticDataset,
    TextPreprocessor,
    TfidfClassifier,
    EmbeddingClassifier,
    Evaluator,
)
from src.classifier.dataset import LABEL_NAMES, DEFAULT_PATH


def main():
    # --- Data: generate and save once, load on subsequent runs ---
    if not DEFAULT_PATH.exists():
        print("Generating and saving synthetic dataset...")
        SyntheticDataset(n=10_000).save(DEFAULT_PATH)

    raw_df = SyntheticDataset.load(DEFAULT_PATH)
    print(f"Class distribution:\n{raw_df['label'].value_counts()}\n")

    # --- Preprocessing ---
    print("Cleaning text...")
    preprocessor = TextPreprocessor()
    clean_df = preprocessor.transform(raw_df)
    print(f"Sample: {clean_df['clean_text'].iloc[0]}\n")

    # --- Split ---
    X = clean_df['clean_text'].tolist()
    y = clean_df['label'].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    evaluator = Evaluator(label_names=LABEL_NAMES)
    results = {}

    # --- TF-IDF + Random Forest ---
    print("=== TF-IDF + Random Forest ===")
    tfidf_model = TfidfClassifier()
    tfidf_model.fit(X_train, y_train)
    results['TF-IDF + Random Forest'] = evaluator.evaluate(y_test, tfidf_model.predict(X_test))

    # --- Embedding + Random Forest (optional) ---
    try:
        print("=== Embeddings + Random Forest ===")
        emb_model = EmbeddingClassifier()
        emb_model.fit(X_train, y_train)
        results['Embeddings + Random Forest'] = evaluator.evaluate(y_test, emb_model.predict(X_test))
    except ImportError as e:
        print(f"[SKIP] {e}\n")

    # --- Comparison table ---
    evaluator.compare(results)


if __name__ == '__main__':
    main()
