"""
Entry point for the LexisNexis NLP Document Classification task.
"""

from sklearn.model_selection import train_test_split

from src.classifier import (
    SyntheticDataset,
    TextPreprocessor,
    TfidfClassifier,
    EmbeddingClassifier,
    Evaluator,
)
from src.classifier.dataset import LABEL_NAMES


def main():
    # --- Data ---
    print("Generating synthetic dataset...")
    raw_df = SyntheticDataset(n=10_000).generate()
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

    # --- TF-IDF baseline ---
    print("=== TF-IDF Classifier ===")
    tfidf_model = TfidfClassifier()
    tfidf_model.fit(X_train, y_train)
    y_pred = tfidf_model.predict(X_test)
    results['TF-IDF + LogReg'] = evaluator.evaluate(y_test, y_pred)

    # --- Embedding model (optional) ---
    try:
        print("=== Embedding Classifier ===")
        emb_model = EmbeddingClassifier()
        emb_model.fit(X_train, y_train)
        y_pred_emb = emb_model.predict(X_test)
        results['Embeddings + LogReg'] = evaluator.evaluate(y_test, y_pred_emb)
    except ImportError as e:
        print(f"[SKIP] {e}\n")

    # --- Summary ---
    evaluator.compare(results)


if __name__ == '__main__':
    main()
