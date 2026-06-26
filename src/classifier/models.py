import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import TfidfFeaturizer, EmbeddingFeaturizer


class TfidfClassifier:
    """
    TF-IDF + Logistic Regression pipeline.
    class_weight='balanced' handles severe class imbalance automatically.
    """

    def __init__(self, C: float = 1.0, max_features: int = 50_000):
        self._featurizer = TfidfFeaturizer(max_features=max_features)
        self._clf = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=C,
            solver='lbfgs',
        )

    def fit(self, texts: list[str], labels: list[str]) -> 'TfidfClassifier':
        X = self._featurizer.fit_transform(texts)
        self._clf.fit(X, labels)
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        X = self._featurizer.transform(texts)
        return self._clf.predict(X)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self._featurizer.transform(texts)
        return self._clf.predict_proba(X)

    @property
    def classes_(self):
        return self._clf.classes_


class EmbeddingClassifier:
    """
    Sentence-embedding + Logistic Regression classifier.
    Requires sentence-transformers to be installed.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', C: float = 1.0):
        self._featurizer = EmbeddingFeaturizer(model_name=model_name)
        self._clf = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=C,
            solver='lbfgs',
        )

    def fit(self, texts: list[str], labels: list[str]) -> 'EmbeddingClassifier':
        X = self._featurizer.fit_transform(texts)
        self._clf.fit(X, labels)
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        X = self._featurizer.transform(texts)
        return self._clf.predict(X)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self._featurizer.transform(texts)
        return self._clf.predict_proba(X)

    @property
    def classes_(self):
        return self._clf.classes_
