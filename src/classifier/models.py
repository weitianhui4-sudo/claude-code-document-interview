import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .features import TfidfFeaturizer, EmbeddingFeaturizer


class TfidfClassifier:
    """
    TF-IDF + Random Forest pipeline.
    class_weight='balanced' handles severe class imbalance automatically.
    """

    def __init__(self, n_estimators: int = 200, max_features: int = 50_000):
        self._featurizer = TfidfFeaturizer(max_features=max_features)
        self._clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
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
    Sentence-embedding + Random Forest classifier.
    Requires sentence-transformers to be installed.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', n_estimators: int = 200):
        self._featurizer = EmbeddingFeaturizer(model_name=model_name)
        self._clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
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
