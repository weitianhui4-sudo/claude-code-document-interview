import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class TfidfFeaturizer:
    """
    Wraps sklearn TfidfVectorizer with sensible defaults for legal text.
    Unigrams + bigrams, sublinear TF scaling, 50k vocabulary cap.
    """

    def __init__(self, max_features: int = 50_000, ngram_range: tuple = (1, 2)):
        self._vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            min_df=2,
            strip_accents='unicode',
            analyzer='word',
        )

    def fit(self, texts: list[str]) -> 'TfidfFeaturizer':
        self._vectorizer.fit(texts)
        return self

    def transform(self, texts: list[str]) -> spmatrix:
        return self._vectorizer.transform(texts)

    def fit_transform(self, texts: list[str]) -> spmatrix:
        return self._vectorizer.fit_transform(texts)

    @property
    def vocabulary_size(self) -> int:
        return len(self._vectorizer.vocabulary_)


class EmbeddingFeaturizer:
    """
    Dense sentence embeddings via SentenceTransformers.
    Raises ImportError at construction if library not installed.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 64):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers is required for EmbeddingFeaturizer. "
                "Install with: pip install sentence-transformers"
            )
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size

    def transform(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    # fit is a no-op (pre-trained model)
    def fit(self, texts: list[str]) -> 'EmbeddingFeaturizer':
        return self

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)
