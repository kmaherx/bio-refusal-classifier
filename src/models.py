"""Classifier protocol and v1 implementation.

V1 has a single concrete impl: sentence-transformer embeddings + sklearn
logistic regression. The Protocol is kept generic so future implementations
slot in without touching ``train.py``:

- ``TFIDFLogReg(ngram_range, max_features, ...)`` — same interface, sklearn
  ``TfidfVectorizer`` + LR.
- ``FineTunedEncoder(model_name, lr, epochs, ...)`` — wraps a ``Trainer`` from
  ``transformers``.
- ``ActivationProbe(layer_idx, pooling, ...)`` — receives pre-extracted
  activations rather than ``list[str]``; the ``X`` parameter is intentionally
  untyped so this works without changing callers.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import ExperimentConfig
from .utils import get_device


@runtime_checkable
class Classifier(Protocol):
    def fit(self, X: Any, y: np.ndarray) -> "Classifier": ...
    def predict(self, X: Any) -> np.ndarray: ...
    def predict_proba(self, X: Any) -> np.ndarray: ...


class EmbeddingLogReg:
    """Encode text with a SentenceTransformer, then fit sklearn LR.

    Encoder is loaded lazily on first ``fit`` so importing this module doesn't
    pull torch into RAM.
    """

    def __init__(
        self,
        embedding_model_name: str,
        classifier_kwargs: dict[str, Any] | None = None,
        batch_size: int = 64,
        device: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.classifier_kwargs = dict(classifier_kwargs or {})
        self.batch_size = batch_size
        self.device = device or get_device()
        self.normalize_embeddings = normalize_embeddings

        self._encoder = None
        self.classifier: LogisticRegression | None = None

    def _ensure_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.embedding_model_name, device=self.device)
        return self._encoder

    def _encode(self, texts: list[str]) -> np.ndarray:
        enc = self._ensure_encoder()
        return enc.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def fit(self, X: list[str], y: np.ndarray) -> "EmbeddingLogReg":
        feats = self._encode(list(X))
        self.classifier = LogisticRegression(**self.classifier_kwargs)
        self.classifier.fit(feats, y)
        return self

    def _check_fit(self) -> None:
        if self.classifier is None:
            raise RuntimeError("EmbeddingLogReg.fit() must be called before predict")

    def predict(self, X: list[str]) -> np.ndarray:
        self._check_fit()
        return self.classifier.predict(self._encode(list(X)))

    def predict_proba(self, X: list[str]) -> np.ndarray:
        self._check_fit()
        return self.classifier.predict_proba(self._encode(list(X)))


def build_classifier(cfg: ExperimentConfig) -> Classifier:
    if cfg.model_type == "embedding_logreg":
        return EmbeddingLogReg(
            embedding_model_name=cfg.embedding_model_name,
            classifier_kwargs=cfg.classifier_kwargs,
            batch_size=cfg.embedding_batch_size,
            normalize_embeddings=cfg.normalize_embeddings,
        )
    raise ValueError(f"unknown model_type {cfg.model_type!r}")
