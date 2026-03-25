"""Deep Learning model for column semantic classification.

Combines pre-computed text embeddings with engineered tabular features
through an MLP classifier implemented in PyTorch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from column_engine.models.base import BaseColumnClassifier


class ColumnMLP(nn.Module):
    """Multi-Layer Perceptron that fuses embedding and tabular features.

    Architecture
    ------------
    1. Optional linear projection of text embeddings
    2. Concatenation with tabular (scalar) features
    3. Two hidden layers with ReLU + dropout
    4. Softmax output
    """

    def __init__(
        self,
        embedding_dim: int,
        n_tabular_features: int,
        n_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding_proj = nn.Linear(embedding_dim, hidden_dim)
        combined_dim = hidden_dim + n_tabular_features
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        emb = torch.relu(self.embedding_proj(embeddings))
        x = torch.cat([emb, tabular], dim=1)
        return self.classifier(x)


class DeepColumnClassifier(BaseColumnClassifier):
    """PyTorch-based deep column classifier.

    Wraps :class:`ColumnMLP` to conform to the :class:`BaseColumnClassifier`
    interface.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        n_tabular_features: Optional[int] = None,
        n_classes: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self._n_tabular: Optional[int] = n_tabular_features
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[ColumnMLP] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self, n_tabular: int) -> None:
        self._n_tabular = n_tabular
        self._model = ColumnMLP(
            embedding_dim=self.embedding_dim,
            n_tabular_features=n_tabular,
            n_classes=self.n_classes,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)

    @staticmethod
    def _split_features(
        X: np.ndarray, embedding_dim: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split a feature matrix into embedding and tabular parts.

        Convention: the first ``embedding_dim`` columns are embedding
        features; the rest are scalar tabular features.
        """
        emb = X[:, :embedding_dim]
        tab = X[:, embedding_dim:]
        return emb, tab

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        emb, tab = self._split_features(X, self.embedding_dim)
        n_tabular = tab.shape[1]
        self._build_model(n_tabular)
        assert self._model is not None

        emb_t = torch.tensor(emb, dtype=torch.float32, device=self.device)
        tab_t = torch.tensor(tab, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)

        dataset = TensorDataset(emb_t, tab_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self._model.train()
        for _ in range(self.epochs):
            for emb_b, tab_b, y_b in loader:
                optimizer.zero_grad()
                logits = self._model(emb_b, tab_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Model has not been trained yet."
        emb, tab = self._split_features(X, self.embedding_dim)
        emb_t = torch.tensor(emb, dtype=torch.float32, device=self.device)
        tab_t = torch.tensor(tab, dtype=torch.float32, device=self.device)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(emb_t, tab_t)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()
