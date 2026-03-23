"""
word2vec_numpy/utils.py
------------------------
Lightweight utilities shared across the library:

  * sigmoid()             — fast look-up-table sigmoid (avoids np.exp in
                            the inner training loop)
  * LinearDecaySchedule   — linear learning-rate schedule
  * cosine_similarity()   — L2-normalised dot product
  * most_similar()        — nearest-neighbour word lookup
  * analogy()             — analogy evaluation (b - a + c ≈ ?)
"""

from __future__ import annotations

import numpy as np


# -----------------------------------------------------------------------
# Fast sigmoid via pre-computed look-up table
# -----------------------------------------------------------------------

class _SigmoidTable:
    """
    Approximates σ(x) = 1 / (1 + exp(-x)) using a pre-computed table.

    Mirrors the ``expTable`` in the original C word2vec implementation.
    Values outside [-max_val, +max_val] are clamped to 0 or 1 respectively.
    """

    def __init__(self, table_size: int = 1_000, max_val: float = 6.0) -> None:
        self.table_size = table_size
        self.max_val = max_val
        xs = np.linspace(-max_val, max_val, table_size, dtype=np.float64)
        self._table = (1.0 / (1.0 + np.exp(-xs))).astype(np.float32)

    def __call__(self, x: float | np.ndarray) -> np.ndarray:
        """
        Evaluate sigmoid for scalar or array input.

        Parameters
        ----------
        x : float or ndarray
            Input value(s).

        Returns
        -------
        ndarray of float32 with the same shape as ``x``.
        """
        x = np.asarray(x, dtype=np.float32)
        # Map x ∈ [-max, +max] → table index ∈ [0, table_size-1]
        idx = ((x + self.max_val) / (2.0 * self.max_val) * self.table_size).astype(np.int32)
        idx = np.clip(idx, 0, self.table_size - 1)
        
        # In C word2vec:
        # if (f > MAX_EXP) g = (label - 1) * alpha;  -> sigmoid=1.0
        # else if (f < -MAX_EXP) g = (label - 0) * alpha; -> sigmoid=0.0
        result = self._table[idx]
        result = np.where(x > self.max_val, 1.0, result)
        result = np.where(x < -self.max_val, 0.0, result)
        
        return result.astype(np.float32)


# Module-level default table (matches default config values)
_default_table = _SigmoidTable()


def sigmoid(
    x: float | np.ndarray,
    table: _SigmoidTable | None = None,
) -> np.ndarray:
    """
    Fast sigmoid approximation via a pre-computed look-up table.

    Parameters
    ----------
    x:
        Scalar or array of input values.
    table:
        Optional custom ``_SigmoidTable`` (uses library default if None).

    Returns
    -------
    np.ndarray of float32.
    """
    t = table if table is not None else _default_table
    return t(x)


def make_sigmoid_table(table_size: int, max_val: float) -> _SigmoidTable:
    """Create a sigmoid table with custom resolution and range."""
    return _SigmoidTable(table_size=table_size, max_val=max_val)


# -----------------------------------------------------------------------
# Learning-rate schedule
# -----------------------------------------------------------------------

class LinearDecaySchedule:
    """
    Linearly decays the learning rate from ``lr0`` down to ``min_lr``
    over ``total_steps`` gradient steps.

    lr(step) = max(lr0 * (1 - step / total_steps), min_lr)

    Matches the alpha decay in the original C word2vec implementation.
    """

    def __init__(self, lr0: float, min_lr: float, total_steps: int) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        self.lr0 = lr0
        self.min_lr = min_lr
        self.total_steps = total_steps

    def get(self, step: int) -> float:
        """Return the learning rate at ``step``."""
        progress = min(step / self.total_steps, 1.0)
        return max(self.lr0 * (1.0 - progress), self.min_lr)


# -----------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.

    Returns 0.0 (rather than NaN) when either vector has zero norm.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def most_similar(
    word: str,
    embeddings: np.ndarray,
    word2idx: dict[str, int],
    idx2word: list[str],
    n: int = 10,
) -> list[tuple[str, float]]:
    """
    Return the ``n`` most similar words to ``word`` by cosine similarity.

    Parameters
    ----------
    word:
        Query word.
    embeddings:
        Matrix of shape ``(vocab_size, embed_dim)``.
    word2idx:
        Token → index mapping.
    idx2word:
        Index → token mapping.
    n:
        Number of neighbours to return (excluding query word itself).

    Returns
    -------
    List of ``(word, similarity)`` tuples, sorted descending.
    """
    if word not in word2idx:
        raise KeyError(f"'{word}' not in vocabulary")

    query_idx = word2idx[word]
    query_vec = embeddings[query_idx]
    query_norm = np.linalg.norm(query_vec)

    if query_norm == 0.0:
        return []

    # Vectorised cosine similarity against all embeddings
    norms = np.linalg.norm(embeddings, axis=1)               # (V,)
    norms = np.where(norms == 0.0, 1e-12, norms)             # avoid /0
    sims = (embeddings @ query_vec) / (norms * query_norm)   # (V,)

    # Exclude the query word itself
    sims[query_idx] = -np.inf

    top_indices = np.argsort(sims)[::-1][:n]
    return [(idx2word[i], float(sims[i])) for i in top_indices]

