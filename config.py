"""
word2vec_numpy/config.py
------------------------
Central configuration object.  Every tuneable knob lives here so that
experiments are fully reproducible and the rest of the codebase stays clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Word2VecConfig:
    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------
    embed_dim: int = 300
    """Dimensionality of the word embedding vectors."""

    model: Literal["skipgram", "cbow"] = "skipgram"
    """Training architecture.  Only 'skipgram' is implemented; 'cbow' is
    reserved for future expansion."""

    loss: Literal["negative_sampling", "hs"] = "negative_sampling"
    """Output approximation.  Only 'negative_sampling' is implemented;
    'hs' (hierarchical softmax) is reserved for future expansion."""

    # ------------------------------------------------------------------
    # Corpus pre-processing
    # ------------------------------------------------------------------
    min_count: int = 5
    """Discard words that appear fewer than this many times."""

    subsample_t: float = 1e-3
    """Frequent-word subsampling threshold (set to 0.0 to disable).
    Words with frequency f are discarded with probability
    P = 1 - sqrt(t / f).  Recommended: 1e-3 to 1e-5."""

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    window: int = 5
    """Maximum context half-window size (tokens on each side of center)."""

    dynamic_window: bool = True
    """If True, sample the actual window size uniformly from [1, window]
    on every center word (matches original word2vec behaviour)."""

    negative: int = 5
    """Number of negative samples drawn per positive (center, context) pair."""

    epochs: int = 5
    """Number of full passes over the corpus."""

    learning_rate: float = 0.025
    """Initial SGD learning rate."""

    min_lr: float = 1e-4
    """Minimum learning rate floor for linear decay."""

    seed: int = 42
    """Random seed for reproducibility."""

    # ------------------------------------------------------------------
    # Negative-sampling noise distribution
    # ------------------------------------------------------------------
    ns_exponent: float = 0.75
    """Unigram distribution exponent.  Negative samples are drawn from
    P(w) ∝ f(w)^ns_exponent.  Value of 0.75 from the original paper."""

    table_size: int = 100_000_00
    """Size of the unigram look-up table used for negative sampling."""

    # ------------------------------------------------------------------
    # Sigmoid look-up table
    # ------------------------------------------------------------------
    sigmoid_table_size: int = 1_000_00
    """Number of entries in the pre-computed sigmoid table."""

    sigmoid_max: float = 6.0
    """Inputs outside [-sigmoid_max, +sigmoid_max] are clamped."""

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.embed_dim < 1:
            raise ValueError(f"embed_dim must be ≥ 1, got {self.embed_dim}")
        if self.min_count < 1:
            raise ValueError(f"min_count must be ≥ 1, got {self.min_count}")
        if self.window < 1:
            raise ValueError(f"window must be ≥ 1, got {self.window}")
        if self.negative < 1:
            raise ValueError(f"negative must be ≥ 1, got {self.negative}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be ≥ 1, got {self.epochs}")
        if not 0.0 < self.learning_rate:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.min_lr >= self.learning_rate:
            raise ValueError("min_lr must be strictly less than learning_rate")
        if self.subsample_t < 0.0:
            raise ValueError(f"subsample_t must be ≥ 0, got {self.subsample_t}")
        if self.model not in ("skipgram", "cbow"):
            raise ValueError(f"Unknown model '{self.model}'. Choose 'skipgram' or 'cbow'.")
        if self.loss not in ("negative_sampling", "hs"):
            raise ValueError(f"Unknown loss '{self.loss}'. Choose 'negative_sampling' or 'hs'.")
        if self.model == "cbow":
            raise NotImplementedError("CBOW is not yet implemented.")
        if self.loss == "hs":
            raise NotImplementedError("Hierarchical softmax is not yet implemented.")
