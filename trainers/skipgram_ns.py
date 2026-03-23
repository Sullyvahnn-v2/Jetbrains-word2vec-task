"""
word2vec_numpy/trainers/skipgram_ns.py
---------------------------------------
Skip-gram with Negative Sampling (SGNS) trainer.

Math recap
----------
For a center word c and one context word o, with K negative samples
n_1 … n_K drawn from the noise distribution:

  Forward
  -------
    pos_score   = σ( W[c] · W_[o])     label = 1
    neg_score_k = σ( W[c] · W_[n_k])   label = 0   for k = 1…K

  Loss (binary cross-entropy, positive label=1, negative label=0)
  ---------------------------------------------------------------
    L = -log(pos_score) - Σ_k log(1 - neg_score_k)

  Gradients  (∂L/∂z for BCE: σ(z) - label)
  -----------------------------------------
    err_pos   = pos_score  - 1          (positive: σ(z) - 1)
    err_neg_k = neg_score_k - 0         (negative: σ(z) - 0)

    ∂L/∂W[c]     = err_pos · W_[o]  + Σ_k err_neg_k · W_[n_k]
    ∂L/∂W_[o]    = err_pos · W[c]
    ∂L/∂W_[n_k]  = err_neg_k · W[c]

  SGD update
  ----------
    W[c]     -= lr · ∂L/∂W[c]
    W_[o]    -= lr · ∂L/∂W_[o]
    W_[n_k]  -= lr · ∂L/∂W_[n_k]

Vectorisation strategy
----------------------
For each center word, ALL context words in the window are batched together
with the negative samples to maximise NumPy throughput and minimise Python
loop overhead.  Within each step:

  * K negatives are drawn once per context word and stacked.
  * All dot products, sigmoid calls, and gradient accumulations use
    NumPy vectorised ops.
  * np.add.at() is used for gradient scatter-adds so that repeated word
    indices (same word appearing multiple times in a window) are handled
    correctly — standard indexed assignment would silently drop duplicates.
"""

from __future__ import annotations

import numpy as np

from ..config import Word2VecConfig
from ..vocabulary import Vocabulary
from ..utils import make_sigmoid_table
from .base import BaseTrainer


class SkipGramNegativeSampling(BaseTrainer):
    """
    Skip-gram with Negative Sampling trainer.

    One call to ``train_pair`` processes a *single center word* together
    with *all its context words* in one vectorised pass.

    Parameters
    ----------
    W:
        Input embedding matrix (syn0), shape ``(V, D)``.  Updated in-place.
    W_:
        Output embedding matrix (syn1neg), shape ``(V, D)``.  Updated in-place.
    vocab:
        Vocabulary (used for negative sampling).
    config:
        Training configuration.
    rng:
        NumPy random generator.
    """

    def __init__(
        self,
        W: np.ndarray,
        W_: np.ndarray,
        vocab: Vocabulary,
        config: Word2VecConfig,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(W, W_, vocab, config, rng)
        self._sigmoid = make_sigmoid_table(
            table_size=config.sigmoid_table_size,
            max_val=config.sigmoid_max,
        )

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def train_pair(self, center: int, contexts: np.ndarray, lr: float) -> float:
        v_c = self.W[center]           # Reference! Modified in-place inside loop!
        K = self.config.negative
        total_loss = 0.0

        for ctx in contexts:
            u_o = self.W_[ctx].copy()    # copy to avoid aliasing if same word

            negatives = self.vocab.sample_negatives(K, self.rng, exclude=ctx)
            U_neg = self.W_[negatives]

            pos_dot = float(v_c @ u_o)
            neg_dots = U_neg @ v_c

            pos_score  = float(self._sigmoid(pos_dot))
            neg_scores = self._sigmoid(neg_dots)

            pos_loss = -np.log(pos_score + 1e-7)
            neg_loss = -np.log(1.0 - neg_scores + 1e-7).sum()
            total_loss += float(pos_loss + neg_loss)

            err_pos = pos_score - 1.0
            err_neg = neg_scores

            # Gradient for center word vector
            grad_v_c_step = err_pos * u_o + (err_neg[:, None] * U_neg).sum(axis=0)

            # Gradient and update for positive output vector
            self.W_[ctx] -= lr * (err_pos * v_c)

            # np.add.at is MANDATORY because negatives can contain DUPLICATES!
            # Standard W_[negatives] -= ... silently drops duplicate updates!
            grad_U_neg = err_neg[:, None] * v_c[None, :] 
            np.add.at(self.W_, negatives, -lr * grad_U_neg)
            
            # Apply center-word gradient sequentially exactly like C Word2Vec!
            v_c -= lr * grad_v_c_step

        return total_loss / max(len(contexts), 1)
