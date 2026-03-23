from __future__ import annotations

import numpy as np

from word2vec_numpy.word2vec_numpy.config import Word2VecConfig
from ..vocabulary import Vocabulary
from ..utils import make_sigmoid_table
from .base import BaseTrainer


class SkipGramHierarchicalSoftmax(BaseTrainer):
    """
    Skip-Gram with Hierarchical Softmax trainer.

    Uses a Huffman binary tree to predict context words from the center word
    in O(log V) operations per context word, instead of comparing against
    negative samples or the full vocabulary.
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

    def train_pair(self, center: int, contexts: np.ndarray, lr: float) -> float:
        v_c = self.W[center]
        total_loss = 0.0

        for ctx in contexts:
            # HS predicts the context word from the center word
            points = self.vocab.point[ctx]   # Internal node indices [0, V-2]
            codes = self.vocab.code[ctx]     # Binary codes (0 or 1)

            if len(points) == 0:
                continue

            U_path = self.W_[points]

            # Forward
            dots = U_path @ v_c
            scores = self._sigmoid(dots)

            # In word2vec.c, the target label at each tree node is (1 - code).
            # So if code is 0, the target label is 1 (take left branch).
            # If code is 1, the target label is 0 (take right branch).
            labels = 1 - codes

            # Loss: Binary Cross-Entropy against the branch labels
            # Pos branch (label=1): -log(score), Neg branch (label=0): -log(1-score)
            loss_pos = np.where(labels == 1, -np.log(scores + 1e-7), 0.0)
            loss_neg = np.where(labels == 0, -np.log(1.0 - scores + 1e-7), 0.0)
            total_loss += float((loss_pos + loss_neg).sum())

            # Error gradient: pred - true
            err = scores - labels

            # Accumulated gradient for v_c
            grad_v_c_step = (err[:, None] * U_path).sum(axis=0)

            # Update the internal node embeddings (W_)
            grad_U_path = err[:, None] * v_c[None, :]
            
            # Since `points` traces a strict path from root to leaf, there are NO duplicates.
            # So we can safely use standard `-=` assignment, but we use add.at for safety 
            # and to maintain idiomatic NumPy consistency with our other trainers.
            np.add.at(self.W_, points, -lr * grad_U_path)

            # Update center word sequentially
            v_c -= lr * grad_v_c_step

        return total_loss / max(len(contexts)+1, 1)
