"""
word2vec_numpy/trainers/base.py
--------------------------------
Abstract base class for all word2vec trainers.

Subclasses implement ``train_pair`` for a single (center, context) pair
and receive the embedding matrices by reference so that updates are applied
in-place (no copies required).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from word2vec_numpy.word2vec_numpy.config import Word2VecConfig
from ..vocabulary import Vocabulary


class BaseTrainer(ABC):
    """
    Common interface for word2vec training objectives.

    Parameters
    ----------
    W:
        Input embedding matrix, shape ``(vocab_size, embed_dim)``.
        Also called ``syn0`` in the C implementation.
    W_:
        Output embedding matrix, shape ``(vocab_size, embed_dim)``.
        Also called ``syn1neg`` in the C implementation.
    vocab:
        Vocabulary object (used for negative sampling).
    config:
        Training configuration.
    rng:
        NumPy random generator for reproducibility.
    """

    def __init__(
        self,
        W: np.ndarray,
        W_: np.ndarray,
        vocab: Vocabulary,
        config: Word2VecConfig,
        rng: np.random.Generator,
    ) -> None:
        self.W = W
        self.W_ = W_
        self.vocab = vocab
        self.config = config
        self.rng = rng

    @abstractmethod
    def train_pair(
        self,
        center: int,
        contexts: np.ndarray,
        lr: float,
    ) -> float:
        """
        Perform a forward pass, compute loss, back-propagate gradients,
        and update the embedding matrices in-place.

        Parameters
        ----------
        center:
            Index of the center word.
        contexts:
            Array of context-word indices for this center word.
        lr:
            Current learning rate (after schedule decay).

        Returns
        -------
        float
            The loss value for this update step.
        """
