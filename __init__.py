"""
word2vec_numpy — Pure-NumPy word2vec library.

Public API
----------
    Word2Vec        — main model class
    Word2VecConfig  — hyperparameter configuration
    Vocabulary      — vocabulary builder
"""

from .config import Word2VecConfig
from .model import Word2Vec
from .vocabulary import Vocabulary

__all__ = ["Word2Vec", "Word2VecConfig", "Vocabulary"]
__version__ = "0.1.0"
