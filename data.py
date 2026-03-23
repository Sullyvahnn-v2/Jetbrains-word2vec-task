"""
word2vec_numpy/data.py
-----------------------
Corpus streaming and (center, context) training-pair generation.

Design goals
------------
* Memory-efficient: sentences are processed one at a time; the full corpus
  is never loaded into RAM.
* Faithful to the original C implementation: dynamic window, on-the-fly
  subsampling, and OOV skipping are all handled here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Generator, Iterable

import numpy as np

from .config import Word2VecConfig
from .vocabulary import Vocabulary


# -----------------------------------------------------------------------
# Sentence sources
# -----------------------------------------------------------------------

def sentences_from_file(
    path: str | Path,
    chunk_size: int = 1_000,
) -> Generator[list[str], None, None]:
    """
    Yield fixed-length token chunks from a whitespace-tokenised text file.

    text8 is a single flat token stream with no sentence boundaries, so we
    split it into chunks of ``chunk_size`` tokens (1 000 by default) to
    approximate sentences.  This matches the conventional approach used by
    most word2vec tutorials.

    Parameters
    ----------
    path:
        Path to the text file (plain text, whitespace-separated tokens).
    chunk_size:
        Number of tokens per pseudo-sentence.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        buffer: list[str] = []
        for line in f:
            tokens = line.split()
            buffer.extend(tokens)
            while len(buffer) >= chunk_size:
                yield buffer[:chunk_size]
                buffer = buffer[chunk_size:]
        if buffer:
            yield buffer


def sentences_from_list(
    sentences: list[list[str]],
) -> Generator[list[str], None, None]:
    """Yield tokenised sentences from an in-memory list."""
    yield from sentences


# -----------------------------------------------------------------------
# Training-pair generator
# -----------------------------------------------------------------------

class Corpus:
    """
    Wraps a sentence source and yields (center_idx, context_idx) integer
    pairs ready for consumption by a trainer.

    Responsibilities
    ----------------
    * Convert tokens to vocabulary indices (skip OOV).
    * Apply frequent-word subsampling on-the-fly.
    * Sample a dynamic window size per center word (if enabled).
    * Yield all valid (center, context) pairs within each window.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        config: Word2VecConfig,
        rng: np.random.Generator,
    ) -> None:
        self.vocab = vocab
        self.config = config
        self.rng = rng

    def iter_pairs(
        self,
        sentences: Iterable[list[str]],
    ) -> Generator[tuple[int, list[int]], None, None]:
        """
        Yield ``(center_word_idx, list_of_context_indices)`` pairs.

        Parameters
        ----------
        sentences:
            An iterable of token lists (from ``sentences_from_file`` or
            ``sentences_from_list``).
        """
        cfg = self.config
        vocab = self.vocab
        rng = self.rng

        for raw_sentence in sentences:
            # ----------------------------------------------------------
            # 1. Encode tokens → integer indices (drop OOV)
            # ----------------------------------------------------------
            sentence: list[int] = vocab.encode(raw_sentence)
            if len(sentence) < 2:
                continue

            # ----------------------------------------------------------
            # 2. Subsampling: discard frequent words stochastically
            # ----------------------------------------------------------
            if cfg.subsample_t > 0.0:
                keep_probs = vocab._keep_probs[sentence]
                keep_mask = rng.random(len(sentence)) < keep_probs
                sentence = [w for w, keep in zip(sentence, keep_mask) if keep]
                if len(sentence) < 2:
                    continue

            # ----------------------------------------------------------
            # 3. Slide window over the sentence
            # ----------------------------------------------------------
            length = len(sentence)
            for pos, center in enumerate(sentence):
                # Dynamic window: shrink uniformly from [1, window]
                if cfg.dynamic_window:
                    win = int(rng.integers(1, cfg.window + 1))
                else:
                    win = cfg.window

                left = max(0, pos - win)
                right = min(length - 1, pos + win)

                contexts = [
                    sentence[ctx_pos]
                    for ctx_pos in range(left, right + 1)
                    if ctx_pos != pos
                ]
                
                if contexts:
                    yield center, contexts

    def count_pairs(self, sentences: Iterable[list[str]]) -> int:
        """Estimate the total number of training pairs (for progress tracking)."""
        total = 0
        for center, contexts in self.iter_pairs(sentences):
            total += len(contexts)
        return total
