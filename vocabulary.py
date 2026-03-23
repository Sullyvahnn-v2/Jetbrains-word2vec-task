"""
word2vec_numpy/vocabulary.py
-----------------------------
Vocabulary construction, subsampling, and the unigram noise table used
for negative sampling.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

from .config import Word2VecConfig


class Vocabulary:
    """Maps words to integer indices and encapsulates all corpus statistics."""

    # Special token placed at index 0; acts as a sentinel / padding value.
    UNKNOWN = "<unk>"

    def __init__(self, config: Word2VecConfig) -> None:
        self.config = config

        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []
        self.counts: np.ndarray = np.array([], dtype=np.int64)  # shape (V,)
        self._keep_probs: np.ndarray = np.array([], dtype=np.float32)
        self._unigram_table: np.ndarray = np.array([], dtype=np.int32)

    @property
    def vocab_size(self) -> int:
        return len(self.idx2word)

    @property
    def total_tokens(self) -> int:
        return int(self.counts.sum())

    def build(self, sentences: Iterable[list[str]]) -> "Vocabulary":
        """
        Build the vocabulary from an iterable of tokenised sentences.

        Steps
        -----
        1. Count all word occurrences.
        2. Drop words that appear fewer than ``min_count`` times.
        3. Sort by descending frequency.
        4. Assign integer indices.
        5. Pre-compute subsampling discard probabilities.
        6. Build the unigram noise table.

        Parameters
        ----------
        sentences:
            Iterable of token lists.  Each inner list is one sentence.

        Returns
        -------
        self  (for method chaining)
        """
        counter: Counter[str] = Counter()
        for sentence in sentences:
            counter.update(sentence)

        filtered = [
            (w, c) for w, c in counter.items()
            if c >= self.config.min_count
        ]
        filtered.sort(key=lambda x: x[1], reverse=True)

        self.idx2word = [w for w, _ in filtered]
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.counts = np.array([c for _, c in filtered], dtype=np.int64)

        self._compute_keep_probs()
        self._build_unigram_table()
        if self.config.loss == "hs":
            self._build_huffman_tree()
        return self

    def _build_huffman_tree(self) -> None:
        """
        Build the Huffman binary tree for Hierarchical Softmax.
        """
        V = self.vocab_size
        if V == 0:
            return

        count = np.zeros(V * 2 + 1, dtype=np.int64)
        binary = np.zeros(V * 2 + 1, dtype=np.int8)
        parent_node = np.zeros(V * 2 + 1, dtype=np.int32)

        count[:V] = self.counts
        count[V:] = int(1e15)

        pos1 = V - 1
        pos2 = V

        for a in range(V - 1):
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min1i = pos1
                    pos1 -= 1
                else:
                    min1i = pos2
                    pos2 += 1
            else:
                min1i = pos2
                pos2 += 1

            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min2i = pos1
                    pos1 -= 1
                else:
                    min2i = pos2
                    pos2 += 1
            else:
                min2i = pos2
                pos2 += 1

            count[V + a] = count[min1i] + count[min2i]
            parent_node[min1i] = V + a
            parent_node[min2i] = V + a
            binary[min2i] = 1

        self.point = []
        self.code = []

        for a in range(V):
            path = []
            code = []
            b = a
            while b != V * 2 - 2:
                code.append(binary[b])
                path.append(parent_node[b] - V) # Map internal nodes to [0, V-2]
                b = parent_node[b]
            self.point.append(np.array(path[::-1], dtype=np.int32))
            self.code.append(np.array(code[::-1], dtype=np.int8))

    def _compute_keep_probs(self) -> None:
        """
        Pre-compute per-word *keep* probability for frequent-word
        subsampling (Mikolov et al. 2013, §2.3).
        """
        if self.config.subsample_t <= 0.0:
            self._keep_probs = np.ones(self.vocab_size, dtype=np.float32)
            return

        total = self.counts.sum()
        freq = self.counts / total  # relative frequency, shape (V,)
        t = self.config.subsample_t
        ratio = t / np.maximum(freq, 1e-12)
        probs = np.sqrt(ratio) + ratio
        self._keep_probs = np.clip(probs, 0.0, 1.0).astype(np.float32)

    def get_keep_prob(self, idx: int) -> float:
        """Return the probability of *keeping* word ``idx`` during training."""
        return float(self._keep_probs[idx])

    def _build_unigram_table(self) -> None:
        """
        Build a large integer table where each word w occupies a number
        of slots proportional to f(w)^ns_exponent.
        """
        exponent = self.config.ns_exponent
        powered = np.power(self.counts.astype(np.float64), exponent)
        powered_sum = powered.sum()

        table_size = self.config.table_size
        table = np.empty(table_size, dtype=np.int32)

        # Build cumulative probability thresholds (CDF) for each word.
        cumsum = np.cumsum(powered) / powered_sum  # shape (V,)

        word_idx = 0
        for i in range(table_size):
            table[i] = word_idx
            # Advance to the next word while the table position exceeds its CDF
            while word_idx < self.vocab_size - 1 and (i + 1) / table_size >= cumsum[word_idx]:
                word_idx += 1

        self._unigram_table = table

    def sample_negatives(
        self,
        count: int,
        rng: np.random.Generator,
        exclude: int | None = None,
    ) -> np.ndarray:
        """
        Draw ``count`` negative sample indices from the noise distribution.

        Parameters
        ----------
        count:
            Number of negative samples to draw.
        rng:
            NumPy random generator for reproducibility.
        exclude:
            Word index that must not be returned (the positive sample).
            Duplicates are replaced with a fresh draw.

        Returns
        -------
        np.ndarray of shape (count,) with dtype int32.
        """
        indices = rng.integers(0, len(self._unigram_table), size=count)
        samples = self._unigram_table[indices]

        if exclude is not None:
            # Re-draw any slot that accidentally hit the positive word
            mask = samples == exclude
            while mask.any():
                replacement = rng.integers(0, len(self._unigram_table), size=mask.sum())
                samples[mask] = self._unigram_table[replacement]
                mask = samples == exclude

        return samples

    def __len__(self) -> int:
        return self.vocab_size

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx

    def __getitem__(self, word: str) -> int:
        return self.word2idx[word]

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert a list of tokens to a list of word indices, skipping OOV."""
        return [self.word2idx[t] for t in tokens if t in self.word2idx]

    @property
    def keep_probs(self):
        return self._keep_probs
