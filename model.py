"""
word2vec_numpy/model.py
------------------------
High-level Word2Vec model: owns the embedding matrices, drives the
training loop, and exposes a clean public API.
"""

from __future__ import annotations

import time

from pathlib import Path
from typing import Iterable

import numpy as np

from .config import Word2VecConfig
from .data import Corpus, sentences_from_file, sentences_from_list
from .trainers.base import BaseTrainer
from .trainers.skipgram_ns import SkipGramNegativeSampling
from .utils import most_similar, LinearDecaySchedule
from .vocabulary import Vocabulary


class Word2Vec:
    """
    Word2Vec model with pluggable trainers.

    Parameters
    ----------
    config:
        Hyperparameter configuration object.  Defaults to
        ``Word2VecConfig()`` (skip-gram, negative sampling, 100-dim).

    Examples
    --------
    >>> cfg = Word2VecConfig(embed_dim=100, epochs=5, negative=5)
    >>> model = Word2Vec(cfg)
    >>> model.train("path/to/corpus.txt")
    >>> vec = model["king"]                      # numpy array
    >>> model.most_similar("king", n=5)          # [(word, score), ...]
    >>> model.analogy("man", "king", "woman")    # [("queen", score), ...]
    >>> model.save("model.npz")
    """

    def __init__(self, config: Word2VecConfig | None = None) -> None:
        self.config = config or Word2VecConfig()
        self.vocab: Vocabulary | None = None
        self.W: np.ndarray | None = None    # input embeddings  (syn0)
        self.W_: np.ndarray | None = None   # output embeddings (syn1neg)
        self._rng = np.random.default_rng(self.config.seed)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        source: str | Path | list[list[str]],
        *,
        verbose: bool = True,
        log_every: int = 10_000,
    ) -> "Word2Vec":
        """
        Train the model on a corpus.

        Parameters
        ----------
        source:
            Either a path to a whitespace-tokenised text file, or an
            in-memory list of tokenised sentences.
        verbose:
            Print progress (loss and words/sec) during training.
        log_every:
            Print a log line after this many training pairs.

        Returns
        -------
        self  (for method chaining)
        """
        cfg = self.config
        sentences_fn = self._make_sentence_fn(source)

        # ----------------------------------------------------------
        # Phase 1: Build vocabulary
        # ----------------------------------------------------------
        if verbose:
            print("Building vocabulary …")
        t0 = time.perf_counter()
        self.vocab = Vocabulary(cfg).build(sentences_fn())
        if verbose:
            elapsed = time.perf_counter() - t0
            print(
                f"  Vocab size : {self.vocab.vocab_size:,} words "
                f"(min_count={cfg.min_count})"
            )
            print(f"  Total tokens : {self.vocab.total_tokens:,}")
            print(f"  Built in {elapsed:.1f}s")

        # ----------------------------------------------------------
        # Phase 2: Initialise embedding matrices
        # ----------------------------------------------------------
        V, D = self.vocab.vocab_size, cfg.embed_dim
        # Uniform initialisation in [-0.5/D, 0.5/D] (matches C word2vec)
        self.W = self._rng.uniform(-0.5 / D, 0.5 / D, (V, D)).astype(np.float32)
        self.W_ = np.zeros((V, D), dtype=np.float32)

        # ----------------------------------------------------------
        # Phase 3: Build trainer
        # ----------------------------------------------------------
        trainer = self._build_trainer()

        # ----------------------------------------------------------
        # Phase 4: Training loop
        # ----------------------------------------------------------
        corpus = Corpus(self.vocab, cfg, self._rng)
        total_tokens = self.vocab.total_tokens
        total_steps = total_tokens * cfg.epochs
        schedule = LinearDecaySchedule(cfg.learning_rate, cfg.min_lr, total_steps)

        global_words = 0
        global_pairs = 0
        t_start = time.perf_counter()

        for epoch in range(1, cfg.epochs + 1):
            epoch_loss = 0.0
            pair_count = 0
            interval_loss = 0.0
            interval_pairs = 0
            
            for raw_sentence in sentences_fn():
                # Process the sentence
                sentence = self.vocab.encode(raw_sentence)
                if len(sentence) < 2:
                    global_words += len(sentence)
                    continue

                if cfg.subsample_t > 0.0:
                    keep_probs = self.vocab._keep_probs[sentence]
                    keep_mask = self._rng.random(len(sentence)) < keep_probs
                    sentence = [w for w, keep in zip(sentence, keep_mask) if keep]
                    if len(sentence) < 2:
                        global_words += len(raw_sentence)
                        continue

                length = len(sentence)
                for pos, center in enumerate(sentence):
                    if cfg.dynamic_window:
                        win = int(self._rng.integers(1, cfg.window + 1))
                    else:
                        win = cfg.window

                    left = max(0, pos - win)
                    right = min(length - 1, pos + win)
                    contexts = [sentence[ctx_pos] for ctx_pos in range(left, right + 1) if ctx_pos != pos]

                    if not contexts:
                        continue

                    ctx_arr = np.array(contexts, dtype=np.int32)
                    lr = schedule.get(global_words)
                    
                    loss = trainer.train_pair(center, ctx_arr, lr)
                    
                    n_contexts = len(contexts)
                    epoch_loss += loss * n_contexts
                    pair_count += n_contexts
                    interval_loss += loss * n_contexts
                    interval_pairs += n_contexts
                    global_pairs += n_contexts

                global_words += len(raw_sentence)

                if verbose and interval_pairs >= log_every:
                    elapsed = time.perf_counter() - t_start
                    wps = global_pairs / elapsed if elapsed > 0 else 0
                    avg = interval_loss / interval_pairs
                    progress = 100 * global_words / max(total_steps, 1)
                    print(
                        f"  Epoch {epoch}/{cfg.epochs}  "
                        f"step {global_pairs:,}  "
                        f"lr {lr:.5f}  "
                        f"loss {avg:.4f}  "
                        f"progress {progress:.1f}%  "
                        f"{wps:,.0f} pairs/s"
                    )
                    interval_loss = 0.0
                    interval_pairs = 0

            if verbose:
                avg_loss = epoch_loss / max(pair_count, 1)
                print(f"Epoch {epoch}/{cfg.epochs} done — avg loss {avg_loss:.4f}")

        if verbose:
            total_time = time.perf_counter() - t_start
            print(f"Training complete in {total_time:.1f}s")

        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __getitem__(self, word: str) -> np.ndarray:
        """Return the embedding vector for ``word``."""
        self._check_trained()
        if word not in self.vocab:
            raise KeyError(f"'{word}' not in vocabulary")
        return self.W[self.vocab[word]].copy()

    def most_similar(self, word: str, n: int = 10) -> list[tuple[str, float]]:
        """Return the ``n`` most similar words by cosine similarity."""
        self._check_trained()
        return most_similar(word, self.W, self.vocab.word2idx, self.vocab.idx2word, n)

    # def analogy(
    #     self,
    #     word_a: str,
    #     word_b: str,
    #     word_c: str,
    #     n: int = 5,
    # ) -> list[tuple[str, float]]:
    #     """
    #     Evaluate the analogy ``word_b - word_a + word_c ≈ ?``.
    #
    #     Example: ``model.analogy("man", "king", "woman")`` should return
    #     something like ``[("queen", 0.72), …]``.
    #     """
    #     self._check_trained()
    #     return analogy(
    #         word_a, word_b, word_c,
    #         self.W, self.vocab.word2idx, self.vocab.idx2word,
    #         n,
    #     )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save the model to a compressed NumPy archive (``.npz``).

        Saves both embedding matrices and the vocabulary.
        """
        self._check_trained()
        path = Path(path)
        np.savez_compressed(
            path,
            W=self.W,
            W_=self.W_,
            idx2word=np.array(self.vocab.idx2word),
            counts=self.vocab.counts,
        )
        print(f"Model saved to {path}.npz")

    @classmethod
    def load(cls, path: str | Path, config: Word2VecConfig | None = None) -> "Word2Vec":
        """
        Load a model from a ``.npz`` archive written by ``save()``.

        Parameters
        ----------
        path:
            Path to the archive (with or without ``.npz`` extension).
        config:
            Optional config override.  If None, a default config is used.
        """
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".npz")

        data = np.load(path, allow_pickle=True)
        model = cls(config)

        model.W = data["W"]
        model.W_ = data["W_"]

        idx2word: list[str] = data["idx2word"].tolist()
        counts: np.ndarray = data["counts"]

        cfg = model.config
        vocab = Vocabulary(cfg)
        vocab.idx2word = idx2word
        vocab.word2idx = {w: i for i, w in enumerate(idx2word)}
        vocab.counts = counts
        vocab._compute_keep_probs()
        vocab._build_unigram_table()
        model.vocab = vocab

        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        if self.W is None or self.vocab is None:
            raise RuntimeError("Model has not been trained yet.  Call train() first.")

    def _make_sentence_fn(self, source: str | Path | list[list[str]]):
        """Return a zero-argument callable that yields sentences."""
        if isinstance(source, (str, Path)):
            return lambda: sentences_from_file(source)
        else:
            return lambda: sentences_from_list(source)

    def _build_trainer(self) -> BaseTrainer:
        cfg = self.config
        if cfg.model == "skipgram" and cfg.loss == "negative_sampling":
            return SkipGramNegativeSampling(
                self.W, self.W_, self.vocab, cfg, self._rng
            )
        raise NotImplementedError(
            f"Trainer for model='{cfg.model}', loss='{cfg.loss}' is not yet implemented."
        )
