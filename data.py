"""
word2vec_numpy/data.py
-----------------------
Corpus streaming and (center, context) training-pair generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

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
