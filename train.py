#!/usr/bin/env python
"""
train.py
--------
Command-line entry point for training a word2vec model.

Downloads the first 10 MB of text8 automatically if no corpus is given,
then trains a skip-gram model with negative sampling and prints nearest
neighbours for a set of probe words.

Usage
-----
    python train.py                          # use text8 excerpt (default)
    python train.py --corpus path/to/file    # use your own corpus
    python train.py --epochs 3 --embed-dim 100 --negative 10
    python train.py --help
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
import io
import time
from pathlib import Path

from word2vec_numpy.model import Word2Vec, Word2VecConfig


# ---------------------------------------------------------------------------
# text8 dataset helpers
# ---------------------------------------------------------------------------

TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
DEFAULT_CACHE = Path("data/text8")
DEFAULT_MAX_BYTES = int(0.5 * 1024 * 1024)


def download_text8(cache_path: Path = DEFAULT_CACHE, max_bytes: int = DEFAULT_MAX_BYTES) -> Path:
    """
    Download text8.zip and extract the first ``max_bytes`` bytes into
    ``cache_path``.  Does nothing if the file already exists.

    Returns
    -------
    Path to the cached text file.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f"Using cached corpus: {cache_path}")
        return cache_path

    print(f"Downloading text8 from {TEXT8_URL} …")
    t0 = time.perf_counter()
    with urllib.request.urlopen(TEXT8_URL) as response:
        raw = response.read()
    elapsed = time.perf_counter() - t0
    print(f"  Downloaded {len(raw) / 1e6:.1f} MB in {elapsed:.1f}s")

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        text = zf.read("text8")  # bytes

    excerpt = text[:max_bytes].decode("utf-8")

    with cache_path.open("w", encoding="utf-8") as f:
        f.write(excerpt)

    print(f"  Saved {len(excerpt) / 1e6:.1f} MB excerpt to {cache_path}")
    return cache_path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a word2vec model (skip-gram + negative sampling) in pure NumPy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Corpus
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to a whitespace-tokenised text file.  "
             "Defaults to a 10 MB excerpt of text8 (auto-downloaded).",
    )
    parser.add_argument(
        "--text8-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="Number of bytes to extract from text8 (only used if --corpus is not set).",
    )

    # Architecture
    parser.add_argument("--embed-dim", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--negative", type=int, default=20)

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--subsample-t", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=12)

    # Output
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save the trained model to this path (e.g. model.npz).",
    )
    parser.add_argument(
        "--save-epochs",
        action="store_true",
        help="If set, saves a checkpoint after every epoch using the --save filename prefix.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an existing .npz model to resume training from.",
    )
    parser.add_argument(
        "--probe-words",
        type=str,
        default="king,runs,seven",
        help="Comma-separated list of words to inspect after training.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve corpus path
    if args.corpus is not None:
        corpus_path = Path(args.corpus)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    else:
        corpus_path = download_text8(max_bytes=args.text8_bytes)

    # Build config
    cfg = Word2VecConfig(
        embed_dim=args.embed_dim,
        window=args.window,
        negative=args.negative,
        epochs=args.epochs,
        learning_rate=args.lr,
        min_lr=args.min_lr,
        min_count=args.min_count,
        subsample_t=args.subsample_t,
        seed=args.seed,
        model="cbow",
        loss="hs"
    )

    print("\n=== word2vec-numpy ===")
    print(f"  model     : {cfg.model}")
    print(f"  loss      : {cfg.loss}")
    print(f"  embed_dim : {cfg.embed_dim}")
    print(f"  window    : {cfg.window} (dynamic={cfg.dynamic_window})")
    print(f"  negative  : {cfg.negative}")
    print(f"  epochs    : {cfg.epochs}")
    print(f"  lr        : {cfg.learning_rate} -> {cfg.min_lr}")
    print(f"  corpus    : {corpus_path}")
    if args.resume:
        print(f"  resume    : {args.resume}")
    print()

    # Train
    if args.resume:
        model = Word2Vec.load(args.resume, config=cfg)
    else:
        model = Word2Vec(cfg)
        
    save_epoch_path = args.save if args.save_epochs else None
    
    model.train(
        corpus_path, 
        verbose=True, 
        resume=bool(args.resume),
        save_epoch_path=save_epoch_path
    )

    # Save
    if args.save:
        model.save(args.save)

    # Probe nearest neighbours
    probe_words = [w.strip() for w in args.probe_words.split(",")]
    print("\n=== Nearest neighbours ===")
    for word in probe_words:
        if word not in model.vocab:
            print(f"  '{word}' not in vocabulary — skipped")
            continue
        neighbours = model.most_similar(word, n=20)
        neighbours_str = ", ".join(f"{w} ({s:.3f})" for w, s in neighbours)
        print(f"  {word:15s}->  {neighbours_str}")


if __name__ == "__main__":
    main()
