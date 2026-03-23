# Word2Vec NumPy

A fully vectorized, pure-NumPy implementation of Word2Vec SGNS, SGHS, CBOW-NS, and CBOW-HS architectures, inspired by the original C implementation by Tomas Mikolov.

## Features

- **Four Architectures**: Skip-Gram and Continuous Bag of Words (CBOW), paired with either Negative Sampling (NS) or Hierarchical Softmax (HS).
- **Checkpointing**: Save models after every epoch and comfortably resume training at any point.
- **Pure Python/NumPy**

## Command Line Usage

The easiest way to invoke training is via the `train.py` script. By default, with no arguments, it dynamically downloads the first 10MB of the Text8 dataset and trains a standard Skip-Gram Negative Sampling model.

```bash
# Standard training on the default Text8 dataset (3 epochs, 100 dimensions)
python train.py --epochs 3 --embed-dim 100 --save model.npz

# Train on your own text corpus
python train.py --corpus dataset.txt --epochs 5 --embed-dim 300 --save my_word2vec.npz

# Train using Continuous Bag of Words (CBOW) with Hierarchical Softmax (HS)
# Note: You can edit config parameters in train.py (or via the Python API) for CBOW/HS.

# Save a checkpoint after every single epoch (`model_epoch_1.npz`, `model_epoch_2.npz`, ...)
python train.py --epochs 10 --save model.npz --save-epochs

# Resume training from an existing checkpoint
python train.py --resume model_epoch_2.npz --epochs 5 --save model_finetuned.npz
```

Once training completes, it will automatically probe the nearest neighbors of "king,runs,seven" to verify semantic clustering. You can change this using `--probe-words "apple,car,house"`.

## Python API Usage

You can deeply integrate `word2vec_numpy` into your own scripts using the `Word2Vec` and `Word2VecConfig` classes.

### 1. Training a Model from Scratch

```python
from word2vec_numpy.word2vec_numpy.model import Word2Vec
from word2vec_numpy.word2vec_numpy.model import Word2VecConfig

# Configure your architecture
cfg = Word2VecConfig(
    model="cbow",  # 'skipgram' or 'cbow'
    loss="hs",  # 'negative_sampling' or 'hs'
    embed_dim=100,  # Vector dimensionality
    window=5,  # Context half-window size
    epochs=5,  # Number of passes over the corpus
    min_count=5,  # Minimum word frequency
    learning_rate=0.05,
    min_lr=0.0001
)

# Initialize and train
model = Word2Vec(cfg)
model.train(
    "path/to/corpus.txt",
    verbose=True,
    save_epoch_path="checkpoints/model.npz"  # Automatically saves after each epoch
)

# Save the final model
model.save("final_model.npz")
```

### 2. Loading and Resuming Training
You can resume training from any saved `.npz` archive without rebuilding the vocabulary.

```python
from word2vec_numpy.word2vec_numpy.model import Word2Vec

# Load weights and vocabulary from disk
model = Word2Vec.load("checkpoints/model_epoch_2.npz")

# Resume training for more epochs
model.config.epochs = 3
model.train("path/to/corpus.txt", resume=True, verbose=True)

model.save("finetuned_model.npz")
```

### 3. Querying Embeddings (Inference)
Use `inspect_model.py` for command-line inference, or query the API directly:

```python
from word2vec_numpy.word2vec_numpy.model import Word2Vec

# Load trained model
model = Word2Vec.load("final_model.npz")

# Get a raw numpy vector
v_king = model["king"]
print(v_king.shape)  # (100,)

# Find nearest neighbors by cosine similarity
neighbors = model.most_similar("king", n=5)
for word, similarity in neighbors:
    print(f"{word}: {similarity:.4f}")
```
