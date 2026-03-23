import sys
from pathlib import Path

from word2vec_numpy import Word2Vec

# --- KONFIGURACJA (zmieniaj poniższe wartości według potrzeb) ---
MODEL_PATH = "model.npz"   # Ścieżka do zapisanego pliku modelu
WORD = "woman"              # Słowo, którego wektor chcesz wyświetlić (lub None)
SIMILAR_WORD = "bike"      # Słowo, dla którego szukasz najbliższych sąsiadów (lub None)
TOP_N = 10                 # Liczba wyświetlanych podobnych słów
# ----------------------------------------------------------------

def main():
    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        print(f"Błąd: Plik modelu '{model_path}' nie istnieje.")
        sys.exit(1)

    print(f"Wczytywanie modelu z {model_path}...")
    model = Word2Vec.load(model_path)
    
    vocab_size = model.vocab.vocab_size
    dim = model.W.shape[1]
    print(f"Model wczytany pomyślnie!")
    print(f"Rozmiar słownika: {vocab_size} słów")
    print(f"Wymiar wektorów: {dim}\n")

    # Wyświetlenie wektora określonego słowa
    if WORD:
        print(f"--- Wektor dla słowa: '{WORD}' ---")
        if WORD in model.vocab:
            vec = model[WORD]
            print(vec)
            print(f"(rozmiar: {vec.shape})")
        else:
            print("Słowo nie znajduje się w słowniku modelu.")
        print()

    # Szukanie najbardziej podobnych słów
    if SIMILAR_WORD:
        print(f"--- Top {TOP_N} słów podobnych do: '{SIMILAR_WORD}' ---")
        if SIMILAR_WORD in model.vocab:
            neighbors = model.most_similar(SIMILAR_WORD, n=TOP_N)
            for word, score in neighbors:
                print(f"  {word:20s} -> {score:.4f}")
        else:
            print("Słowo nie znajduje się w słowniku modelu.")
        print()

if __name__ == "__main__":
    main()
