"""
build_index.py - Step 2 of the RAG pipeline / RAG sürecinin 2. adımı

EN: Turns every chunk in chunks.jsonl into a vector (embedding) and stores all
    vectors in a FAISS index, so we can find the chunks closest to a question.
TR: chunks.jsonl içindeki her parçayı bir vektöre (embedding) çevirir ve tüm
    vektörleri bir FAISS indeksine kaydeder; böylece bir soruya en yakın
    parçaları bulabiliriz.

Run / Çalıştır:  python build_index.py
"""
import json                                        # EN: read chunks       / TR: parçaları okumak için
from pathlib import Path

import faiss                                       # EN: fast vector search / TR: hızlı vektör arama
import numpy as np                                 # EN: arrays            / TR: diziler
from sentence_transformers import SentenceTransformer  # EN: embedding model / TR: embedding modeli

CHUNKS_FILE = Path("chunks.jsonl")
INDEX_FILE = Path("index.faiss")                   # EN: saved vectors     / TR: kaydedilen vektörler
META_FILE = Path("index_meta.json")                # EN: model name + chunk order / TR: model adı + parça sırası

# EN: Small English embedding model (~130 MB). Runs on CPU, no GPU needed.
# TR: Küçük İngilizce embedding modeli (~130 MB). CPU'da çalışır, GPU gerekmez.
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# EN: bge models work better when the QUESTION (not the chunks) gets this prefix.
# TR: bge modelleri, SORUNUN (parçaların değil) başına bu ifade eklenince daha iyi çalışır.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_chunks():
    """EN: Read chunks.jsonl into a list of dicts.  TR: chunks.jsonl'i sözlük listesine okur."""
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    model = SentenceTransformer(MODEL_NAME)        # EN: downloads once to HF_HOME / TR: bir kez HF_HOME'a iner

    # EN: normalize_embeddings=True makes every vector length 1, so the dot
    #     product of two vectors equals their cosine similarity.
    # TR: normalize_embeddings=True her vektörün uzunluğunu 1 yapar; böylece iki
    #     vektörün iç çarpımı kosinüs benzerliğine eşit olur.
    vectors = model.encode(
        [c["text"] for c in chunks],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    print("Vector shape:", vectors.shape)          # EN: (number of chunks, 384) / TR: (parça sayısı, 384)

    # EN: IndexFlatIP = exact search with inner product (= cosine here).
    #     With ~1,300 chunks exact search is fast enough; no approximation needed.
    # TR: IndexFlatIP = iç çarpımla (burada = kosinüs) birebir arama.
    #     ~1.300 parça için birebir arama yeterince hızlı; yaklaşık yönteme gerek yok.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_FILE))

    # EN: Row i of the index = chunk i. We save the ids in the same order.
    # TR: İndeksteki i. satır = i. parça. Kimlikleri aynı sırayla kaydediyoruz.
    META_FILE.write_text(json.dumps({"model": MODEL_NAME, "ids": [c["id"] for c in chunks]}), encoding="utf-8")
    print(f"Saved {INDEX_FILE} and {META_FILE}")

    # --- Quick test / Hızlı test ---------------------------------------------
    questions = [
        "Is an AI system used for credit scoring considered high-risk?",
        "What is the right to be forgotten?",
        "How quickly must a bank report a major ICT-related incident?",
    ]
    for q in questions:
        qv = model.encode([QUERY_PREFIX + q], normalize_embeddings=True).astype(np.float32)
        scores, rows = index.search(qv, 3)         # EN: top 3 closest chunks / TR: en yakın 3 parça
        print(f"\nQ: {q}")
        for score, row in zip(scores[0], rows[0]):
            c = chunks[row]
            print(f"  {score:.3f}  {c['doc']} - {c['section']} - {c['title'][:60]}")


if __name__ == "__main__":
    main()