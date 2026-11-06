"""
rag.py - Step 3 of the RAG pipeline / RAG sürecinin 3. adımı

EN: Answers a question about the EU AI Act, GDPR and DORA:
    1) search  : FAISS finds the 20 chunks closest to the question (bi-encoder)
    2) rerank  : a cross-encoder reads question + chunk together and keeps the best 5
    3) generate: an LLM (Groq) writes a short answer using ONLY those 5 chunks,
                 with citations like [1], [2]
TR: AB AI Act, GDPR ve DORA hakkındaki bir soruyu cevaplar:
    1) arama     : FAISS soruya en yakın 20 parçayı bulur (bi-encoder)
    2) reranking : cross-encoder soru ve parçayı birlikte okuyup en iyi 5'ini seçer
    3) üretim    : bir LLM (Groq) SADECE bu 5 parçayı kullanarak kısa bir cevap
                   yazar ve [1], [2] gibi kaynak gösterir

Run / Çalıştır:
    python rag.py "Is credit scoring AI high-risk?"
    python rag.py "Is credit scoring AI high-risk?" --no-rerank     (compare / karşılaştır)
"""
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv                       # EN: reads .env file / TR: .env dosyasını okur
from groq import Groq                                # EN: LLM API client  / TR: LLM API istemcisi
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()                                        # EN: puts GROQ_API_KEY into the environment / TR: GROQ_API_KEY'i ortama yükler

CHUNKS_FILE = Path("chunks.jsonl")
INDEX_FILE = Path("index.faiss")
META_FILE = Path("index_meta.json")

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # EN: same as build_index.py / TR: build_index.py ile aynı

# EN: Small cross-encoder (~90 MB), trained to score (question, passage) pairs.
# TR: Küçük cross-encoder (~90 MB), (soru, paragraf) çiftlerine alaka puanı vermek için eğitilmiş.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "openai/gpt-oss-120b"                    # EN: Groq model / TR: Groq modeli

TOP_K_SEARCH = 20   # EN: candidates from FAISS            / TR: FAISS'ten gelen aday sayısı
TOP_K_FINAL = 5     # EN: chunks given to the LLM          / TR: LLM'e verilen parça sayısı

SYSTEM_PROMPT = """You answer questions about the EU AI Act, GDPR and DORA.
Rules:
- Use ONLY the numbered sources below. Do not use outside knowledge.
- After every claim, cite the source number in brackets, e.g. [1] or [2][3].
- If the sources do not contain the answer, reply exactly:
  "I could not find this in the provided regulations."
- Keep the answer short: at most 6 sentences, plain English.
- This is information, not legal advice."""


class RAG:
    def __init__(self):
        # EN: Load everything once; answering many questions is then fast.
        # TR: Her şeyi bir kez yükle; sonra çok soru cevaplamak hızlı olur.
        with CHUNKS_FILE.open(encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(INDEX_FILE))
        self.embedder = SentenceTransformer(meta["model"])   # EN: same model as the index / TR: indeksle aynı model
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.client = Groq()                                  # EN: uses GROQ_API_KEY / TR: GROQ_API_KEY'i kullanır

    def search(self, question, k=TOP_K_SEARCH):
        """EN: Bi-encoder search: question vector vs all chunk vectors.
           TR: Bi-encoder arama: soru vektörü tüm parça vektörleriyle karşılaştırılır."""
        qv = self.embedder.encode([QUERY_PREFIX + question], normalize_embeddings=True).astype(np.float32)
        scores, rows = self.index.search(qv, k)
        return [dict(self.chunks[r], score=float(s)) for s, r in zip(scores[0], rows[0])]

    def rerank(self, question, candidates, k=TOP_K_FINAL):
        """EN: Cross-encoder reads (question, chunk) TOGETHER -> more accurate, slower.
           TR: Cross-encoder (soru, parça) çiftini BİRLİKTE okur -> daha doğru, daha yavaş."""
        pairs = [(question, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)                 # EN: one relevance score per pair / TR: her çift için bir alaka puanı
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:k]

    def retrieve(self, question, use_rerank=True):
        """EN: search (+ rerank) -> final chunks.  TR: arama (+ rerank) -> son parçalar."""
        candidates = self.search(question)
        return self.rerank(question, candidates) if use_rerank else candidates[:TOP_K_FINAL]

    def generate(self, question, sources):
        """EN: Ask the LLM to answer from the numbered sources only.
           TR: LLM'den sadece numaralı kaynaklara dayanarak cevap vermesini ister."""
        context = "\n\n".join(f"[{i}] {s['text']}" for i, s in enumerate(sources, 1))
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,                                    # EN: same input -> same output / TR: aynı girdi -> aynı çıktı
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"},
            ],
        )
        text = response.choices[0].message.content.strip()
        # EN: gpt-oss sometimes writes citations as 【1】; make them [1]
        # TR: gpt-oss bazen kaynakları 【1】 diye yazıyor; [1] yap
        return text.replace("\u3010", "[").replace("\u3011", "]")

    def answer(self, question, use_rerank=True):
        sources = self.retrieve(question, use_rerank)
        return self.generate(question, sources), sources


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_rerank = "--no-rerank" not in sys.argv
    question = args[0] if args else "Is an AI system used for credit scoring considered high-risk?"

    rag = RAG()
    text, sources = rag.answer(question, use_rerank)

    print(f"\nQ: {question}\n")
    print(text)
    print("\nSources / Kaynaklar:")
    for i, s in enumerate(sources, 1):
        print(f"  [{i}] {s['doc']} - {s['section']} - {s['title'][:60]} (p. {s['page']})")


if __name__ == "__main__":
    main()