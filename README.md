# EU Regulation RAG Assistant

[![Docker build](https://github.com/Mervecaliskann/eu-regulation-rag/actions/workflows/docker.yml/badge.svg)](https://github.com/Mervecaliskann/eu-regulation-rag/actions/workflows/docker.yml)

**English** · [Türkçe](README.tr.md)

A RAG (retrieval-augmented generation) system that answers questions about three EU regulations and cites the article it used:

- **EU AI Act** (Regulation 2024/1689, 144 pages)
- **GDPR** (Regulation 2016/679, 88 pages)
- **DORA**, digital operational resilience for the financial sector (Regulation 2022/2554, 79 pages)

The main goal was to **measure** every step, not only build it: how often the right article is found, whether a reranker helps, and whether the answers stay faithful to the sources.

## Example output

```text
Q: What is the right to be forgotten?

The "right to be forgotten" (right of erasure) allows a data subject to obtain from a controller
the deletion of personal data concerning them without undue delay, when certain conditions are met
such as the data no longer being necessary, consent being withdrawn, the subject objecting to
processing, unlawful processing, or a legal obligation to erase the data [3]. The controller must
then erase the data and, if it has made the data public, take reasonable steps ... to inform other
controllers processing that data to remove any links, copies or replications [5]. The right does
not apply where erasure would conflict with freedom of expression, legal obligations, public-health
or research interests ... [5].

Sources:
  [1] GDPR - Article 17 - Right to erasure ('right to be forgotten') (p. 43)
  [2] GDPR - Recital 66 (p. 13)
  [3] GDPR - Article 17 - Right to erasure ('right to be forgotten') (p. 43)
  [4] GDPR - Recital 65 (p. 12)
  [5] GDPR - Article 17 - Right to erasure ('right to be forgotten') (p. 43)
```

Out-of-scope question:

```text
Q: What is the VAT rate for restaurants in Germany?
I could not find this in the provided regulations.
```

All 35 test answers with their judge verdicts are in [`eval_results.csv`](eval_results.csv).

## How it works

```text
PDF (AI Act, GDPR, DORA)
  -> ingest.py       clean text, split into recitals / articles / annexes  -> chunks.jsonl (1,264 chunks)
  -> build_index.py  embed every chunk (bge-small-en-v1.5, 384 numbers)    -> index.faiss
Question
  -> rag.py          1. search : FAISS finds the 20 closest chunks (bi-encoder)
                     2. rerank : a cross-encoder reads question + chunk together, keeps the best 5
                     3. answer : LLM answers ONLY from these 5 chunks, with [n] citations
  -> eval.py         35 test questions: Hit@1, Recall@5, Recall@20, MRR, faithfulness, refusals
```

## What each file does

### `ingest.py`: PDF → chunks
Reads each PDF with pypdf, drops page headers, page numbers and footnotes, splits the text into **recitals** (the "why" paragraphs before the articles), **articles** and **annexes**, and cuts long articles into pieces.

```python
HEADING = re.compile(r"^(Article (\d+)|ANNEX ([IVXLC]+))\s*$", re.MULTILINE)  # a line that is ONLY "Article 12"
if int(h.group(2)) == next_art:      # accept only the next expected number (1, 2, 3 ...)
    heads.append(h); next_art += 1
```
Output: 1,264 chunks. AI Act 180 recitals / 113 articles / 13 annexes, GDPR 173 / 99 / 0, DORA 104 / 64 / 0.

### `build_index.py`: chunks → vectors
Turns each chunk into a vector of 384 numbers that represents its meaning, and stores all vectors in a FAISS index.

```python
vectors = model.encode(texts, batch_size=32, normalize_embeddings=True)  # (1264, 384)
index = faiss.IndexFlatIP(384)     # inner product on normalized vectors = cosine similarity
index.add(vectors)
```

### `rag.py`: question → answer
```python
candidates = search(question)                   # 20 chunks from FAISS
best5 = rerank(question, candidates)            # cross-encoder scores each (question, chunk) pair
answer = llm(SYSTEM_PROMPT, best5, question)    # "use ONLY the numbered sources, cite [n],
                                                #  say 'I could not find this' if missing"
```

### `eval.py`: measure
For each test question, FAISS returns 20 candidates **once**; the same 20 are ranked two ways (FAISS order vs reranker order), so the only difference is the reranker. Then the LLM answers from the reranked top 5, and a second model (judge) checks if every claim is supported by the sources.

### `api.py`: web service
FastAPI wrapper around `rag.py`; see [Web API](#web-api-fastapi).

## Technologies and settings

| Part | Choice | Value | Why |
|---|---|---|---|
| PDF parsing | pypdf + regex | | Text-based PDFs; headers and footnotes removed with patterns |
| Chunking | by article / recital / annex | max 1,500 chars, 200 overlap | An article is the natural unit of a law. 1,500 chars ≈ 330 tokens, inside the model's 512-token limit. Overlap keeps the context of a split sentence. |
| Chunk header | `GDPR - Article 17 - Right to erasure` | | Lets search find an article by its title |
| Embeddings | `BAAI/bge-small-en-v1.5` | 384 dims, normalized | Free, runs on a CPU laptop (no GPU) |
| Vector index | FAISS `IndexFlatIP` | exact search | 1,264 vectors (~1.9 MB): exact search takes milliseconds |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 20 → 5 | Reads question and chunk together; small (~90 MB) for low RAM |
| LLM | `openai/gpt-oss-120b` (Groq) | temperature 0 | Repeatable answers; grounded prompt with citations and a refusal sentence |
| Judge | `openai/gpt-oss-20b` (Groq) | | A different model checks the answers, so the model does not grade itself |
| API | FastAPI + uvicorn, Docker | | Models loaded once at startup; automatic request validation and docs |

## Key concepts

| Concept | Meaning |
|---|---|
| Chunk | A small searchable piece of a document |
| Embedding | A fixed-length list of numbers (here 384) representing meaning; similar texts get close vectors |
| Cosine similarity | Similarity of two vectors by their angle (1 = same direction) |
| Bi-encoder | Question and chunk are embedded separately and compared; fast |
| Cross-encoder (reranker) | Question and chunk are read together into one relevance score; slower, more precise |
| Grounding | The LLM may only use the given sources |
| Faithfulness | Share of answers whose every claim is supported by the sources |
| Recall@k / Hit@1 | Is the right article in the top k / ranked first? |
| MRR | Mean of 1/rank of the right article (1st = 1, 2nd = 0.5, 3rd = 0.33) |

## Results

Test set: 35 questions ([`test_questions.json`](test_questions.json)). 32 have a known correct article (13 AI Act, 12 GDPR, 7 DORA) and 3 are out of scope.

| Retrieval metric | FAISS only | FAISS + reranker |
|---|---|---|
| Hit@1 (right article ranked first) | 53.1% (17/32) | **59.4%** (19/32) |
| Recall@5 (right article among the 5 chunks the LLM sees) | 84.4% (27/32) | **93.8%** (30/32) |
| Recall@20 (right article among the 20 candidates) | 100% | 100% |
| MRR | 0.665 | **0.722** |

| Answer metric (reranker, top 5) | Result |
|---|---|
| Faithfulness (every claim supported by the sources, LLM judge) | 87.5% (28/32) |
| Correct refusals on out-of-scope questions | 3/3 |

Note: the citation cleanup in `rag.py` (`[1†L1-L3]` → `[1]`) was added after this run, so some answers in `eval_results.csv` still show the raw markers. It changes only the text format, not retrieval or the scores.

## Findings

- **The reranker helps most when FAISS ranks badly:** data portability 20th → 1st, automated loan decision 9th → 2nd, ICT risk responsibility 12th → 2nd.
- **It also made some questions worse:** high-risk classification 1st → 5th, legal bases for processing 3rd → 7th. A likely reason: this reranker was trained on web search data, not legal text.
- **Faithful is not the same as correct.** For *"maximum fine for a prohibited AI practice"* the right article (AI Act Art. 99: up to EUR 35 million or 7% of turnover) was ranked 7th and never reached the LLM. The LLM answered "EUR 1.5 million" from Art. 100, which covers fines for EU institutions. The judge marked it faithful because it matched the sources it was given. A faithfulness check cannot catch retrieval errors, so both retrieval and answer metrics are needed.
- **Judge failures** were small additions not in the sources (for example "a dated record" of consent). One answer cited `[7]` although only 5 sources were given.
- **PDF cleaning mattered:** a reference "... in accordance with Article 49" wrapped onto its own line and was first read as a new article (114 instead of 113). A sequence check fixed it.

## Web API (FastAPI)

`api.py` turns the pipeline into a web service. The models are loaded **once** when the server starts; each question then takes about 1-3 seconds.

```bash
uvicorn api:app --port 8000
# open http://127.0.0.1:8000/docs to try it in the browser
```

| Endpoint | What it does |
|---|---|
| `POST /ask` | `{"question": "...", "use_rerank": true}` → answer, numbered sources, latency |
| `GET /health` | Is the server up and are the models loaded? |
| `GET /docs` | Interactive test page generated by FastAPI |

Requests are validated automatically (question must be 3-500 characters, otherwise `422`); if the LLM call fails (for example a rate limit) the API returns `503` instead of crashing.

Same question, with and without the reranker:

| | `use_rerank: true` | `use_rerank: false` |
|---|---|---|
| Sources | **GDPR Article 22**, GDPR Recital 71, AI Act ... | only AI Act recitals |
| Answer | Allowed only under a GDPR exception, with the right to human intervention and to contest the decision | "Yes, it is high-risk under the AI Act" (misses GDPR Art. 22) |
| Latency | ~3.0 s | ~1.0 s |

The reranker costs about 2 seconds on CPU but finds the article that actually answers the question.

A `Dockerfile` is included (CPU-only torch, models downloaded at build time, port 7860).  A GitHub Actions workflow builds the image on every push and smoke-tests the running API (`/health` and input validation).

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

1. Download the PDFs from EUR-Lex into `data/` as `ai_act.pdf`, `gdpr.pdf`, `dora.pdf`:
   [AI Act](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689) ·
   [GDPR](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679) ·
   [DORA](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32022R2554)
2. Create a `.env` file with `GROQ_API_KEY=...` (never commit it).
3. Run:

```bash
python ingest.py
python build_index.py
python rag.py "Can a bank reject a loan using only an automated decision?"
python eval.py --retrieval   # retrieval metrics only, no API calls
python eval.py               # full evaluation (~10-15 min on the Groq free tier)
```

`chunks.jsonl` and `index.faiss` are included, so `rag.py` and `eval.py` work without the first two steps.

## Limits

- Small test set (32 in-scope questions): one question changes Recall@5 by about 3 points. The results show a direction, not a final number.
- I wrote the test questions and matched the expected articles myself; some questions could also be answered by other articles.
- 104 of DORA's 106 recitals were detected; the text of the other two is merged into the previous recital.
- pypdf sometimes joins words (`withoutundue`).
- The documents are the original published texts; later amendments are not included.
- English only. This is information, not legal advice.

## Next steps

- Hybrid search (BM25 + vectors with Reciprocal Rank Fusion) for exact terms and numbers such as the fine amount
- A larger or legal-domain reranker (for example `bge-reranker-base`) and compare
- Deploy the API (Docker) to a public host
- Check that every citation number exists in the given sources

Source documents: © European Union, [EUR-Lex](https://eur-lex.europa.eu), reused under the Commission's reuse policy.
