"""
eval.py - Step 4: evaluation / 4. adım: ölçüm

EN: Measures the RAG system on a fixed test set (test_questions.json).
    Part A - retrieval: is the right article found, and at which rank?
             FAISS order (no reranker) vs reranked order, on the SAME 20 candidates.
             Metrics: Hit@1, Recall@5, Recall@20, MRR.
    Part B - answers: the LLM answers from the reranked top 5; a second LLM (judge)
             checks if every claim is supported by the sources (faithfulness).
             Out-of-scope questions must get "I could not find this ...".
TR: RAG sistemini sabit bir test setiyle (test_questions.json) ölçer.
    A bölümü - retrieval: doğru madde bulunuyor mu, kaçıncı sırada?
             FAISS sırası (reranker'sız) ile reranker sırası, AYNI 20 aday üzerinde.
             Metrikler: Hit@1, Recall@5, Recall@20, MRR.
    B bölümü - cevaplar: LLM reranker'ın seçtiği 5 parçadan cevap verir; ikinci bir
             LLM (hakem) her iddianın kaynaklarda olup olmadığını kontrol eder
             (faithfulness). Kapsam dışı sorular "I could not find this ..." almalı.

Run / Çalıştır:
    python eval.py                 (A + B, ~70 Groq calls, ~10-15 min on the free tier)
    python eval.py --retrieval     (only A, no API calls / sadece A, API çağrısı yok)
"""
import csv
import json
import re
import sys
import time
from pathlib import Path

from groq import RateLimitError

from rag import RAG, TOP_K_FINAL, TOP_K_SEARCH

TEST_FILE = Path("test_questions.json")
OUT_CSV = Path("eval_results.csv")         # EN: one row per question / TR: her soru için bir satır
OUT_SUMMARY = Path("eval_summary.json")    # EN: final numbers / TR: özet sayılar
JUDGE_MODEL = "openai/gpt-oss-20b"         # EN: smaller model as judge / TR: hakem olarak daha küçük model
NOT_FOUND = "could not find this"          # EN: refusal sentence from rag.py / TR: rag.py'deki "bulamadım" cümlesi

JUDGE_PROMPT = """You check an answer against its sources.
Sources:
{sources}

Answer:
{answer}

Is EVERY factual claim in the answer supported by the sources?
Reply with JSON only: {{"supported": true or false, "reason": "<one short sentence>"}}"""


def with_retry(fn, *args, tries=8):
    """EN: Groq free tier allows ~8,000 tokens per minute per model. If we hit the
           limit (error 429), wait the time Groq suggests (or 15 s) and try again.
       TR: Groq ücretsiz katmanı model başına dakikada ~8.000 token'a izin veriyor.
           Sınıra takılırsak (hata 429), Groq'un önerdiği süre (ya da 15 sn) bekleyip
           tekrar dene."""
    for attempt in range(tries):
        try:
            return fn(*args)
        except RateLimitError as e:
            m = re.search(r"try again in ([\d.]+)s", str(e))
            wait = float(m.group(1)) + 1 if m else 15
            print(f"      rate limit, waiting {wait:.0f}s (attempt {attempt + 1}/{tries})")
            time.sleep(wait)
    raise RuntimeError("Groq rate limit: too many retries")


def first_relevant_rank(ranked, expected):
    """EN: 1-based rank of the first chunk whose doc+section is expected; None if missing.
       TR: Beklenen doc+bölüme ait ilk parçanın sırası (1'den başlar); yoksa None."""
    for i, c in enumerate(ranked, 1):
        if f"{c['doc']} | {c['section']}" in expected:
            return i
    return None


def retrieval_metrics(ranks):
    """EN: ranks = list of first-relevant ranks (None = not found).
       TR: ranks = ilk doğru parçanın sıraları listesi (None = bulunamadı)."""
    n = len(ranks)
    return {
        "hit@1": sum(r == 1 for r in ranks if r) / n,                     # EN: right one is first / TR: doğru parça 1. sırada
        f"recall@{TOP_K_FINAL}": sum(r <= TOP_K_FINAL for r in ranks if r) / n,  # EN: in the LLM's 5 / TR: LLM'e giden 5'te
        f"recall@{TOP_K_SEARCH}": sum(r is not None for r in ranks) / n,  # EN: in the 20 candidates / TR: 20 adayda
        "mrr": sum(1 / r for r in ranks if r) / n,                        # EN: 1st=1, 2nd=0.5, 3rd=0.33 ... / TR: 1.=1, 2.=0,5 ...
    }


def judge(rag, answer, sources):
    """EN: Ask a second LLM if the answer is supported. TR: İkinci LLM'e cevabın desteklenip desteklenmediğini sor."""
    src = "\n\n".join(f"[{i}] {s['text']}" for i, s in enumerate(sources, 1))
    resp = rag.client.chat.completions.create(
        model=JUDGE_MODEL, temperature=0,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(sources=src, answer=answer)}],
    )
    text = resp.choices[0].message.content
    m = re.search(r"\{.*\}", text, re.DOTALL)              # EN: take the JSON part / TR: JSON kısmını al
    try:
        data = json.loads(m.group(0))
        return bool(data.get("supported")), data.get("reason", "")
    except (AttributeError, json.JSONDecodeError):
        return None, "judge output not parsed: " + text[:100]


def main():
    retrieval_only = "--retrieval" in sys.argv
    tests = json.loads(TEST_FILE.read_text(encoding="utf-8"))
    rag = RAG()                                             # EN: load models ONCE / TR: modelleri BİR KEZ yükle

    rows, ranks_faiss, ranks_rerank = [], [], []
    faithful, refusals_ok, in_scope_n, out_scope_n = 0, 0, 0, 0

    for i, t in enumerate(tests, 1):
        q, expected = t["question"], set(t["expected"])
        candidates = rag.search(q)                          # EN: 20 chunks in FAISS order / TR: FAISS sırasıyla 20 parça
        reranked = rag.rerank(q, [dict(c) for c in candidates], k=TOP_K_SEARCH)  # EN: same 20, new order / TR: aynı 20, yeni sıra
        row = {"id": i, "question": q, "expected": "; ".join(sorted(expected))}

        if expected:                                        # EN: in-scope question / TR: kapsam içi soru
            r_f, r_r = first_relevant_rank(candidates, expected), first_relevant_rank(reranked, expected)
            ranks_faiss.append(r_f)
            ranks_rerank.append(r_r)
            row.update(rank_faiss=r_f, rank_rerank=r_r)

        if not retrieval_only:
            top5 = reranked[:TOP_K_FINAL]
            answer = with_retry(rag.generate, q, top5)
            refused = NOT_FOUND in answer.lower()
            row["answer"] = answer.replace("\n", " ")
            if expected:
                in_scope_n += 1
                ok, reason = with_retry(judge, rag, answer, top5)
                faithful += bool(ok)
                row.update(faithful=ok, judge_reason=reason, refused=refused)
            else:
                out_scope_n += 1
                refusals_ok += refused                     # EN: must refuse / TR: reddetmeli
                row.update(refused=refused)
            time.sleep(5)                                   # EN: stay under Groq rate limits / TR: Groq hız sınırı için

        rows.append(row)
        print(f"{i:2d}/{len(tests)}  faiss={row.get('rank_faiss', '-')!s:>4}  rerank={row.get('rank_rerank', '-')!s:>4}  {q[:60]}")

    summary = {
        "n_questions": len(tests),
        "n_in_scope": len(ranks_faiss),
        "faiss_only": retrieval_metrics(ranks_faiss),
        "faiss_plus_reranker": retrieval_metrics(ranks_rerank),
    }
    if not retrieval_only:
        summary["faithfulness"] = faithful / max(in_scope_n, 1)
        summary["correct_refusals"] = f"{refusals_ok}/{out_scope_n}"

    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + json.dumps(summary, indent=2))
    print(f"\nSaved {OUT_CSV} and {OUT_SUMMARY}")


if __name__ == "__main__":
    main()