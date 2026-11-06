"""
api.py - Step 5: web service / 5. adım: web servisi

EN: Turns the RAG system into a web API with FastAPI.
    The models are loaded ONCE when the server starts; every question after that
    takes only the search + rerank + LLM time.
TR: RAG sistemini FastAPI ile bir web servisine çevirir.
    Modeller sunucu açılırken BİR KEZ yüklenir; sonraki her soru sadece
    arama + rerank + LLM süresi kadar sürer.

Run locally / Bilgisayarda çalıştır:
    uvicorn api:app --port 8000
    then open / sonra aç:  http://127.0.0.1:8000/docs
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from rag import RAG

state = {}   # EN: holds the loaded RAG object / TR: yüklenen RAG nesnesini tutar


@asynccontextmanager
async def lifespan(app: FastAPI):
    # EN: runs once at startup: load FAISS, embedder, reranker, Groq client
    # TR: açılışta bir kez çalışır: FAISS, embedder, reranker, Groq istemcisi yüklenir
    state["rag"] = RAG()
    yield
    state.clear()   # EN: runs at shutdown / TR: kapanışta çalışır


app = FastAPI(
    title="EU Regulation RAG API",
    description="Ask questions about the EU AI Act, GDPR and DORA. Answers cite the articles they use. "
                "Information, not legal advice.",
    version="1.0",
    lifespan=lifespan,
)


# EN: Request and response shapes. FastAPI checks them automatically and
#     returns error 422 if a request does not match.
# TR: İstek ve cevap şekilleri. FastAPI bunları otomatik kontrol eder ve
#     uymayan bir istekte 422 hatası döner.
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500,
                          examples=["Can a bank reject a loan using only an automated decision?"])
    use_rerank: bool = True


class Source(BaseModel):
    n: int          # EN: citation number [n] / TR: kaynak numarası [n]
    doc: str        # EN: AI Act / GDPR / DORA
    section: str    # EN: e.g. "Article 22" / TR: örn. "Article 22"
    title: str
    page: int


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    latency_ms: int


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")    # EN: open the test page / TR: test sayfasını aç


@app.get("/health")
def health():
    """EN: Is the server up and are the models loaded?  TR: Sunucu açık ve modeller yüklü mü?"""
    return {"status": "ok", "models_loaded": "rag" in state}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """EN: Answer one question with citations.  TR: Bir soruyu kaynaklarıyla cevaplar."""
    start = time.perf_counter()
    try:
        answer, sources = state["rag"].answer(req.question, use_rerank=req.use_rerank)
    except Exception as e:
        # EN: e.g. Groq rate limit or missing API key -> 503 "service unavailable"
        # TR: örn. Groq hız sınırı veya eksik API anahtarı -> 503 "servis kullanılamıyor"
        raise HTTPException(status_code=503, detail=f"LLM call failed: {type(e).__name__}")
    return AskResponse(
        question=req.question,
        answer=answer,
        sources=[Source(n=i, doc=s["doc"], section=s["section"], title=s["title"], page=s["page"])
                 for i, s in enumerate(sources, 1)],
        latency_ms=int((time.perf_counter() - start) * 1000),
    )