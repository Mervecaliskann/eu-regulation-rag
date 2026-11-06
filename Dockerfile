# EN: Small official Python image / TR: Küçük resmi Python imajı
FROM python:3.10-slim

# EN: Hugging Face Spaces runs containers as user 1000; create it
# TR: Hugging Face Spaces konteyneri 1000 numaralı kullanıcıyla çalıştırır; onu oluştur
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_DISABLE_XET=1
WORKDIR /home/user/app

# EN: CPU-only torch first (~200 MB instead of ~2.5 GB with CUDA), then the rest
# TR: Önce sadece CPU'lu torch (CUDA'lı ~2,5 GB yerine ~200 MB), sonra diğerleri
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# EN: Download both models at build time, so the server starts fast
# TR: İki modeli imaj oluşturulurken indir; sunucu hızlı açılsın
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# EN: Code and the prebuilt index / TR: Kod ve hazır indeks
COPY --chown=user rag.py api.py chunks.jsonl index.faiss index_meta.json ./

# EN: Port 7860 is what Hugging Face Spaces expects / TR: Hugging Face Spaces 7860 portunu bekler
EXPOSE 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]