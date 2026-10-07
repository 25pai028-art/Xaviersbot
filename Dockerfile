# Xavier's Assistant on a Hugging Face Docker Space (or any container host).
# The answering AI is remote (Groq), so the container only runs search (bge-m3, ChromaDB, BM25) and the web app.
# No crawling here: the knowledge base comes from a private dataset (see scripts/space_start.py).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    SCHEDULER_ENABLED=false \
    ENVIRONMENT=production

# Hugging Face runs containers as user 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY --chown=user app ./app
COPY --chown=user scripts ./scripts

USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    DATA_DIR=/home/user/data

EXPOSE 7860
CMD ["python", "-m", "scripts.space_start"]
