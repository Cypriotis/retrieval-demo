FROM python:3.12-slim
WORKDIR /service
ENV PYTHONUNBUFFERED=1 HF_HOME=/models TOKENIZERS_PARALLELISM=false
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
COPY reranker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY reranker/main.py .
COPY app/passages.py ./passages.py
RUN mkdir -p /models && useradd --uid 10001 --create-home reranker && chown reranker:reranker /models
USER reranker
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
