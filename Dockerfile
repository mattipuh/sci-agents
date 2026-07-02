FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)"

COPY . .

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
