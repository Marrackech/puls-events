FROM python:3.11-slim

WORKDIR /app

RUN useradd -m appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --timeout 300 --retries 5

COPY scripts/ ./scripts/
COPY api/ ./api/
COPY vectorstore/ ./vectorstore/
COPY data/ ./data/
COPY .env .

RUN chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
