FROM python:3.11-slim

WORKDIR /app

# dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# copia dependências primeiro (cache do Docker)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" || pip install --no-cache-dir fastapi uvicorn python-dotenv langchain langchain-groq langchain-community langchain-ollama chromadb sentence-transformers mlflow xgboost scikit-learn pandas numpy shap joblib langfuse

# copia o código
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/
COPY data/raw/ ./data/raw/
COPY .env.example .env

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]