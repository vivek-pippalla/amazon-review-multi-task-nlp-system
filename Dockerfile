# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder — install heavy dependencies in a throwaway layer
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU-only first (CPU wheel ≈300 MB vs ≈2.5 GB for the CUDA build).
# Must run before requirements.txt so pip sees torch as already satisfied.
RUN pip install --no-cache-dir --prefix=/install \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies (torch already present, will be skipped)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt \
 && pip install --no-cache-dir --prefix=/install gunicorn

# Download the spaCy English model into the install prefix
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python -m spacy download en_core_web_sm --target /install/lib/python3.11/site-packages

# Pre-download HuggingFace models so runtime startup never hits the network.
# Without this, the app downloads ~1.15 GB at every first start and the
# health check times out before the server is ready.
ENV HF_HOME=/hf_cache
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python -c "\
from transformers import (\
    AutoTokenizer, AutoModelForSequenceClassification,\
    T5Tokenizer, T5ForConditionalGeneration\
);\
AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english');\
AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english');\
T5Tokenizer.from_pretrained('t5-base', legacy=False);\
T5ForConditionalGeneration.from_pretrained('t5-base');\
print('[build] HuggingFace models cached.')"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime — lean final image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="amazon-review-sentiment-absa"
LABEL description="Amazon Review Sentiment & ABSA — Flask inference server"

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy pre-downloaded HuggingFace model cache (avoids network calls at startup)
COPY --from=builder /hf_cache /hf_cache

# Runtime system libraries required by TensorFlow / PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application source
COPY app.py \
     absa.py \
     absa_with_llm.py \
     db_connection.py \
     llm_check.py \
     summary.py \
     summary_with_llm.py \
     ./

COPY templates/ templates/
COPY static/    static/

# Tell HuggingFace where the pre-cached models live
ENV HF_HOME=/hf_cache

# Suppress TensorFlow OneDNN verbose output
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_CPP_MIN_LOG_LEVEL=2

# Flask / Gunicorn production settings
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 5000

# Health-check: poke the home page; start-period covers TF model load time
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fs http://localhost:5000/ || exit 1

# One worker (each loads the full ML model); 4 threads handle concurrency
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
