# Performance Optimization
## Amazon Review Sentiment & ABSA System

---

## Current Baseline

| Scenario | P50 Latency | P99 Latency | Throughput (req/s) | RAM |
|---|---|---|---|---|
| Full pipeline, transformer path | ~1.8s | ~5.8s | 2-5 | ~1.3GB |
| Full pipeline, LLM path | ~5.5s | ~12s | 0.5-1 | ~3.3GB |
| Single sentiment only (/api/sentiment, no DB write) | ~1.6s | ~5.2s | 3-6 | ~1.3GB |
| Batch of 10 (transformer, sequential) | ~18s | ~58s | 0.5 | ~1.3GB |

These numbers assume CPU-only inference on a modern laptop/server CPU. GPU would 10-50x the transformer latency.

---

## Optimization 1: TensorFlow Configuration

### `TF_ENABLE_ONEDNN_OPTS=0`

Set in `app.py` before TF import:
```python
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
```

oneDNN (Intel's Deep Neural Network Library) is TensorFlow's default backend on Intel CPUs. On many CPUs it produces warnings about "oneDNN custom operations" that spam logs on every inference call. Disabling it removes the log noise. Performance impact on most hardware: negligible (< 5% inference speed difference).

**When to enable:** If running on modern Intel CPUs with AVX-512 support, oneDNN can improve BiLSTM inference by 10-20%. Test with your specific hardware. For production, benchmark both settings.

### `compile=False` on Model Load

```python
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
```

Without `compile=False`, Keras reconstructs the optimizer (Adam's moment estimates, gradient accumulators, learning rate state). These are used for training — during inference they are never accessed. For a BiLSTM model, optimizer state adds approximately 50-100MB RAM and 2-3 seconds to load time.

**Memory savings:** ~50-100MB RAM eliminated.
**Startup savings:** ~2-3 seconds at initialization.
**Risk:** None. `model.predict()` does not use optimizer state.

### `verbose=0` on `model.predict()`

```python
pred = float(model.predict(padded, verbose=0)[0][0])
```

Without `verbose=0`, Keras prints a progress bar to stdout for every prediction:
```
1/1 [==============================] - 0s 58ms/step
```
This I/O operation adds ~20-50ms per prediction on some systems and pollutes logs in production.

**Savings:** 20-50ms per inference call eliminated.
**Benefit:** Clean production logs.

---

## Optimization 2: Model Loading Strategy — Eager vs Lazy

### Current: Eager Loading (at startup)

```python
# All models loaded at app startup
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
_bert_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
_model = T5ForConditionalGeneration.from_pretrained("t5-base")
```

**Pros:**
- Zero cold start per request — first request as fast as the 100th
- Model loading errors fail fast at startup (no user-facing error on first request)
- All memory allocated upfront — system operator knows the RAM requirement

**Cons:**
- 30-60 second startup time before the first request can be served
- Full RAM allocated from startup even if only some models are used
- Each gunicorn worker loads all models independently: N workers × 1.3GB RAM

**When eager loading is right:** Production systems where startup latency is acceptable and per-request latency must be minimal. This is the correct choice for most serving scenarios.

### Alternative: Lazy Loading (on first request)

```python
_model = None  # deferred

def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model
```

**Pros:**
- Near-instant startup (process ready in <5 seconds)
- Only loads models that are actually used in that worker's lifetime
- Lower RAM for workers that handle only lightweight routes (e.g., GET /api/stats)

**Cons:**
- First request to any route has 30-60 second latency spike — user-facing cold start
- Multiple concurrent requests arriving before model loads all trigger loading simultaneously (race condition without a lock)
- Model loading errors surface as 500 responses to real users

**Recommendation:** Eager loading for production, lazy loading for development only. If startup time is critical (e.g., Kubernetes liveness probes have short timeouts), consider a separate model warm-up container that loads models before the main container starts.

---

## Optimization 3: Redis Result Caching

**Current state:** No caching. Every call to POST /analyze runs the full pipeline, even for reviews that have been analyzed before. The database deduplication (hash check → skip INSERT) prevents *storage* duplicates but does not prevent *computation* duplicates.

**Proposed caching layer:**

```python
import redis
_redis = redis.Redis(host="localhost", port=6379, db=0)
CACHE_TTL = 7 * 24 * 3600  # 7 days

def complete_pipeline_cached(review: str) -> dict:
    key = f"result:{hash_review(review)}"
    cached = _redis.get(key)
    if cached:
        return json.loads(cached)
    
    result = complete_pipeline(review)
    _redis.setex(key, CACHE_TTL, json.dumps(result))
    return result
```

**Expected impact:**
- Same review analyzed twice: full pipeline avoided on second call (1.8s → ~5ms Redis read)
- For popular products (iPhone, Kindle, AirPods): high review text overlap across users
- Estimated cache hit rate in e-commerce context: 40-80% for top 1000 products
- Overall throughput improvement: 2-5× under realistic review distribution

**Memory cost:** Worst case — 10,000 unique cached results × ~2KB per result = 20MB Redis memory. Negligible.

**Note:** Redis caching prevents computation; the DB deduplication layer prevents storage. These are complementary and both necessary.

---

## Optimization 4: DistilBERT Aspect Batching

### Current: Sequential per-aspect calls

```python
# In get_aspect_sentiment_improved() via get_aspect_context()
for aspect in aspects:
    sentiment, _ = get_aspect_sentiment_improved(review, aspect)
    # Each calls analyze_sentiment_with_transformer() → one DistilBERT forward pass
```

For a review with 5 aspects, this is 5 sequential DistilBERT calls. Each call:
1. Tokenize the clause (~5ms)
2. Forward pass through DistilBERT (~130ms on CPU)
3. Softmax → score (~1ms)

Total: 5 × ~136ms = ~680ms for 5 aspects.

### Proposed: Batched DistilBERT call

```python
def batch_score_aspects(contexts: list[str]) -> list[float]:
    """Score all aspect contexts in one DistilBERT forward pass."""
    inputs = _bert_tokenizer(
        contexts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,  # pad to same length within batch
    ).to(torch.device("cpu"))
    
    with torch.no_grad():
        logits = _bert_model(**inputs).logits
        scores = torch.softmax(logits, dim=1)[:, 1].tolist()
    
    return scores
```

**Latency with batching:**
- Tokenize all clauses (~5ms × 1 batch = 5ms)
- One DistilBERT forward pass with batch_size=5 (~180ms — not 5× because matrix operations are parallelized)
- Total: ~185ms vs ~680ms sequential

**Speedup:** ~3.7× for 5 aspects. More pronounced for reviews with more aspects.

**Caveat:** Batching requires all contexts to be padded to the same length within the batch. Long contexts pad shorter ones. For reviews with one very long context and several short ones, the padding overhead reduces the batching benefit.

---

## Optimization 5: Ollama-Specific Optimization

### `num_predict` Token Cap

```python
# In absa_with_llm.py
options={"temperature": 0, "num_predict": 400}  # ABSA

# In summary_with_llm.py  
options={"temperature": 0.3, "num_predict": 150}  # Summary
```

`num_predict` caps the maximum tokens generated. Without this, llama3.2 may generate verbose explanations:

*Without num_predict=400 (ABSA):* Model might output 800 tokens of explanation before or after the aspect list. Generation is linear in token count — 800 tokens = ~5 seconds on CPU.

*With num_predict=400:* Hard stop at 400 tokens. Aspect lists for typical reviews are 5-15 lines × ~8 tokens = 40-120 tokens. 400 is generous enough to never cut valid output but prevents runaway generation.

*With num_predict=150 (summary):* 60-word summary = ~80 tokens. 150 is a 2× safety margin. Prevents the model from generating a paragraph when asked for 2-3 sentences.

**Savings:** 30-50% latency reduction for cases where the model would otherwise generate verbose output.

### `temperature=0` Eliminates Sampling Overhead

Temperature=0 uses greedy decoding (always select the highest-probability token). This skips the sampling step (no need to compute probabilities and sample from the distribution). Minor CPU savings, but primarily a correctness optimization rather than a performance optimization.

---

## Optimization 6: Memory Reduction via Half-Precision

### Current State: float32 everywhere

All models load in float32 (the PyTorch and TensorFlow default):
- DistilBERT: ~260MB (float32)
- T5-base: ~850MB (float32)

### Proposed: float16 for CPU inference

```python
# For PyTorch models (DistilBERT, T5)
from transformers import AutoModelForSequenceClassification
_bert_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    torch_dtype=torch.float16,
)
```

**Memory savings:**
- DistilBERT float16: ~130MB (50% reduction)
- T5-base float16: ~425MB (50% reduction)
- Total transformer path: ~1.3GB → ~0.7GB

**Caveats:**
- float16 on CPU: PyTorch CPU doesn't have native float16 kernels on most architectures. It emulates float16 using float32 internally, so memory is saved but inference speed doesn't improve on CPU (may even decrease due to conversion overhead).
- float16 on GPU: native hardware support → 50% memory + 2× throughput. This is the primary use case for float16.
- Accuracy: minimal change (float16 has 3 significant decimal digits vs float32's 7). For sentiment classification (3 classes), this is negligible.

**Recommendation:** Implement float16 only if deploying with GPU. CPU-only deployment sees memory savings without speed improvement.

---

## Optimization 7: Async Processing with FastAPI

### Current Flask (synchronous):

A slow Ollama call (30 seconds on CPU) blocks one Flask worker thread for 30 seconds. With 4 gunicorn threads, 4 simultaneous slow requests block all workers.

### FastAPI with async Ollama calls:

```python
import httpx

async def call_ollama_async(review_text: str) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "http://localhost:11434/api/chat",
            json={"model": "llama3.2", "messages": [...], "stream": False}
        )
        return _parse_response(response.json()["message"]["content"])

@app.post("/analyze")
async def analyze(request: ReviewRequest):
    # This coroutine yields during the Ollama HTTP call
    # allowing other requests to be served concurrently
    result = await call_ollama_async(request.review)
    return result
```

**Benefit:** While waiting for Ollama (30 seconds), the async coroutine yields and other requests run. Effective concurrency scales to the event loop limit, not the thread count.

**Caveat:** Ollama itself is still single-process. Concurrent async requests all queue in Ollama — async helps the Python server not block, but doesn't speed up Ollama. True parallelism requires multiple Ollama instances.

---

## Optimization 8: Gunicorn Configuration

### Current (dev server):
```python
app.run(debug=True, host="0.0.0.0", port=5000)
```
Dev server: single-threaded, debug mode, not production-safe.

### Recommended gunicorn configuration:

```bash
gunicorn app:app \
  --workers 2 \
  --threads 4 \
  --bind 0.0.0.0:5000 \
  --timeout 120 \
  --keep-alive 5
```

**Why 2 workers:** Each worker loads all models independently. On a 4GB machine:
- 1 worker + Ollama: 1 × 1.3GB + 2GB = 3.3GB (comfortable with 4GB)
- 2 workers + Ollama: 2 × 1.3GB + 2GB = 4.6GB (tight but feasible with 8GB)
- 4 workers + Ollama: 4 × 1.3GB + 2GB = 7.2GB (requires 8GB minimum)

**Why 4 threads per worker:** Threads share the worker's memory (including loaded models). Flask route handlers that block on Ollama or T5 yield the GIL during I/O operations (HTTP call to Ollama, though GIL is held during TF inference). 4 threads allows 4 concurrent requests per worker with shared model memory. The GIL prevents true CPU parallelism for model inference (one thread actually runs model.predict at a time), but allows I/O overlap.

**`--timeout 120`:** Worker killed if a request takes longer than 120 seconds. Prevents Ollama hangs from killing the server permanently. A killed worker is automatically restarted by gunicorn.

---

## Optimization Priority Table

| Optimization | Implementation Effort | RAM Savings | Latency Improvement | Priority |
|---|---|---|---|---|
| `verbose=0` on model.predict | Trivial (done) | 0 | 20-50ms | Done |
| `compile=False` on load | Trivial (done) | 50-100MB | 2-3s startup | Done |
| `TF_ENABLE_ONEDNN_OPTS=0` | Trivial (done) | 0 | Log cleanup | Done |
| `num_predict` caps in Ollama | Trivial (done) | 0 | 30-50% LLM | Done |
| `temperature=0` for ABSA | Trivial (done) | 0 | Correctness | Done |
| DistilBERT aspect batching | Medium (1-2 days) | 0 | 3-4× ABSA speed | High |
| Redis result caching | Medium (1 day) | 0 | 2-5× throughput | High |
| Middle-truncation | Low (2 hours) | 0 | Accuracy improvement | High |
| Float16 model loading (GPU) | Medium (1 day) | 50% | 2× (GPU only) | Medium (GPU only) |
| Async FastAPI migration | High (1 week) | 0 | Concurrency improvement | Medium |
| Ollama timeout guard | Low (2 hours) | 0 | Reliability | High |
| Background LLM health monitor | Medium (1 day) | 0 | Reliability | Medium |

---

## Memory Optimization Summary

| Component | Current (float32) | With float16 | Notes |
|---|---|---|---|
| BiLSTM | ~15MB | ~8MB | Keras, minimal savings |
| spaCy en_core_web_sm | ~50MB | ~50MB | Not float16 applicable |
| DistilBERT SST-2 | ~260MB | ~130MB | PyTorch, GPU benefit |
| T5-base | ~850MB | ~425MB | PyTorch, GPU benefit |
| llama3.2 (Ollama) | ~2GB | ~1GB (Q4 quant) | Ollama supports quantization |
| **Total (transformer path)** | **~1.3GB** | **~0.7GB (GPU)** | |
| **Total (LLM path)** | **~3.3GB** | **~2.0GB** | |

**Ollama quantization:** Ollama supports loading quantized models (4-bit, 8-bit). `ollama pull llama3.2:latest` pulls the Q4_K_M quantized version by default (~2GB). The full float16 version is ~4GB. The default quantized model is already the optimized choice.
