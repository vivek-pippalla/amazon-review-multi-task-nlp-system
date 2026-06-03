# Interview Preparation Guide
## Amazon Review Sentiment & ABSA Project

---

## 30-Second Pitch

> "I built an end-to-end NLP system that analyzes Amazon reviews using a BiLSTM for overall sentiment, aspect-based sentiment analysis using both traditional NLP and an LLM, and abstractive summarization. It stores results in MySQL, exposes a REST API with analytics endpoints, and automatically selects between LLM and transformer paths depending on what's available at runtime."

---

## 2-Minute Explanation

Start with the problem: Amazon product reviews are unstructured text. A simple star rating tells you nothing about *which* features users liked or disliked. This system extracts three things from every review: (1) an overall sentiment label (Positive/Negative/Neutral) with a confidence score, (2) aspect-based sentiment — which specific product features are mentioned and what users think of each, and (3) an abstractive summary of the review.

The architecture has four layers:

**ML Layer** — A BiLSTM trained on Amazon review data classifies overall sentiment. It outputs a probability score; the application layer maps this to three labels using confidence thresholds (>0.5 Positive, 0.15–0.5 Neutral, <0.15 Negative). This threshold design was intentional: the model was trained binary, so Neutral is carved from the low-confidence zone rather than being a trained class.

**NLP Layer** — Aspect-Based Sentiment Analysis (ABSA) runs via one of two paths selected at startup: if Ollama with llama3.2 is available, the LLM path is used; otherwise the system uses spaCy dependency parsing plus DistilBERT SST-2 fine-tuned on sentiment. This selection is done once at import time, not per request — so there is no per-request routing overhead.

**Persistence Layer** — MySQL stores every analyzed review with its summary, sentiment, confidence score, and analysis source. SHA-256 hashing prevents duplicate storage. Thread-local PyMySQL connections ensure correctness under Flask's multi-threaded request handling.

**API Layer** — Flask exposes five routes: POST /analyze (analyze + persist), POST /api/sentiment (analyze only, no DB write), POST /api/batch/analyze (up to 10 reviews), GET /api/stats (aggregate statistics), GET /api/aspects/trending (most-mentioned aspects over N days), and GET /api/reviews (paginated history).

Key engineering decisions I made: using `threading.local()` for database connections (PyMySQL is not thread-safe), SHA-256 hashing for deduplication (MySQL prefix indexes only check the first 255 bytes — broken for long reviews), `compile=False` when loading the Keras model (skips optimizer reconstruction which is unnecessary for inference and wastes memory), and `temperature=0` for LLM ABSA (classification tasks need determinism, unlike summarization which benefits from slight variation).

---

## Architecture Deep-Dive: Module by Module

### `app.py` — Application Entry Point and Orchestrator

**What it does:** Initializes all models at startup, defines all routes, orchestrates the complete_pipeline function.

**Why Flask over FastAPI:** Flask was chosen for simplicity and the team's familiarity with it. The synchronous model is acceptable for single-user or low-concurrency deployment; the main bottleneck is model inference, not I/O. FastAPI would offer async support and automatic OpenAPI docs, but adds complexity. For a research/demo system, Flask's lower ceremony is the right tradeoff.

**Pipeline selection at import time:**
```python
if LLM_AVAILABLE:
    from absa_with_llm import aspect_based_sentiment_llm as aspect_based_sentiment
    from summary_with_llm import generate_summary
    ANALYSIS_SOURCE = "llm"
else:
    from absa import aspect_based_sentiment_improved as aspect_based_sentiment
    from summary import generate_summary
    ANALYSIS_SOURCE = "transformer"
```
This is a static import — Python resolves it once when the module loads. The function reference `aspect_based_sentiment` points to whichever implementation was available at startup. No branching per request.

**`complete_pipeline()` flow:**
1. `clean_text()` — lowercase + strip non-alpha (must match training preprocessing exactly)
2. `tokenizer_obj.texts_to_sequences()` → `pad_sequences(maxlen=150)` → BiLSTM predict
3. Confidence threshold mapping → overall sentiment label
4. `generate_summary()` — T5 or LLM
5. `aspect_based_sentiment()` — transformer or LLM path

**Input validation:**
- `MAX_REVIEW_LENGTH = 2000`: prevents large payload attacks and runaway inference time
- `MAX_BATCH_SIZE = 10`: prevents batch abuse
- `.strip()` on all review text: rejects whitespace-only input
- `isinstance(review, str)` in batch: prevents type confusion if client sends non-string values

### `db_connection.py` — Thread-Safe Database Connection Manager

**The problem it solves:** Flask with its default Werkzeug server, and especially with gunicorn `--threads N`, runs multiple threads in a single process. PyMySQL connections are not thread-safe. If two threads share one connection, Thread A might issue a SELECT and Thread B's result arrives during Thread A's fetchone() — corrupting both queries.

**The solution:**
```python
_local = threading.local()

def get_connection():
    if not hasattr(_local, "db") or _local.db is None or not _local.db.open:
        _local.db = pymysql.connect(...)
    return _local.db
```
`threading.local()` is a dict-like object where each key is the thread's identity. `_local.db` in Thread 1 is a completely different variable from `_local.db` in Thread 2. Each thread gets its own connection, created lazily on first use.

**Connection lifecycle:**
- Created: lazily on first `get_connection()` call within a request
- Closed: via `@app.teardown_appcontext` which runs after every request regardless of success or failure
- The teardown ensures connections are never leaked even if the route handler raises an exception

### `llm_check.py` — LLM Availability Detection

**Three-stage startup check:**
1. Check if local Ollama is running and has llama3.2 (`GET /api/tags`)
2. If not running, attempt `ollama serve` and wait 10 seconds, then re-check
3. If local Ollama fails, attempt Docker Model Runner (`docker model pull ai/llama3.2`)
4. If all three fail: `LLM_AVAILABLE = False` → transformer path used

**Why this matters:** The application needs to know at startup which modules to import. Python's module system means you cannot easily hot-swap function references after import. The three-stage check gives the system every reasonable opportunity to find a working LLM before falling back.

### `absa.py` — Traditional NLP ABSA Pipeline

Five-strategy aspect extraction:
1. **Noun chunks** via spaCy: catches most product feature nouns
2. **Dependency parsing**: opinion verb → object noun; nsubj/dobj/pobj nouns
3. **Compound nouns**: "battery" + compound child "life" → "battery life"
4. **Regex patterns**: `\w+ quality`, `\w+ life`, `\w+ performance`, etc.
5. **Qualifier filtering**: removes standalone qualifiers when the main aspect is already present

Post-extraction deduplication: substring removal → character n-gram cosine similarity grouping (threshold 0.75) → longest-wins selection.

Sentiment scoring: try DistilBERT SST-2 on the aspect's clause (extracted by splitting on contrastive conjunctions like "but", "however", "although"). If DistilBERT unavailable, fall back to lexicon-based scoring with negation handling and intensity modifiers.

**Key fixes in this file:**
- `neutral_count` renamed from `neutral_indicators`: Python 3 classifies a name as local to a function if it is assigned anywhere in that function. Using `neutral_indicators = sum(...)` shadowed the module-level set, causing UnboundLocalError.
- Bracket markers removed from DistilBERT input: `[camera]` prefix added noise because SST-2 was never trained on bracket notation.
- "not"/"no" removed from `negative_indicators`: caused "not bad" to accumulate two negative signals and score very negative instead of mildly positive.
- `filter_and_deduplicate_aspects` fixed to return `[]` not `{}`: returning an empty dict caused type errors in the caller that expected a list.

### `absa_with_llm.py` — LLM-Based ABSA

**Why LLM for ABSA:** LLMs understand context, implication, sarcasm, and informal language that rule-based systems miss. "The thing that charges it keeps disconnecting" — a traditional system may not extract "charger" from "thing that charges it"; an LLM infers the referent.

**Prompt design:** Two-part prompt: system role defines what counts as an aspect and what to exclude (generic nouns, time references, opinion words); user role provides five worked examples then the actual review. Few-shot examples are the single most impactful prompt engineering technique for structured output tasks.

**Output parsing with strict regex:**
```python
_LINE_RE = re.compile(
    r"^([a-zA-Z][a-zA-Z0-9 \-/]{1,58}):\s*(Positive|Negative|Neutral)\s*$",
    re.IGNORECASE,
)
```
Any line not matching this pattern is silently discarded. This means preamble text like "Here are the aspects:" is ignored, as are hallucinated verbose descriptions.

**Retry + fallback chain:** Attempt 1 → if empty parse, Attempt 2 → if still empty, fallback to transformer ABSA. The model sometimes outputs a preamble on the first attempt but not the second — retry catches this without needing to change the prompt.

**`temperature=0`:** Classification task. The set of valid outputs (Positive/Negative/Neutral) is fixed. Any temperature above 0 introduces non-determinism — the model might swap Positive and Negative labels for identical reviews on different calls. This was observed in testing at temperature=0.3.

### `summary_with_llm.py` — LLM Summarization

**`temperature=0.3`:** Unlike ABSA (classification), summarization is a generative task. A temperature of 0 produces robotic, repetitive phrasing. 0.3 introduces controlled variation that makes summaries read more naturally while remaining factually grounded.

**`num_predict=150`:** caps output to ~60 words. Without this, the model sometimes generates several paragraphs. The system prompt says "Maximum 60 words" but LLMs do not always respect word count instructions precisely; the token cap is the hard enforcement mechanism.

### `summary.py` — T5 Summarization

**`num_beams=4`:** Beam search maintains 4 candidate sequences simultaneously. With `num_beams=2`, output tends to be a list of keywords: "good, fast, works well." With 4, the model produces complete sentences. At 6, quality improvement is marginal but compute cost increases.

**`no_repeat_ngram_size=3`:** T5 has a known failure mode where it repeats 2-3 word phrases: "works great, works great, works great." Setting this to 3 prevents any 3-gram from appearing twice in the output. Tested with 2 — still repeated pairs; 3 eliminates the problem.

**`length_penalty=2.0`:** Values >1.0 penalize longer sequences, encouraging conciseness. Without this, T5 pads output to `max_length=80`.

**Short review passthrough:** Reviews under 20 words are returned as-is. T5's compressor produces worse output than the source when there is almost nothing to compress.

---

## BiLSTM Choice: Why Not BERT?

| Dimension | BiLSTM | BERT-base |
|---|---|---|
| Model size | ~15MB | ~440MB |
| Inference time (CPU) | 50–150ms | 500–2000ms |
| Fine-tuning infra needed | No (trained from scratch) | Yes (HuggingFace Trainer) |
| Interpretability | Moderate (attention weights) | Low (12-layer black box) |
| Context window | 150 tokens (sufficient for most reviews) | 512 tokens |
| Training data size needed | 50K+ examples sufficient | Millions for fine-tuning |

For Amazon reviews (typically 20–300 words), a BiLSTM with 150-token window covers approximately the 85th percentile of review lengths. The BiLSTM reads the sequence forward and backward simultaneously (via the Bidirectional wrapper), so "amazing camera but terrible battery" captures both the positive and negative signals from both directions.

The key argument against BERT for this use case: BERT-base is 30x larger than the BiLSTM, 10-30x slower on CPU, and requires a fine-tuning pipeline to achieve comparable performance. For a single-developer project without GPU training infrastructure, BiLSTM is the pragmatic choice.

DistilBERT *is* used in this system — but only for aspect-level sentiment (a different task), where it is used inference-only with a pre-fine-tuned checkpoint (distilbert-base-uncased-finetuned-sst-2-english). No training required.

---

## Fallback Systems: Three Layers

**Layer 1: Startup LLM detection (`llm_check.py`)**
- Runs once at startup
- Three attempts: local Ollama → start Ollama → Docker Model Runner
- Result sets `LLM_AVAILABLE` which determines which modules are imported
- Why: static import decision is cleaner and faster than per-request branching

**Layer 2: Per-request retry in LLM modules**
- Both `absa_with_llm.py` and `summary_with_llm.py` retry once on empty/failed response
- Handles transient Ollama hiccups (process temporarily busy, brief timeout)
- Why: LLMs sometimes output preamble or empty responses on first attempt but not second

**Layer 3: Transformer fallback from LLM modules**
- If both LLM attempts fail: `absa_with_llm` imports and calls `aspect_based_sentiment_improved`
- If both LLM attempts fail: `summary_with_llm` imports and calls `t5_summary`
- Why: the API must always return a result; "LLM unavailable" should not surface to the user as an error

---

## Production Concerns Addressed

| Concern | Implementation |
|---|---|
| Thread safety | `threading.local()` DB connections |
| Connection leaks | `@teardown_appcontext` closes connection after every request |
| Large payload attacks | `MAX_REVIEW_LENGTH = 2000` |
| Batch abuse | `MAX_BATCH_SIZE = 10` |
| Duplicate data | SHA-256 hash + `INSERT IGNORE` double-layer guard |
| Error string stored as data | Fallback returns `"Summary not available."` not exception message |
| Partial batch failures | Per-item try/except in batch route — one failure doesn't abort batch |
| SQL injection | Parameterized queries throughout (`cursor.execute(query, (param,))`) |
| Model load errors | Startup-time `RuntimeError` for missing model files — fail fast |

---

## Scalability Limitations

1. **Single-process Ollama**: Ollama runs one inference at a time. Concurrent LLM requests queue inside the Ollama process. Under load, ABSA and summarization requests for different reviews wait on the same Ollama instance.

2. **In-process model loading**: Each gunicorn worker loads all models independently. With 4 workers: 4 × 1.3GB = 5.2GB RAM (transformer path). With LLM path: 4 × BiLSTM + 1 Ollama process = ~2.4GB + 2GB = 4.4GB.

3. **No async processing**: Flask is synchronous. A slow Ollama request (30+ seconds for long reviews on CPU) blocks the worker thread for that duration.

4. **MySQL single instance**: No read replicas. Analytics queries compete with write queries. For analytics-heavy workloads, pre-aggregation or a separate read replica is needed.

5. **No request timeout**: If Ollama hangs, the worker thread waits indefinitely. Production fix: add `timeout` parameter to `ollama.chat()` or wrap in `concurrent.futures.ThreadPoolExecutor` with a timeout.

---

## ML Tradeoffs

**BiLSTM vs BERT for overall sentiment:** BiLSTM is 30x smaller, 10x faster, no fine-tuning pipeline needed. Accuracy is lower on complex negation and sarcasm. For most straightforward Amazon reviews, the quality difference is small. BERT would be better for ambiguous reviews.

**spaCy+DistilBERT vs fine-tuned ABSA model:** A model fine-tuned specifically for ABSA (like BERT-ABSA or SemEval-trained models) would have significantly better aspect extraction recall. The current system's extraction relies on spaCy's noun chunks, which miss informal aspect references like "the thing that charges it" or "the wobbly part on top." The LLM path partially compensates for this via contextual inference.

**T5-base vs GPT for summarization:** T5-base is encoder-decoder, specifically designed for text-to-text tasks including summarization. GPT is decoder-only and better at open-ended generation. For constrained summarization (fixed-length, factual, no hallucination), T5-base's encoder-decoder architecture is the better fit. T5 was pre-trained with "summarize:" task prefix which aligns exactly with this use case.

**Rule-based lexicon vs neural for aspect sentiment:** Rule-based (lexicon + negation handling) is fast, explainable, and deterministic. It fails on: out-of-lexicon words, sarcasm, double negation, domain-specific language. Neural (DistilBERT) handles these better but adds 260MB RAM and 100-300ms latency per aspect call. The hybrid approach in this system uses DistilBERT when available and falls back to lexicon — best of both worlds at the cost of complexity.

---

## 20+ Interview Questions with Strong Answers

---

### Q1: "Walk me through the architecture of your project."

**Strong answer:** "The system has four layers. At the bottom is MySQL, which stores every analyzed review along with its summary, ABSA results, and a SHA-256 hash for deduplication. Above that is the ML layer — a BiLSTM model trained on Amazon review data gives the overall sentiment probability. Alongside it is the NLP layer — aspect-based sentiment analysis using either a traditional pipeline (spaCy dependency parsing + DistilBERT SST-2) or llama3.2 via Ollama, depending on what's available at startup. The API layer is Flask exposing five routes: single analysis, API-only analysis, batch analysis, statistics, trending aspects, and review history. The key design decision I'm most proud of is that the LLM/transformer choice is made once at startup — not per request — which means zero routing overhead per call."

---

### Q2: "Why did you choose Flask over FastAPI?"

**Strong answer:** "Flask was the right choice for this project's scale and purpose. The main bottleneck is model inference — BiLSTM runs in 50-150ms, DistilBERT in 100-300ms, T5 in 500-2000ms — not I/O latency. FastAPI's async advantage helps most when you have many I/O-bound operations like external API calls or database round-trips that can be overlapped. For CPU-bound model inference, async doesn't help. If I were scaling to handle hundreds of concurrent requests, I would migrate to FastAPI with background tasks for async LLM calls via `httpx.AsyncClient` to Ollama, but for this project Flask's simplicity was the right tradeoff."

---

### Q3: "How does your system handle concurrent requests?"

**Strong answer:** "Two main mechanisms. First, database connections: PyMySQL connections are not thread-safe — sharing one connection across threads causes cursor corruption where Thread A's fetchone gets Thread B's result. I use `threading.local()` to give each Flask thread its own isolated connection. The connection is created lazily on first use within a thread and closed by the `@teardown_appcontext` hook after every request. Second, model inference: TensorFlow and PyTorch models loaded in the same process are generally safe for multi-threaded read access since the weights are not modified during inference. However, Ollama is single-process — concurrent LLM requests queue inside it, which is a scalability bottleneck I acknowledge."

---

### Q4: "What is thread-local storage and why did you use it?"

**Strong answer:** "`threading.local()` creates a dict-like object where each key is implicitly the calling thread's identity. When Thread 1 writes `_local.db = connection1` and Thread 2 writes `_local.db = connection2`, these are completely separate values — reading `_local.db` from Thread 1 always returns `connection1`, from Thread 2 always returns `connection2`. I used it because PyMySQL's documentation explicitly states connections should not be shared across threads. Without thread-local storage, two concurrent requests could share one connection: Thread A sends a SELECT, Thread B sends an INSERT, and Thread A's cursor.fetchone() might retrieve Thread B's INSERT result ID instead of the SELECT data — silent data corruption. Thread-local storage is the standard solution for per-thread resources in Python web applications."

---

### Q5: "How does the BiLSTM model work?"

**Strong answer:** "The model has four components. First, an Embedding layer that maps each token index to a dense vector (learned end-to-end, not pre-trained). Second, a Bidirectional LSTM layer — the LSTM processes the sequence left-to-right and right-to-left simultaneously; the two hidden states are concatenated, so the output at each position has context from both directions. For sentiment classification, the final hidden state captures the whole review's meaning. Third, Dense + Dropout + Dense for classification: the first Dense layer with ReLU learns feature combinations; Dropout at 0.3 prevents overfitting by randomly zeroing activations during training; the final Dense with sigmoid outputs a probability between 0 and 1. At inference, I apply asymmetric thresholds: above 0.5 is Positive, 0.15–0.5 is Neutral, below 0.15 is Negative. The Neutral band is carved from the low-confidence zone because the model was trained binary — it has no concept of Neutral; I'm inferring it from uncertainty."

---

### Q6: "Why BiLSTM instead of a transformer like BERT?"

**Strong answer:** "Three reasons. Size: BiLSTM is approximately 15MB; BERT-base is 440MB. On a machine running DistilBERT (260MB) and T5-base (850MB) already, adding BERT would push RAM over 1.5GB for models alone. Speed: BiLSTM runs in 50-150ms on CPU; BERT runs in 500-2000ms. For an API that needs to respond in under a second, BERT's latency is problematic without GPU acceleration. Infrastructure: training the BiLSTM required only the standard Keras training loop — no HuggingFace Trainer, no fine-tuning pipeline, no cloud GPU time. For a project built and maintained solo, that simplicity matters. The quality tradeoff is real — BERT handles complex negation and sarcasm better — but for most straightforward Amazon reviews, BiLSTM accuracy is competitive."

---

### Q7: "How does your ABSA system work?"

**Strong answer:** "ABSA extracts which product features are mentioned and what users think of each. The transformer path has two stages. Stage one is aspect extraction: spaCy processes the review and extracts noun chunks, dependency-parsed subject/object nouns, compound noun phrases like 'battery life', and regex-matched patterns like '\w+ quality'. A five-pass filtering removes generic words ('product', 'thing', 'it'), short tokens, all-stopword phrases. Then deduplication: substring removal (keep 'battery life' not 'battery' when both appear), then character n-gram cosine similarity grouping to merge near-duplicates like 'audio' and 'audio quality'. Stage two is sentiment scoring: for each aspect, extract the clause it appears in (splitting on contrastive conjunctions like 'but', 'however'), run DistilBERT SST-2 on that clause, map the score to Positive/Negative/Neutral. The LLM path skips all this and sends the review directly to llama3.2 with a structured prompt and few-shot examples, then parses the response with a strict regex."

---

### Q8: "What's the difference between your two ABSA implementations?"

**Strong answer:** "The traditional path in `absa.py` uses spaCy for aspect extraction — it follows grammatical structure (noun chunks, dependency arcs, compound nouns) and regex patterns. Sentiment is scored by DistilBERT SST-2 on the extracted clause, falling back to a lexicon-based scorer with negation handling. It's deterministic, explainable, and about 300MB RAM. The LLM path in `absa_with_llm.py` sends the review to llama3.2 with a system prompt defining what counts as an aspect, five worked examples, and strict output format rules. The LLM understands implicit references ('the thing that charges it' → charger), sarcasm, and compound opinions. The output is parsed by a strict regex that accepts only lines matching `aspect: Positive|Negative|Neutral`. The LLM path is better for informal language and implicit references; the transformer path is better for formal reviews with clear noun phrases and is faster (300ms vs 1-5 seconds)."

---

### Q9: "How do you prevent duplicate reviews from being stored?"

**Strong answer:** "Two layers of protection. Application layer: before any INSERT, I compute `hashlib.sha256(review.strip().encode()).hexdigest()` — a 64-character hex string that uniquely represents the review content. I query `SELECT id FROM reviews WHERE review_hash = %s`. If a row exists, I return the existing ID and skip the INSERT entirely. Database layer: `INSERT IGNORE INTO absa_results` — if somehow two concurrent requests with the same review race past the application-layer check simultaneously (a race condition under very high load), the DB-level UNIQUE constraint on `(review_id, aspect)` silently discards the duplicate row rather than raising an error. Why SHA-256 instead of relying on a UNIQUE constraint on review_text? Because MySQL's UNIQUE index on a TEXT column uses only a prefix — by default 255 bytes. Two reviews that share the same first 255 characters but differ afterward would be treated as duplicates by the index but are actually different reviews."

---

### Q10: "What bugs did you find and fix in this project?"

**Strong answer:** "Eight bugs I can enumerate specifically. One: global `_db` PyMySQL connection replaced with `threading.local()` — concurrent requests were corrupting each other's DB state. Two: SHA-256 deduplication replacing the broken `UNIQUE(review_text(255))` prefix index — reviews longer than 255 characters could create duplicates. Three: `UnboundLocalError` in `analyze_clause_sentiment` — I used `neutral_indicators = sum(...)` which shadows the module-level set name; Python 3 treats any name assigned in a function as local throughout the function, even before the assignment; renamed to `neutral_count`. Four: `filter_and_deduplicate_aspects` returned `{}` (empty dict) instead of `[]` (empty list) when no aspects found — caller code expected a list. Five: error string stored as review summary — `summary_with_llm.py` was catching the exception and returning `str(e)` which got written to the DB; fixed to return `'Summary not available.'`. Six: duplicate ABSA rows on re-submission of the same review — fixed with hash check plus `INSERT IGNORE`. Seven: bracket markers `[camera]` in DistilBERT input — SST-2 was never trained on brackets and scored differently; removed. Eight: 'not'/'no' in `negative_indicators` causing double-negation — 'not bad' accumulated two negative signals and scored very negative; removed."

---

### Q11: "How does your fallback system work?"

**Strong answer:** "Three layers. At startup, `llm_check.py` runs a three-stage check: ping local Ollama API, try starting Ollama via subprocess, try Docker Model Runner. If any succeeds, `LLM_AVAILABLE = True` and the LLM modules are imported. If all fail, transformer modules are imported instead — this is a static one-time decision. The second layer is per-request retry inside the LLM modules: both `absa_with_llm` and `summary_with_llm` attempt the LLM call twice before giving up. This handles transient Ollama hiccups. The third layer is the fallback at the bottom of each LLM module: if both attempts fail, `absa_with_llm` imports and calls `aspect_based_sentiment_improved` from `absa.py`; `summary_with_llm` imports and calls the T5 pipeline. The user sees a result either way — the LLM being unavailable mid-session never surfaces as a user-visible error."

---

### Q12: "What are the scalability limitations of your system?"

**Strong answer:** "I can list four concrete ones. Ollama is single-process — concurrent ABSA and summarization requests queue serially inside it. Under 10 concurrent requests, two users submitting long reviews simultaneously means one waits the full 5-15 seconds for Ollama to finish the first before starting. Model loading — each gunicorn worker loads all models independently. Four workers × 1.3GB (transformer path) = 5.2GB RAM minimum. No async — Flask is synchronous; a slow Ollama response blocks a worker thread for its entire duration, reducing effective concurrency. MySQL — a single instance handles both write (analysis results) and read (analytics) queries. Under high analytics query load, write latency increases. For 10x scale, I would add Kafka for async processing, separate model microservices, and MySQL read replicas."

---

### Q13: "How would you scale this to handle 10,000 reviews per day?"

**Strong answer:** "10,000 reviews per day is about 7 per minute — current single-server can handle this. But if you mean peak load of 100 concurrent reviews, the architecture needs to change. I would decouple ingestion from processing: POST /analyze publishes the review to a Kafka topic and returns a job ID immediately. A pool of consumer workers pull reviews and run the pipeline. The client polls GET /status/{job_id} for results. This makes the API non-blocking. For model serving, I would separate BiLSTM into a TensorFlow Serving container, ABSA into a FastAPI microservice, and summarization into another. Each scales independently based on queue depth. Redis caches results keyed by review hash — the same review analyzed twice hits cache. MySQL gets a read replica for analytics queries. Kubernetes with HPA scales ABSA workers based on Kafka consumer lag metric."

---

### Q14: "How does T5 summarization work?"

**Strong answer:** "T5 (Text-to-Text Transfer Transformer) is an encoder-decoder model. The encoder reads the input — I prepend 'summarize:' as a task prefix, which T5 was pre-trained to recognize. The decoder autoregressively generates the summary token by token, attending to the encoder's full representation at each step. I use beam search with 4 beams: instead of greedily picking the highest-probability next token, the decoder maintains 4 candidate sequences simultaneously and picks the one with the highest cumulative log-probability at the end. Key parameters I tuned: `num_beams=4` (2 produced keyword lists, 4 produces full sentences), `no_repeat_ngram_size=3` (prevents repeated phrases — without this, 'works great, works great, works great' was a real output), `length_penalty=2.0` (penalizes length, encourages conciseness), `max_length=80` (caps output tokens)."

---

### Q15: "What is beam search and why does num_beams=4 matter?"

**Strong answer:** "Beam search is a search strategy for sequence generation. Greedy decoding picks the single highest-probability token at each step, which is fast but locally optimal — you can get stuck in a low-quality sequence because early choices constrain later ones. Beam search with k=4 keeps 4 candidate sequences alive simultaneously. At each step, each candidate is expanded by all possible next tokens, giving 4×vocabulary_size candidates, then the top 4 by cumulative log-probability are kept. At the end, the highest-scoring complete sequence is returned. With `num_beams=2`, my T5 summaries were essentially keyword extraction: 'good battery, fast performance, decent camera.' With `num_beams=4`, it produces: 'The reviewer is highly satisfied with the battery life and performance, noting some concerns about camera quality in low light.' The inflection point is at 4 — going to 6 adds marginal quality at significant compute cost."

---

### Q16: "Why did you use SHA-256 for deduplication instead of a UNIQUE constraint on the text?"

**Strong answer:** "MySQL's UNIQUE index on a TEXT column requires specifying a prefix length — the index only stores and compares the first N bytes. In practice this is set to 255 bytes. For reviews shorter than 255 characters this works correctly. But for a 500-character review, two different reviews that share the same first 255 characters would be flagged as duplicates by the index — a false positive. Conversely, if the index is set too short, genuine duplicates with differences beyond the prefix length would not be detected — a false negative. SHA-256 hashes the entire review content to a fixed 64-character hex string. A UNIQUE constraint on `review_hash VARCHAR(64)` is an exact equality check with no prefix truncation. The SHA-256 collision probability is astronomically small (2^-256) — effectively zero for any real dataset of product reviews."

---

### Q17: "What is the `analysis_source` column for?"

**Strong answer:** "It's an ENUM column on the `reviews` table with values 'llm' and 'transformer'. It records which pipeline produced the analysis for each stored review. This serves two purposes. Operationally: if a bug is discovered in one path, I can query `SELECT * FROM reviews WHERE analysis_source = 'llm'` to find affected records and know which ones need re-analysis. Analytically: it enables A/B quality comparison between the two paths. I can pull 100 LLM-analyzed reviews and 100 transformer-analyzed reviews, manually annotate a sample, and compute precision/recall for each path. This column makes the pipeline transparent in the data layer, not just in the application logs."

---

### Q18: "How does negation handling work in your ABSA?"

**Strong answer:** "In the lexicon-based path, I have a `check_negation_context()` function that looks for negation words in a 5-word window before the sentiment word. For 'not great' — the word 'great' appears in `positive_indicators`; the function finds 'not' within 5 words before 'great'; it returns True (negated). The `analyze_clause_sentiment()` function then flips the signal: negated positive words add to the negative count, negated negative words add to the positive count. One critical fix was removing 'not' and 'no' from `negative_indicators`. In the original code, the word 'not' was in both `negative_indicators` and checked by the negation window. For 'not bad': the word 'not' matched `negative_indicators` adding 1 negative signal; then 'bad' matched `negative_indicators` adding another; then the negation window for 'bad' found 'not' and flipped it positive — net result: -2 + 1 = -1, still negative. Correct result should be mildly positive. Removing 'not'/'no' from the lexicon and using them only in the negation window fixed this."

---

### Q19: "What is the difference between /analyze and /api/sentiment?"

**Strong answer:** "Both routes run the complete pipeline — BiLSTM sentiment, ABSA, and summarization — on the submitted review. The difference is persistence and source metadata. `/analyze` writes the result to MySQL via `save_review_with_absa()`: review text, hash, summary, sentiment, confidence score, aspect results, and analysis source are all stored. It also strips the `_source` key before returning the response so internal metadata isn't exposed. `/api/sentiment` runs the same pipeline but makes no DB write. It's designed for clients that want real-time analysis without contributing to the review history — for example, a testing harness, a developer checking their integration, or a system that handles its own storage. The `_source` key is also popped from the response."

---

### Q20: "How would you add authentication to this API?"

**Strong answer:** "For this system, API key authentication is the simplest and most appropriate mechanism. I would add a `Flask-Limiter` integration and a middleware check: each request must include an `X-API-Key` header; the middleware hashes the key with SHA-256, looks it up in a `api_keys` table in MySQL, and rejects unauthorized requests with 401. I would also add rate limiting per key — Flask-Limiter with a Redis backend provides this: `@limiter.limit('100/hour')` on each route. For production, I would also add HTTPS enforcement at the Nginx layer (reject HTTP, redirect to HTTPS), set CORS headers to only allow specific origins for browser clients, and add a request ID header for tracing requests through logs. JWT tokens are another option but are better suited for user-session authentication rather than service-to-service API access."

---

### Q21: "What monitoring would you add in production?"

**Strong answer:** "Four categories. Latency: Prometheus histograms per route with P50/P95/P99 buckets. I specifically want to track the BiLSTM, ABSA, and summary stages independently so I can identify which component degrades under load. Error rates: Prometheus counters for 4xx and 5xx responses per route. A sudden spike in 500s for `/analyze` usually means an Ollama crash or model loading failure. Drift detection: track the distribution of sentiment labels over time. If the Positive/Negative/Neutral ratio shifts significantly over a week, it could indicate model drift (the language of reviews has shifted away from the training distribution) or a bug in the pipeline. Model confidence: track the distribution of `confidence_score` values. If the average score drifts toward 0.5, the model is becoming more uncertain — a sign it needs retraining. All of this feeds into Grafana dashboards with alerting on anomalies."

---

### Q22: "What is `INSERT IGNORE` and why did you use it?"

**Strong answer:** "`INSERT IGNORE` is a MySQL extension that executes the INSERT statement but suppresses errors that would normally arise from constraint violations. If a `UNIQUE` constraint would reject the row, `INSERT IGNORE` silently discards the row and reports zero rows affected instead of raising an error. I use it on `INSERT INTO absa_results (review_id, aspect, sentiment)` where `(review_id, aspect)` is a composite UNIQUE key. The primary deduplication guard is the application-layer hash check — if the review hash exists, the function returns early and never reaches the ABSA INSERT. But under concurrent load, two requests for the same review could both pass the hash check before either has committed the INSERT (a classic TOCTOU race). The `INSERT IGNORE` is the second line of defense: if both threads proceed to the ABSA INSERT, the second one is silently discarded. This gives two-layer protection: app-level check (reduces unnecessary DB round-trips) + DB-level constraint (handles the race condition)."

---

## Quick Reference: Key Numbers

| Component | RAM | Latency (CPU) |
|---|---|---|
| BiLSTM | ~100MB | 50-150ms |
| spaCy en_core_web_sm | ~50MB | 50-200ms |
| DistilBERT SST-2 | ~260MB | 100-300ms per aspect |
| T5-base | ~850MB | 500-2000ms |
| llama3.2 (Ollama) | ~2GB | 1000-5000ms |
| **Total (transformer path)** | **~1.3GB** | **~2-8s** |
| **Total (LLM path)** | **~3.3GB** | **~5-15s** |

| Threshold | Meaning |
|---|---|
| BiLSTM pred > 0.5 | Positive |
| BiLSTM pred >= 0.15 | Neutral |
| BiLSTM pred < 0.15 | Negative |
| num_beams = 4 | Quality inflection point for T5 |
| no_repeat_ngram_size = 3 | Eliminates phrase repetition |
| temperature = 0 | Deterministic (ABSA classification) |
| temperature = 0.3 | Controlled variation (summarization) |
| MAX_REVIEW_LENGTH = 2000 | Input validation cap |
| MAX_BATCH_SIZE = 10 | Batch abuse prevention |
