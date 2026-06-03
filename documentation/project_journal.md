# Project Engineering Journal
## Amazon Review Sentiment & ABSA System

*A first-person technical diary of decisions, mistakes, and learnings built during development. This is not a polished retrospective — it's the record of what actually happened.*

---

## Entry 1 — First Pass: Global DB Connection

The first version of `db_connection.py` was simple: one global `_db` object.

```python
# First version — DO NOT USE
_db = None

def get_connection():
    global _db
    if _db is None or not _db.open:
        _db = pymysql.connect(...)
    return _db
```

This worked perfectly during manual testing. I would submit a review, check the DB, the row was there. Ship it, right?

Wrong. When I ran two browser tabs simultaneously and submitted two reviews at once, I started seeing corrupted data. One review's summary would end up attached to another review's row. Sometimes the insert would partially succeed — review stored but no ABSA rows. Sometimes I'd get a cryptic `ProgrammingError: cursor is not connected` traceback.

I spent a few hours thinking this was a MySQL issue or a transaction bug. Then I read the PyMySQL documentation more carefully: "PyMySQL does not support shared connections across threads." Flask's development server has a `threaded=True` mode by default in newer versions. Two requests hitting the server simultaneously means two threads calling `get_connection()` and both getting back the same global `_db` object. Thread A opens a cursor, Thread B opens a cursor on the same connection, Thread A calls `fetchone()`, and gets Thread B's result set.

The fix was `threading.local()`. This is a Python standard library object where attribute access is keyed by the calling thread's identity. `_local.db` in Thread 1 is a completely different variable from `_local.db` in Thread 2 — they share the same object but not the same namespace. Each thread gets its own connection, created lazily on first request.

The second part was the teardown. A connection opened in a thread needs to be closed when the request ends, not held open forever. Flask's `@teardown_appcontext` decorator runs a function after every request context, whether the request succeeded or raised an exception. Perfect lifecycle management with no manual cleanup needed in each route handler.

**Lesson:** Always check thread-safety docs before using a database client in a multi-threaded web server. The failure mode is silent data corruption, not an obvious crash.

---

## Entry 2 — ABSA v1: The Lexicon-Only Disaster

The first ABSA implementation was pure lexicon matching. I had a dictionary of product features and scanned for their presence in the review text:

```python
# First version — too brittle
for feature in FEATURES:
    if feature in review.lower():
        aspects[feature] = score_sentiment(review)
```

This produced hilariously wrong results:
- "The phone has good battery" → extracts "battery" (correct) but `score_sentiment(review)` scored the *entire review* not the battery sentence
- "I returned this because the battery died after 3 days" → no match because "battery" maps to "battery" but the word used was "battery" — wait, that should work. The problem was the sentiment scoring: the function saw "returned" and "died" in the full review and scored it Positive because "good" from the first sentence dominated

Two problems: (1) substring matching on the full review misses context, and (2) sentiment scoring the full review for each aspect makes every aspect have the same sentiment.

I added spaCy to extract noun chunks and moved to sentence-level sentiment scoring. Better, but still missed compound nouns. "Battery life is great" — spaCy's noun chunker might extract "battery" and "life" as separate tokens or as "battery life" depending on the sentence structure. Inconsistent results.

Added the compound noun strategy: find a NOUN token, check if it has children with `dep_="compound"`, concatenate them. `"battery" ← compound ← "life"` → "battery life". This was the key fix.

Still had duplicate aspects: "battery" and "battery life" both extracted from the same sentence. Added substring deduplication: if aspect A is a substring of aspect B, keep B. Then added character n-gram cosine similarity grouping to catch near-duplicates like "audio quality" and "sound quality" — these should be merged, they describe the same thing.

Each fix added complexity. The pipeline went from 20 lines to 300 lines over three iterations. The important insight: rule-based NLP is a ratchet — you keep adding rules to fix the previous rules' failure modes, and you never quite feel done.

**Lesson:** Start with the extraction strategy before the sentiment strategy. Getting the right nouns extracted is harder than scoring them.

---

## Entry 3 — The Deduplication Bug That Almost Got Me

Early in development, I added a UNIQUE constraint on review_text in MySQL:

```sql
ALTER TABLE reviews ADD UNIQUE (review_text(255));
```

The `(255)` is MySQL's way of creating a prefix index on a TEXT column — it indexes only the first 255 characters. I tested it with a few short reviews, it worked. Submitting the same review twice rejected the duplicate with an IntegrityError.

Two weeks later, a user testing with a long product description (a 400-word review about a laptop) submitted it twice. Both inserts succeeded. I checked the database — two rows with identical review text, different IDs. The prefix index only compared the first 255 bytes, and both reviews obviously shared the same first 255 bytes.

The fix was SHA-256 hashing. `hashlib.sha256(text.strip().encode()).hexdigest()` produces a fixed 64-character hex string regardless of the input length. A UNIQUE constraint on a VARCHAR(64) `review_hash` column is an exact equality check — no prefix, no truncation, no false negatives for long reviews.

```python
def hash_review(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
```

The application-layer check (SELECT before INSERT) is the primary guard. The `INSERT IGNORE` on ABSA rows is the secondary guard for the race condition where two concurrent requests for the same review both pass the SELECT check before either commits.

I also changed the schema: dropped the broken UNIQUE(review_text(255)) and added:
```sql
ALTER TABLE reviews ADD COLUMN review_hash VARCHAR(64);
ALTER TABLE reviews ADD UNIQUE KEY (review_hash);
```

**Lesson:** MySQL prefix indexes are a footgun for deduplication. If you care about uniqueness, hash the content and index the hash.

---

## Entry 4 — The Python UnboundLocalError That Took an Hour to Debug

This one was embarrassing because I knew the rule but applied it incorrectly under time pressure.

In `analyze_clause_sentiment()`, the original code:

```python
# Broken — UnboundLocalError
neutral_indicators = sum(1 for word in neutral_indicators if word in clause_lower)
```

The left-hand side assigns to `neutral_indicators`. Python 3's scoping rule: if a name is assigned *anywhere* in a function, Python classifies it as a local variable *throughout the entire function*, including lines before the assignment. So even though `neutral_indicators` is imported at module scope, the assignment on this line makes Python treat it as local for the whole function. When the right-hand side tries to iterate `neutral_indicators` (the set), Python looks for the local variable `neutral_indicators` — which hasn't been assigned yet at that point in execution — and raises `UnboundLocalError: local variable 'neutral_indicators' referenced before assignment`.

The fix: rename the result variable to not collide with the module-level set name:

```python
# Fixed
neutral_count = sum(1 for word in neutral_indicators if word in clause_lower)
```

The same issue applied to `pos_count` and `neg_count` — those were fine because the module-level names were `positive_indicators` and `negative_indicators` (plural with "indicators"), while the loop variables were `pos_count` and `neg_count`. No collision.

The reason this took an hour: the traceback pointed to line 372 where `neutral_indicators` was referenced, but the *cause* was line 372 itself where it was both read and assigned. I kept looking at the module-level import and couldn't see why it would be unbound. Only after re-reading the Python data model docs on name binding did I recognize the pattern.

**Lesson:** In Python 3, never use the same name for both a module-level variable and a local assignment result inside a function. The scoping rule applies retroactively to the entire function, not just lines after the assignment.

---

## Entry 5 — Bracket Markers and DistilBERT's Confusion

My first instinct for aspect-level sentiment was to tell DistilBERT which aspect to focus on by marking it in the input:

```python
# First version
context_text = f"[{aspect}] {clause}"
# e.g., "[camera] the camera is blurry in low light"
```

The reasoning seemed sound: BERT-style models are trained on `[CLS] text [SEP]` formats, they handle special tokens, so maybe `[camera]` would help the model focus on camera-related sentiment.

I tested this by running the same clause with and without brackets for 20 reviews. The results were inconsistent. "The camera is blurry in low light" scored Negative (correct) without brackets, but "[camera] the camera is blurry in low light" sometimes scored Neutral. The brackets were not neutral decorators — they were tokens that SST-2 had never seen during fine-tuning, because the SST-2 dataset is movie reviews without any bracket notation. The model didn't know what to do with `[camera]` as a standalone token and apparently treated it as something that softened the sentiment signal.

Removing the brackets and passing the clause directly fixed this. DistilBERT SST-2 was fine-tuned to classify sentiment of natural text — just give it natural text.

The broader lesson: when using a pre-trained fine-tuned model, your input format at inference must match the training distribution as closely as possible. Any systematic deviation — unusual tokens, prefixes, special formatting — can hurt accuracy in ways that are not obvious.

**Lesson:** Don't add annotations to model inputs unless you know the model was trained with those annotations. SST-2 → natural text only.

---

## Entry 6 — LLM Temperature and Non-Determinism

Integrating Ollama was straightforward. `ollama.chat()` with the right model name, system prompt, user prompt, done. The first version used `temperature=0.3` for both ABSA and summarization — my reasoning was that some temperature gives more natural output.

The problem became visible when I submitted the same review three times and compared ABSA outputs:

```
Run 1: camera: Positive, battery: Negative, performance: Neutral
Run 2: camera: Negative, battery: Positive, performance: Neutral   ← labels swapped!
Run 3: camera: Positive, battery: Negative, performance: Positive
```

At temperature=0.3, there is enough randomness in the sampling that the model sometimes reverses Positive and Negative labels for the same input. This is catastrophic for a classification task where correctness matters more than naturalness.

Switched ABSA to `temperature=0`. Temperature=0 is equivalent to greedy decoding — always pick the highest-probability next token. No randomness. The same review always produces the same output. For a three-class classification problem (Positive/Negative/Neutral), determinism is non-negotiable.

For summarization, I kept `temperature=0.3`. At temperature=0, summaries sounded robotic: "The reviewer liked the camera. The battery was bad. The performance was good." At 0.3, they read more naturally: "The reviewer finds the camera impressive but notes consistent battery drain issues, though performance under daily use remains satisfactory." The slight variation at 0.3 doesn't cause correctness issues for an open-ended generation task the way it does for classification.

**Lesson:** Temperature is a correctness knob for classification tasks and a quality knob for generation tasks. For any task with a fixed label space, use temperature=0.

---

## Entry 7 — T5 num_beams: The Keyword Extraction Problem

Early T5 summaries with `num_beams=2`:

```
Input: "The battery lasts all day with heavy use. Camera is excellent in all conditions. 
        Build quality feels premium. The price is a bit high for what you get but overall 
        very satisfied with the purchase."

Output: "good battery, excellent camera, premium build, high price"
```

That's not a summary. That's a list of adjective-noun pairs. num_beams=2 isn't enough search width for T5 to commit to sentence structures. With `num_beams=4`:

```
Output: "The reviewer is highly satisfied with the device's all-day battery life, excellent 
        camera performance, and premium build quality, though notes the price is slightly 
        high relative to the value offered."
```

That's an actual abstractive summary. The difference is that with 4 beams, the decoder explores more candidate continuations and finds ones that commit to a full sentence structure rather than the greedy keyword-list paths.

The next problem appeared immediately: with `num_beams=4` but without `no_repeat_ngram_size`:

```
Output: "The reviewer appreciates the battery life, battery life being exceptional, 
        with battery life lasting all day."
```

T5 was trained on large text corpora and learned that phrases that appear frequently in the source also appear frequently in targets. Without ngram repetition prevention, it happily repeated "battery life" three times in one sentence.

Setting `no_repeat_ngram_size=3` prevents any 3-gram from appearing twice. I tried `no_repeat_ngram_size=2` first — still got "battery life, battery life is great." With 3: clean. The tradeoff is that with a very short max_length (e.g., 30 tokens), aggressive ngram blocking can cause the decoder to get stuck avoiding previously used phrases and produce awkward output. At max_length=80 this isn't an issue.

**Lesson:** num_beams and no_repeat_ngram_size are not independent. Get beams working first (at least 4), then add ngram blocking. Test with reviews that naturally repeat the same noun frequently.

---

## Entry 8 — Error String in the Database

One day while auditing the MySQL database, I found this in the `summarized_review` column:

```
Could not generate summary: [Errno 111] Connection refused
```

Also found:
```
[Errno 110] Connection timed out
```

And:
```
ollama._types.ResponseError: model 'llama3.2' not found, try pulling it first
```

The original `summary_with_llm.py` had:

```python
# Original broken version
def generate_summary(text):
    try:
        result = _call_llm(text)
        return result
    except Exception as e:
        return f"Could not generate summary: {e}"  # returned exception string!
```

And in `app.py`, `save_review_with_absa` unconditionally wrote whatever `generate_summary()` returned to the DB as `summarized_review`. So error messages from Ollama were being stored as the summary.

The fix has two parts:
1. In `summary_with_llm.py`: catch exception → log it → retry → if retry fails → call T5 fallback → if T5 fails → return `"Summary not available."` (a safe constant, not an exception string)
2. In `save_review_with_absa`: no change needed — once generate_summary never returns exception strings, the DB is clean

The `"Summary not available."` sentinel is useful because it's distinguishable from real summaries in data audits. A simple `SELECT * FROM reviews WHERE summarized_review LIKE '[Errno%'` would have found the corrupted rows before the fix.

**Lesson:** Functions that catch exceptions should return safe sentinel values, not `str(exception)`. Error information belongs in logs, not in user-facing data columns.

---

## Entry 9 — LLM Prompt Drift and Regex Anchoring

The first LLM ABSA prompt was minimal:

```
List the aspects mentioned in this review and their sentiment:
aspect: sentiment
```

The model's output was creative:

```
camera: the camera quality is quite good with excellent low light performance
battery: battery life is poor, only lasting about 4 hours
price: expensive
```

The first line has verbose explanation after the sentiment. The third line has no explanation but a single-word sentiment that isn't capitalized consistently. My first parser split on `:` and took the second half — so "camera" → "the camera quality is quite good..." — the aspect was fine but the sentiment was the full sentence.

I rewrote the prompt with explicit format rules, capitalization requirements, and five worked examples. Then tightened the parser regex from `(.+?): (.+)` (matches anything) to:

```python
_LINE_RE = re.compile(
    r"^([a-zA-Z][a-zA-Z0-9 \-/]{1,58}):\s*(Positive|Negative|Neutral)\s*$",
    re.IGNORECASE,
)
```

Key constraints added:
- `^` and `$` anchors: must match the entire line, not just a substring
- `[a-zA-Z]` start: aspect must begin with a letter (no "- camera:", no "1.")
- `{1,58}` length: minimum 2 chars total, max 59 — rejects single characters and runaway phrases
- `[a-zA-Z0-9 \-/]` character class: only letters, digits, spaces, hyphens, slashes — rejects model preamble like "here are the aspects:"
- `(Positive|Negative|Neutral)` at end: sentiment must be exactly one of three values, nothing else on the line

After this, the only lines parsed are well-formed `aspect: Sentiment` lines. All preamble, verbose explanations, and malformed outputs are silently ignored. The retry logic exists because the first attempt sometimes produces preamble ("Here are the aspects I found:") but the second attempt almost never does.

**Lesson:** Strict regex parsers are better than lenient parsers for LLM output. The model will always find a way to produce output you don't expect; your parser should accept only what you explicitly allow.

---

## Entry 10 — Docker and the Ollama Sidecar Pattern

My first attempt at containerization was ambitious: include everything in one Dockerfile. Python app, models, AND Ollama + llama3.2. 

```dockerfile
# First attempt — don't do this
RUN curl -fsSL https://ollama.ai/install.sh | sh
RUN ollama pull llama3.2  # This hangs in Docker build
```

Problems:
1. `ollama pull` during Docker build requires Ollama to be running as a server process, but `RUN` commands execute without a daemon
2. llama3.2 is 2GB — every `docker build` re-downloads 2GB
3. Container startup ran `ollama serve` as a subprocess via `llm_check.py`, then waited 10 seconds for it to initialize — 10 second mandatory wait on every container start
4. The full container image was 12GB (Python + TF + PyTorch + HuggingFace models + Ollama + llama3.2)

Switched to the sidecar pattern: Ollama runs as a separate container (or native process on the host), the app connects to it via `OLLAMA_BASE_URL=http://localhost:11434` (or `http://ollama:11434` in docker-compose). The app container has no Ollama binary.

```yaml
# docker-compose approach
services:
  app:
    build: .
    environment:
      OLLAMA_BASE_URL: http://ollama:11434
    depends_on:
      ollama:
        condition: service_healthy

  ollama:
    image: ollama/ollama
    volumes:
      - ollama_data:/root/.ollama
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 10s
      timeout: 5s
      retries: 5
```

Benefits: Ollama's models are stored in a volume and persist across container restarts (no re-download). App container is ~4GB (Python + ML models only). Startup time drops from 60+ seconds to ~30 seconds (model loading, no Ollama startup wait). The `llm_check.py` three-stage check still works: stage 1 pings `OLLAMA_BASE_URL`, finds Ollama healthy, sets `LLM_AVAILABLE=True`.

The `depends_on: condition: service_healthy` ensures the app container doesn't start until Ollama is accepting connections. Without this, the app starts, `llm_check.py` pings Ollama (which is still initializing), gets a timeout, sets `LLM_AVAILABLE=False`, and imports transformer modules — even though Ollama would have been ready 20 seconds later.

**Lesson:** Ollama is a server application, not a library. Run it as its own process or container, connect to it over HTTP. Don't try to embed it in your application container.

---

## Entry 11 — "not bad" and Double Negation

A review came through: "Not bad for the price, not bad at all." BiLSTM scored it Neutral (reasonable — it's a qualified positive). But the traditional ABSA path scored every aspect Negative.

Traced it: in the original `negative_indicators` set, "not" and "no" were included as negative words. The `analyze_clause_sentiment` function for the clause "not bad for the price" did:
- Iterate `negative_indicators` checking each word against clause_lower
- Found "not" in `negative_indicators` → neg_count += 1
- Found "bad" in `negative_indicators` → neg_count += 1
- neg_count = 2
- Then `check_negation_context("not bad for the price", "bad")` → True (found "not" before "bad") → flip: negated_neg += 1
- Apply flips: neg_count = 2 - 1 = 1, pos_count = 0 + 1 = 1
- Still net neutral-ish, but "not" getting counted as a negative word in the first pass was the problem

The fix: remove "not", "no", and related negation words from `negative_indicators`. Negation words don't express sentiment — they modify sentiment. They belong in `negation_words` only, not in the sentiment lexicons.

After the fix: "not bad for the price" → neg_count = 1 (only "bad"), negated_neg = 1 (bad is negated by not), adjusted: neg_count = 0, pos_count = 1 → Positive. Correct.

**Lesson:** Negation words and sentiment words serve different purposes. Keep them in separate sets and apply them in sequence: first count raw sentiment, then apply negation flips.

---

## Retrospective: What I Would Do Differently

**Architecture:** I would start with async from day one. Flask works for demos, but if you're serious about production, FastAPI with async I/O is worth the initial learning curve. The synchronous Ollama call blocking a Flask worker is the biggest production risk in the current design.

**Testing:** No automated tests in this project. Every fix was manual — submit a review, check the output, adjust the code. For a system with 8 bugs found in production-adjacent testing, automated regression tests would have caught at least 5 of them (the UnboundLocalError, the double-negation, the bracket markers, the empty dict return, the error string in DB).

**Model evaluation:** I trained the BiLSTM but didn't formally evaluate it on a held-out test set. I know it works on obvious examples but I have no precision/recall numbers. Before presenting this as production-ready, a proper evaluation on 1000 labeled reviews is necessary.

**Logging:** Using `print()` throughout. In production, every print goes to stdout with no context (no timestamp, no request ID, no severity). Should have used Python's `logging` module from the start with structured JSON output.

**Configuration:** Hardcoded values scattered across files: `MAX_LENGTH = 150` in app.py, `_SHORT_REVIEW_THRESHOLD = 20` in summary.py, threshold values (0.5, 0.15) in complete_pipeline. These should be in a centralized config file or environment variables.

These aren't excuses — they're the honest gap between a working demo and a production system. The value of this project is understanding where those gaps are.
