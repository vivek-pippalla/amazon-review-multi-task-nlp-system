# Application Flow Documentation

This document describes every significant control flow in the Amazon Review Sentiment ABSA
backend. Each section covers the logical path through the code, all branching conditions, and
the precise function calls involved. Every flow includes a Mermaid sequence or flowchart diagram.

---

## Table of Contents

1. [Request Lifecycle](#1-request-lifecycle)
2. [Sentiment Prediction Flow](#2-sentiment-prediction-flow)
3. [ABSA Flow](#3-absa-flow)
4. [Summarization Flow](#4-summarization-flow)
5. [Database Write Flow](#5-database-write-flow)
6. [Model Selection at Startup](#6-model-selection-at-startup)
7. [Fallback Flow](#7-fallback-flow)
8. [Exception Flow](#8-exception-flow)
9. [Batch Processing Flow](#9-batch-processing-flow)

---

## 1. Request Lifecycle

All analysis requests follow the same outer lifecycle: HTTP arrives, Flask routes it, validation
runs, the NLP pipeline executes, the result is optionally persisted, and a JSON response is
returned. Errors at any stage produce a structured JSON error body.

The two single-review routes differ in only one step: `POST /analyze` writes to the database;
`POST /api/sentiment` does not.

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Flask
    participant Validator
    participant Pipeline as complete_pipeline()
    participant DB as save_review_with_absa()
    participant MySQL

    Client->>Flask: POST /analyze  {"review": "..."}
    Flask->>Flask: request.get_json()
    Flask->>Validator: (data.get("review") or "").strip()

    alt review is empty string
        Validator-->>Client: 400 {"error": "No review text provided"}
    else len(review) > 2000
        Validator-->>Client: 400 {"error": "Review exceeds 2000 characters"}
    else valid
        Validator->>Pipeline: complete_pipeline(review)

        alt pipeline raises exception
            Pipeline-->>Client: 500 {"error": "<exception message>"}
        else success
            Pipeline-->>Flask: result dict
            Flask->>DB: save_review_with_absa(review, summary, sentiment, confidence, aspects, source)

            alt DB raises exception
                DB->>MySQL: rollback()
                DB-->>Client: 500 {"error": "<exception message>"}
            else saved (or duplicate skipped)
                DB-->>Flask: review_id
                Flask-->>Client: 200 {"Overall Sentiment", "Confidence Score", "Summary", "Aspect-based Sentiments"}
            end
        end
    end
```

### Route Differences

| Route | Saves to DB | Returns `_source` field |
|---|---|---|
| `POST /analyze` | Yes | No (popped before response) |
| `POST /api/sentiment` | No | No (popped via `.pop("_source", None)`) |
| `POST /api/batch/analyze` | Yes per item | No (popped per item) |

---

## 2. Sentiment Prediction Flow

`complete_pipeline()` in `app.py` handles all sentiment inference. The review text is cleaned
before model input, but the ORIGINAL un-cleaned text is forwarded to both the summarizer and
the ABSA module so that transformers see proper casing and punctuation.

```mermaid
sequenceDiagram
    autonumber
    participant Caller
    participant clean_text
    participant Tokenizer as Keras Tokenizer (tokenizer.pkl)
    participant PadSeq as pad_sequences()
    participant BiLSTM as BiLSTM model (best_model.keras)
    participant Threshold as Threshold mapper

    Caller->>clean_text: review (original)
    Note over clean_text: 1. text.lower()<br/>2. re.sub(r"[^a-zA-Z\s]", "", text)  — strips digits, punctuation<br/>3. re.sub(r"\s+", " ", text).strip()  — collapses whitespace
    clean_text-->>Caller: cleaned (lowercase, alpha+spaces only)

    Caller->>Tokenizer: texts_to_sequences([cleaned])
    Note over Tokenizer: Maps each word to its integer index<br/>from the training corpus vocabulary.<br/>Unknown words map to 0 (OOV).
    Tokenizer-->>Caller: seq  [[12, 47, 8, ...]]

    Caller->>PadSeq: pad_sequences(seq, maxlen=150, padding="post", truncating="post")
    Note over PadSeq: Short sequences: zeros appended at end.<br/>Long sequences: tokens beyond index 150 dropped.
    PadSeq-->>Caller: padded  shape (1, 150)

    Caller->>BiLSTM: model.predict(padded, verbose=0)
    Note over BiLSTM: Bidirectional LSTM, sigmoid output unit.<br/>Returns scalar in [0.0, 1.0].
    BiLSTM-->>Caller: pred  (e.g. 0.7312)

    Caller->>Threshold: apply thresholds
    Note over Threshold: pred > 0.5  → "Positive"<br/>pred >= 0.15 → "Neutral"<br/>pred < 0.15  → "Negative"
    Threshold-->>Caller: overall_sentiment, confidence_score = round(pred, 4)
```

### Threshold Mapping Table

| BiLSTM Output Range | Assigned Sentiment | Notes |
|---|---|---|
| `(0.5, 1.0]` | Positive | Model confident review is positive |
| `[0.15, 0.5]` | Neutral | Ambiguous or mixed signal |
| `[0.0, 0.15)` | Negative | Model confident review is negative |

The confidence score stored in the database is the raw sigmoid value rounded to 4 decimal
places, not a probability of the assigned class. For a Negative prediction of `0.03`, the
confidence score is `0.03` (close to 0 = strongly negative). For a Positive prediction of
`0.97`, the confidence score is `0.97`.

---

## 3. ABSA Flow

Aspect-Based Sentiment Analysis runs on the ORIGINAL review text (not cleaned). There are two
code paths depending on `LLM_AVAILABLE` set at startup.

### 3a. Traditional Path (spaCy + DistilBERT)

```mermaid
sequenceDiagram
    autonumber
    participant Caller
    participant Norm as normalize_text()
    participant Extract as extract_aspects_improved()
    participant Filter as filter_and_deduplicate_aspects()
    participant Sentiment as get_aspect_sentiment_improved()
    participant Transformer as DistilBERT (SST-2)
    participant Lexicon as Rule-based lexicon

    Caller->>Norm: normalize_text(review)
    Note over Norm: lowercase + collapse whitespace (no stripping of punctuation —<br/>spaCy needs original punctuation for dependency parse)

    Norm-->>Caller: normalized

    Caller->>Extract: extract_aspects_improved(normalized)
    Note over Extract: Strategy 1: spaCy noun chunks (1–4 words)<br/>Strategy 2: Dependency parse — opinion verb→object, nsubj/dobj/pobj nouns<br/>Strategy 3: Compound nouns (token.dep_ == "compound")<br/>Strategy 4: Regex for "X quality", "X life", "X performance", "X speed", etc.<br/>Strategy 5: Qualifier-aware filter — remove redundant qualifiers<br/>Final: lemmatize all candidates
    Extract-->>Caller: candidates (list of strings)

    Caller->>Filter: filter_and_deduplicate_aspects(candidates, normalized)
    Note over Filter: 1. Remove generic/excluded words (is_generic())<br/>2. _deduplicate_by_substring() — keep longer of "battery" vs "battery life"<br/>3. get_related_aspects() — char n-gram cosine similarity (threshold=0.75)<br/>4. Build hierarchy: prefer longer/more-specific form<br/>5. Re-apply qualifier filter + length cap (<=4 words)
    Filter-->>Caller: aspects (deduplicated list)

    loop For each aspect
        Caller->>Sentiment: get_aspect_sentiment_improved(review, aspect)
        Sentiment->>Sentiment: get_aspect_context(review, aspect)
        Note over Sentiment: Split review into sentences containing aspect.<br/>For each sentence, split on contrastive conjunctions<br/>(but/however/although/though/yet/nevertheless/nonetheless).<br/>Extract only the clause(s) where the aspect appears.

        alt DistilBERT available
            Sentiment->>Transformer: analyze_sentiment_with_transformer(context_clause, aspect)
            Note over Transformer: Tokenize context (max 512 tokens, truncate).<br/>softmax(logits)[0][1] → SST-2 positive score.<br/>avg score across all clauses.
            Transformer-->>Sentiment: avg_score

            alt avg_score > 0.6
                Sentiment-->>Caller: "Positive"
            else avg_score < 0.4
                Sentiment-->>Caller: "Negative"
            else
                Sentiment-->>Caller: "Neutral"
            end

        else DistilBERT unavailable
            Sentiment->>Lexicon: analyze_clause_sentiment(context, aspect)
            Note over Lexicon: Count pos/neg/neutral indicator words.<br/>Apply negation context (5-word window).<br/>Apply intensity modifiers (strengthen ×1.5, weaken ×0.75).<br/>Compute weighted score → Positive/Neutral/Negative.
            Lexicon-->>Caller: sentiment label
        end
    end

    Caller-->>Caller: {aspect: sentiment, ...}
```

### 3b. LLM Path (llama3.2 via Ollama)

```mermaid
sequenceDiagram
    autonumber
    participant Caller
    participant LLMFunc as aspect_based_sentiment_llm()
    participant Ollama as ollama.chat (llama3.2)
    participant Parser as _parse_response()
    participant Fallback as aspect_based_sentiment_improved()

    Caller->>LLMFunc: review_text

    loop attempt in [0, 1]
        LLMFunc->>Ollama: chat(model="llama3.2", messages=[system+user], temperature=0, num_predict=400)
        Note over Ollama: System prompt: _SYSTEM (aspect definition rules, sentiment rules, output format)<br/>User prompt: _USER_TEMPLATE with 5 few-shot examples + "{review_text}"<br/>temperature=0 → deterministic classification output
        Ollama-->>LLMFunc: response["message"]["content"]

        LLMFunc->>Parser: _parse_response(content)
        Note over Parser: Split on newlines.<br/>Apply _LINE_RE regex per line:<br/>  ^([a-zA-Z][a-zA-Z0-9 \-/]{1,58}):\s*(Positive|Negative|Neutral)\s*$<br/>Matched lines → {aspect.lower(): sentiment.capitalize()}

        alt result is non-empty dict
            Parser-->>Caller: {aspect: sentiment, ...}
        else result is empty (parse failed or LLM echoed prompt)
            Note over LLMFunc: Retry on attempt 0. After attempt 1, fall through to fallback.
        end
    end

    LLMFunc->>Fallback: aspect_based_sentiment_improved(review_text)
    Note over Fallback: Full transformer ABSA pipeline (3a above)
    Fallback-->>Caller: {aspect: sentiment, ...}
```

---

## 4. Summarization Flow

Summarization also runs on the ORIGINAL review text and has two paths.

### 4a. T5-base Path (transformer mode)

```mermaid
sequenceDiagram
    autonumber
    participant Caller
    participant Guard as Short-review guard
    participant T5Tok as T5Tokenizer
    participant T5Model as T5ForConditionalGeneration (t5-base)
    participant Decoder

    Caller->>Guard: len(text.split()) < 20?
    alt fewer than 20 words
        Guard-->>Caller: return text unchanged (passthrough)
    else 20+ words
        Guard->>T5Tok: tokenizer("summarize: " + text, max_length=512, truncation=True)
        Note over T5Tok: Prepends "summarize: " task prefix required by T5.<br/>Hard cap at 512 tokens; longer reviews are truncated.
        T5Tok-->>Guard: input_ids tensor

        Guard->>T5Model: model.generate(input_ids, max_length=80, min_length=20, num_beams=4, no_repeat_ngram_size=3, length_penalty=2.0, early_stopping=True)
        Note over T5Model: num_beams=4: beam search for quality.<br/>no_repeat_ngram_size=3: prevents phrase repetition.<br/>length_penalty=2.0: penalises verbose outputs.<br/>Output: token id sequence.
        T5Model-->>Guard: generated_ids

        Guard->>Decoder: tokenizer.decode(ids[0], skip_special_tokens=True)
        Decoder-->>Caller: summary string

        alt T5 raises exception
            T5Model-->>Caller: "Summary not available."
        end
    end
```

### 4b. LLM Path (llama3.2 via Ollama)

```mermaid
sequenceDiagram
    autonumber
    participant Caller
    participant Guard as Short-review guard
    participant LLMFunc as generate_summary() [LLM]
    participant Ollama as ollama.chat (llama3.2)
    participant T5Fallback as generate_summary() [T5]

    Caller->>Guard: len(text.split()) < 20?
    alt fewer than 20 words
        Guard-->>Caller: return text unchanged (passthrough)
    else 20+ words
        loop attempt in [0, 1]
            Guard->>Ollama: chat(model="llama3.2", messages=[system+user], temperature=0.3, num_predict=150)
            Note over Ollama: System: neutral 3rd-person, 2–3 sentences, max 60 words,<br/>no invented details, no "In summary" opener.<br/>temperature=0.3 for natural phrasing while staying factual.

            alt response non-empty
                Ollama-->>Caller: summary string
            else empty response or exception
                Note over Guard: Retry on attempt 0.
            end
        end

        Guard->>T5Fallback: generate_summary(text)  [T5 path]
        Note over T5Fallback: Full T5 pipeline (4a above)

        alt T5 also fails
            T5Fallback-->>Caller: "Summary not available."
        else T5 succeeds
            T5Fallback-->>Caller: T5 summary string
        end
    end
```

---

## 5. Database Write Flow

`save_review_with_absa()` in `app.py` handles all persistence. It is idempotent: submitting the
same review text twice results in one DB row, not two. Idempotency is enforced by a SHA-256
hash check before any INSERT.

```mermaid
sequenceDiagram
    autonumber
    participant Caller
    participant HashFn as hash_review()
    participant Cursor as MySQL Cursor (DictCursor)
    participant MySQL

    Caller->>HashFn: hash_review(review)
    Note over HashFn: hashlib.sha256(review.strip().encode("utf-8")).hexdigest()<br/>Returns 64-char hex string (CHAR(64) in DB).
    HashFn-->>Caller: review_hash

    Caller->>Cursor: get_cursor()
    Caller->>MySQL: SELECT id FROM reviews WHERE review_hash = %s

    alt row exists (duplicate)
        MySQL-->>Caller: existing["id"]  — early return, no INSERT
    else new review
        Caller->>MySQL: INSERT INTO reviews (review_hash, review_text, summarized_review, overall_sentiment, confidence_score, analysis_source) VALUES (...)
        MySQL-->>Caller: cursor.lastrowid → review_id

        loop for aspect, sentiment in aspects.items()
            Caller->>MySQL: INSERT IGNORE INTO absa_results (review_id, aspect, sentiment) VALUES (...)
            Note over MySQL: IGNORE silently skips if (review_id, aspect) already exists.<br/>Enforced by UNIQUE KEY uq_review_aspect.
        end

        Caller->>MySQL: commit()
        MySQL-->>Caller: OK
        Caller-->>Caller: return review_id
    end

    alt any exception during INSERT or commit
        Caller->>MySQL: get_connection().rollback()
        Caller-->>Caller: raise exception → propagates to route handler → 500
    end

    Note over Caller: cursor.close() always runs (finally block)
```

---

## 6. Model Selection at Startup

Before Flask starts serving requests, `initialize_llm()` is called to probe whether a local LLM
is available. The result permanently sets `LLM_AVAILABLE` and determines which module is imported
for ABSA and summarization.

```mermaid
flowchart TD
    A([Flask app.py starts]) --> B[from llm_check import LLM_AVAILABLE, initialize_llm]
    B --> C[initialize_llm]

    C --> D{check_local_ollama\nGET /api/tags on localhost:11434\ntimeout=3s}
    D -->|HTTP 200 + 'llama3.2' in response| E[LLM_AVAILABLE = True\nstage: local Ollama]
    D -->|unreachable or model missing| F[start_local_ollama\nsubprocess: ollama serve\nsleep 10s\ncheck_local_ollama again]

    F -->|llama3.2 found after start| E
    F -->|still not found| G[start_docker_model_runner\ndocker model pull ai/llama3.2\ndocker model run ai/llama3.2]

    G -->|docker commands succeed| H[LLM_AVAILABLE = True\nUSE_DOCKER_MODEL_RUNNER = True]
    G -->|docker commands fail| I[LLM_AVAILABLE = False\nfallback logged]

    E --> J{LLM_AVAILABLE?}
    H --> J
    I --> J

    J -->|True| K["from absa_with_llm import aspect_based_sentiment_llm as aspect_based_sentiment\nfrom summary_with_llm import generate_summary\nANALYSIS_SOURCE = 'llm'"]
    J -->|False| L["from absa import aspect_based_sentiment_improved as aspect_based_sentiment\nfrom summary import generate_summary\nANALYSIS_SOURCE = 'transformer'"]

    K --> M[Flask begins serving requests]
    L --> M
```

---

## 7. Fallback Flow

The system has multiple fallback layers to ensure that every request returns a meaningful
response rather than an unhandled exception.

```mermaid
flowchart TD
    A([complete_pipeline called]) --> B{LLM_AVAILABLE?}

    B -->|False — set at startup| C[Use transformer path\nabsa_with_llm NOT imported\nsummary_with_llm NOT imported]

    B -->|True| D[Call LLM functions]

    D --> E{LLM ABSA:\naspect_based_sentiment_llm}
    E -->|attempt 0 succeeds and parse non-empty| F[Return aspect dict]
    E -->|attempt 0: empty parse| G[Retry attempt 1]
    G -->|attempt 1 succeeds and non-empty| F
    G -->|attempt 1 fails or empty| H[Import and call\naspect_based_sentiment_improved\ntransformer fallback]
    H -->|transformer succeeds| F
    H -->|transformer also fails| I[Return empty dict {}]

    D --> J{LLM Summary:\ngenerate_summary LLM}
    J -->|attempt 0 returns non-empty string| K[Return summary]
    J -->|attempt 0 fails or empty| L[Retry attempt 1]
    L -->|attempt 1 returns non-empty string| K
    L -->|attempt 1 fails or empty| M[Import and call\ngenerate_summary T5 fallback]
    M -->|T5 succeeds| K
    M -->|T5 also raises exception| N[Return 'Summary not available.']

    C --> O[Transformer ABSA\nDistilBERT scoring]
    O -->|DistilBERT unavailable| P[Rule-based lexicon\nanalyze_clause_sentiment]
```

---

## 8. Exception Flow

```mermaid
flowchart TD
    A([Request arrives]) --> B{JSON body parseable?}
    B -->|No / missing 'review' key| C[review = empty string after strip]
    C --> D[400: 'No review text provided']

    B -->|Yes| E{len review > 2000?}
    E -->|Yes| F[400: 'Review exceeds 2000 characters']
    E -->|No| G[complete_pipeline]

    G -->|raises any Exception| H[traceback.print_exc\n500: error message string]
    G -->|returns dict| I[save_review_with_absa — /analyze only]

    I -->|raises Exception| J[rollback\n500: error message string]
    I -->|returns review_id| K[jsonify result\n200 OK]

    subgraph Batch
        BA([Batch item loop]) --> BB{isinstance str and non-empty?}
        BB -->|No| BC[item result: failed, 'Empty or non-string review'\nfailed += 1\ncontinue to next item]
        BB -->|Yes| BD{len > 2000?}
        BD -->|Yes| BE[item result: failed, 'Exceeds 2000 character limit'\nfailed += 1\ncontinue]
        BD -->|No| BF[complete_pipeline + save_review_with_absa]
        BF -->|raises Exception| BG[item result: failed, str e\nfailed += 1\ncontinue]
        BF -->|success| BH[item result: success, data dict\nsucceeded += 1]
    end

    K --> Batch
```

---

## 9. Batch Processing Flow

`POST /api/batch/analyze` applies the full pipeline to each review independently. A failure for
one item never aborts processing for the remaining items.

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Flask
    participant ItemValidator
    participant Pipeline as complete_pipeline()
    participant DB as save_review_with_absa()

    Client->>Flask: POST /api/batch/analyze  {"reviews": ["r1", "r2", ..., "rN"]}
    Flask->>Flask: data.get("reviews")

    alt reviews is None or not a list
        Flask-->>Client: 400 {"error": "'reviews' must be a non-empty list"}
    else len(reviews) > 10
        Flask-->>Client: 400 {"error": "Batch size exceeds the maximum of 10"}
    else valid list
        loop i, review in enumerate(reviews)
            Flask->>ItemValidator: isinstance(review, str) and review.strip() non-empty

            alt not string or empty
                ItemValidator-->>Flask: {index: i, status: "failed", error: "Empty or non-string review"}
            else len(review.strip()) > 2000
                ItemValidator-->>Flask: {index: i, status: "failed", error: "Exceeds 2000 character limit"}
            else
                Flask->>Pipeline: complete_pipeline(review.strip())

                alt pipeline exception
                    Pipeline-->>Flask: {index: i, status: "failed", error: str(e)}
                else success
                    Pipeline-->>Flask: result dict
                    Flask->>DB: save_review_with_absa(...)

                    alt DB exception
                        DB-->>Flask: {index: i, status: "failed", error: str(e)}
                    else saved
                        DB-->>Flask: review_id
                        Flask->>Flask: {index: i, status: "success", data: result}
                    end
                end
            end
        end

        Flask-->>Client: 200 {total, succeeded, failed, results: [...]}
    end
```

### Batch Response Structure

```json
{
  "total": 3,
  "succeeded": 2,
  "failed": 1,
  "results": [
    {"index": 0, "status": "success", "data": { "Overall Sentiment": "Positive", "..." }},
    {"index": 1, "status": "failed",  "error": "Exceeds 2000 character limit"},
    {"index": 2, "status": "success", "data": { "Overall Sentiment": "Negative", "..." }}
  ]
}
```

HTTP status is always `200` for the batch endpoint. Partial failure is indicated in the
`failed` counter and per-item `status` fields — not via a non-200 HTTP status code.
