# Failure Analysis
## Amazon Review Sentiment & ABSA System

---

## Overview

This document catalogs known failure modes across all three pipeline components (overall sentiment, ABSA, summarization), with root causes, real-world examples, and mitigation status. Understanding these failures is as important as understanding the happy path — a system deployed without a failure analysis is a system with unknown reliability.

---

## Failure Mode 1: Sarcasm

**Severity:** High

**Component:** BiLSTM sentiment classification

**Description:** The BiLSTM was trained on literal reviews and has no model of irony or sarcasm. Positive sentiment words trigger positive classification regardless of the underlying meaning.

**Example:**
```
Review: "Oh great, another battery that dies in 2 hours. Absolutely fantastic product."
BiLSTM: Positive (confidence: 0.87)
Correct: Negative
```
The words "great" and "fantastic" dominate the BiLSTM's prediction. The model has no understanding that these words are being used ironically.

**Root cause:** BiLSTM operates on lexical patterns. Sarcasm requires world knowledge — understanding that "great, another battery that dies in 2 hours" is a known complaint pattern with ironic framing. This requires cultural and contextual understanding beyond what a BiLSTM can learn from token sequences.

**LLM path behavior:** llama3.2 handles this significantly better. The LLM has broader world knowledge and understands common sarcasm patterns. For the example above, llama3.2 correctly identifies the overall sentiment as Negative via contextual inference.

**Current mitigation:** None for the BiLSTM path. `analysis_source` column allows identifying which path produced the analysis — if the LLM path is available, sarcasm is handled.

**Recommended fix:** Add a sarcasm detection pre-pass. Options: (1) rule-based patterns (exclamations + negative context), (2) fine-tuned sarcasm classifier (available in HuggingFace: `cardiffnlp/twitter-roberta-base-irony`), (3) rely on LLM path which handles this naturally. Full sarcasm detection is an open NLP research problem — no solution is 100% accurate.

---

## Failure Mode 2: Mixed Sentiment Reviews

**Severity:** High

**Component:** BiLSTM sentiment classification (ABSA partially compensates)

**Description:** The BiLSTM produces one scalar output for the entire review. Mixed reviews ("camera is amazing but battery is terrible") produce a single probability that averages both sentiments. The label depends on which sentiment dominates.

**Example:**
```
Review: "The camera takes stunning photos in all conditions. However the battery is 
         absolutely terrible — dies in 3 hours with normal use."
BiLSTM: Positive (confidence: 0.71)
ABSA: {camera: Positive, battery: Negative}
```
The camera clause contains more sentiment words and longer text, so the BiLSTM leans Positive. The ABSA correctly captures both aspects. But the headline sentiment (Positive) misrepresents the review — a user who cares about battery life would be misled.

**Root cause:** Single-label classification is fundamentally ill-suited for mixed-sentiment text. The model must collapse a multi-dimensional sentiment space into one label.

**Current mitigation:** ABSA partially compensates — the per-aspect breakdown gives a more accurate picture than the headline label. Users and consumers of the API should use `Aspect-based Sentiments` as the primary signal when aspects are present.

**Recommended fix:** Aspect-weighted aggregate sentiment — if 3 out of 4 aspects are Negative but the headline is Positive, the headline should be downgraded to Neutral. Implement: `if len(negative_aspects) > len(positive_aspects): downgrade_headline_one_level()`. This heuristic would improve accuracy for mixed reviews at minimal implementation cost.

---

## Failure Mode 3: Long Reviews (>150 Tokens)

**Severity:** Medium

**Component:** BiLSTM sequence truncation

**Description:** Reviews longer than 150 tokens are truncated at token 150 (`truncating="post"`). Information after position 150 is completely discarded.

**Example:**
```
Review: [200-word detailed review where positive qualities are described in the 
        first 100 words and the significant defect is described in words 150-200]
BiLSTM: Positive (all defect information truncated)
Correct: Negative or Neutral
```

**Approximate impact:** Approximately 15% of Amazon reviews exceed 150 words (based on the 85th percentile threshold design). Of these, only a subset will have their key sentiment signal in the truncated portion — but for detailed technical reviews, negative caveats are often saved for the end ("Overall great, but one major issue...").

**Root cause:** Fixed-length sequence design. The `maxlen=150` was chosen to cover 85% of reviews efficiently. The alternative — `maxlen=500` — would significantly increase model memory and training time while adding value only for 15% of reviews.

**Current mitigation:** `MAX_REVIEW_LENGTH=2000` character limit prevents extremely long inputs from causing excessive inference time. The LLM path handles long context better (llama3.2 has a 128K context window).

**Recommended fix:** Two options. (1) Sliding window: split the review into overlapping 150-token windows, predict on each, aggregate (mean or weighted-by-window-position). (2) Hierarchical model: encode each sentence independently with the BiLSTM, then aggregate sentence-level embeddings with attention. Option 2 is higher quality but more complex.

---

## Failure Mode 4: Ambiguous Aspect References

**Severity:** Medium

**Component:** spaCy-based aspect extraction in `absa.py`

**Description:** Informal or indirect aspect references that require world knowledge to resolve are missed by spaCy's grammatical analysis.

**Examples:**
```
"The thing that charges it keeps falling out" → spaCy may extract "thing" (generic, filtered out)
                                               → Correct: "charging port" or "charger"

"The wobbly bit on top" → spaCy extracts "bit" (generic, filtered out)
                        → Correct: "hinge" or "lid" depending on product

"It heats up like a furnace" → spaCy extracts no product aspect
                             → Correct: "heat dissipation" or "thermal management"
```

**Root cause:** spaCy's dependency parser understands grammatical structure but not semantics. "Thing that charges it" is a relative clause modifying "thing" — spaCy sees a generic noun. Resolving the referent to "charger" requires knowledge that phones have chargers.

**LLM path behavior:** llama3.2 resolves most of these correctly via inference. "The thing that charges it keeps falling out" → `charging port: Negative`. The LLM understands what you're describing from context.

**Current mitigation:** Aspect category matching partially covers common indirect references (words like "charge", "charging" map to the "battery" category). The LLM path handles this better.

**Recommended fix:** If the transformer path is required, add an entity linking step that maps informal references to canonical aspect names using a product-domain knowledge graph or a fine-tuned NER model.

---

## Failure Mode 5: LLM Hallucinated Aspects

**Severity:** Medium

**Component:** `absa_with_llm.py` — llama3.2 ABSA

**Description:** The LLM invents aspects that are not mentioned in the review. The model has been trained to produce helpful, detailed output and sometimes adds "reasonable" aspects that aren't actually present.

**Example:**
```
Review: "Great camera and excellent battery life."
LLM output:
  camera: Positive
  battery: Positive
  build quality: Positive       ← hallucinated (not mentioned)
  customer service: Positive    ← hallucinated (not mentioned)
```

**Root cause:** Large language models are trained to be helpful and complete. The model "knows" that product reviews often discuss build quality and customer service, so it adds these as likely aspects even when not mentioned. This is a known failure mode of instruction-following LLMs.

**Current mitigation:** The system prompt explicitly lists what to exclude and includes the instruction "Do not invent details or opinions not present in the original review." The `_LINE_RE` regex rejects malformed lines but cannot detect hallucinated aspect names that are syntactically valid.

**Recommended fix:** Post-generation verification step: for each extracted aspect, check if the aspect string (or any synonym) appears in the review text. If not, discard. This would eliminate hallucinated aspects at the cost of missing implicit references like "the charging port" being described as "the thing that charges it" — a tradeoff between hallucination and implicit extraction.

---

## Failure Mode 6: Malformed LLM Responses

**Severity:** Low (handled by retry)

**Component:** `absa_with_llm.py` and `summary_with_llm.py`

**Description:** The LLM sometimes produces preamble, disclaimers, or formatting deviations before the actual output.

**Example:**
```
Attempt 1 output:
"Here are the aspects I identified in the review:
- camera: Positive
- battery: Negative"

Parsed result: {} (empty — the lines don't match _LINE_RE because of "- " prefix and preamble)

Attempt 2 output:
"camera: Positive
battery: Negative"

Parsed result: {camera: Positive, battery: Negative} ✓
```

The model almost never produces preamble on the second attempt when the first attempt failed. This behavioral pattern is consistent enough to make single retry reliable.

**Root cause:** LLMs have a tendency to add "helpful" framing around structured output, especially when the user prompt contains instructions like "here are examples." The few-shot examples in `_USER_TEMPLATE` use direct format, but the model sometimes adds framing anyway.

**Current mitigation:** Retry logic in both LLM modules. After two failed attempts, fallback to transformer pipeline.

**Recommended fix:** Add explicit instruction to system prompt: "Output ONLY the aspect-sentiment pairs. No preamble, no headers, no bullet points." Also consider: use `options={"stop": ["\n\n"]}` in ollama.chat to stop generation at the first blank line, preventing multi-paragraph preamble.

---

## Failure Mode 7: Ollama Process Unavailability

**Severity:** High (if LLM path deployed, no graceful degradation without retry)

**Component:** `absa_with_llm.py`, `summary_with_llm.py`, `llm_check.py`

**Description:** Ollama process crashes or becomes unreachable after the initial startup check. `LLM_AVAILABLE=True` was set at startup, so the LLM modules were imported. At request time, `ollama.chat()` throws `ConnectionRefusedError`.

**Scenario:**
1. App starts, Ollama is running → `LLM_AVAILABLE=True`
2. Ollama process OOM-killed by the OS (llama3.2 on 2GB RAM is near the system limit)
3. Next request arrives → `ollama.chat()` throws `ConnectionRefusedError`
4. Retry (attempt 2) → same error
5. Fallback to transformer pipeline → request succeeds

**Current mitigation:** The two-attempt retry + fallback chain catches this scenario. The fallback to transformer ABSA and T5 summary means users see a slightly different quality result but no error.

**Long-term gap:** The system does not attempt to restart Ollama if it crashes. If Ollama is dead, every subsequent request pays the retry cost (two connection attempts + timeout per request). A background health check thread that monitors Ollama and updates `LLM_AVAILABLE` dynamically would fix this.

**Recommended fix:** Add a background thread that pings `OLLAMA_BASE_URL/api/tags` every 30 seconds and updates a global `_llm_healthy` flag. Route handlers check this flag before calling LLM functions, skipping the retry overhead when Ollama is known to be down.

---

## Failure Mode 8: Request Timeout (Ollama on CPU)

**Severity:** Medium

**Component:** `absa_with_llm.py`, `summary_with_llm.py`

**Description:** llama3.2 inference on CPU can take 30-60 seconds for long reviews. The current `ollama.chat()` call has no timeout parameter. A worker thread is blocked for the entire duration.

**Example:** A 500-word review submitted to the LLM ABSA path on a CPU-only machine. llama3.2 processes at ~5 tokens/second. The ABSA prompt + review + generation is approximately 600 tokens = 120 seconds. The Flask worker is blocked for 2 minutes.

**Current mitigation:** `MAX_REVIEW_LENGTH=2000` character limit bounds the maximum review size. At 5 tokens/word and ~5 chars/word, 2000 chars ≈ 400 words ≈ 600 tokens total prompt ≈ 120s worst case. This is too slow but bounded.

**Recommended fix:** Wrap `ollama.chat()` in a `concurrent.futures.ThreadPoolExecutor` with a timeout:
```python
with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(_call_llm, review_text)
    try:
        result = future.result(timeout=30)
    except TimeoutError:
        future.cancel()
        raise  # triggers fallback
```
30-second timeout is generous for llama3.2 on GPU but may be tight on CPU. Alternative: add `timeout=30` directly to the ollama library's HTTP client configuration.

---

## Failure Mode 9: Model Inference Numerical Instability

**Severity:** Low (theoretical, not observed)

**Component:** BiLSTM (`model.predict()`)

**Description:** TensorFlow's float32 arithmetic can produce NaN values under extreme conditions (gradient explosion during training, corrupted model weights). A NaN prediction would propagate through the threshold logic, resulting in `pred > 0.5` evaluating to False (NaN comparisons return False), leading to "Negative" classification for any review.

**Current mitigation:** None. The `pred = float(model.predict(...)[0][0])` line would return `nan` and silently produce incorrect labels.

**Recommended fix:**
```python
pred = float(model.predict(padded, verbose=0)[0][0])
if np.isnan(pred) or np.isinf(pred):
    pred = 0.5  # Default to Neutral on numerical failure
    # Log the anomaly for investigation
```

---

## Failure Mode 10: OOV Tokenizer Edge Cases

**Severity:** Low-Medium

**Component:** Keras Tokenizer in `app.py`

**Description:** Words not in the training vocabulary map to index 0. Index 0 is also the padding token. A review composed primarily of OOV words produces a sequence of zeros — the model sees a padding-only input and produces an arbitrary, unreliable prediction.

**Example:**
```
Review: "Amazeballs gadget, totally bespoke unboxing experience!" 
→ After clean_text: "amazeballs gadget totally bespoke unboxing experience"
→ "amazeballs", "bespoke": likely OOV (not in Amazon review training vocabulary)
→ Sequence: [0, 42, 0, 0, 831, 29] (partial OOV)
```

For a review where 80%+ of content words are OOV, the BiLSTM effectively sees noise.

**Current mitigation:** None. OOV words silently become padding tokens.

**Recommended fix:** Compute OOV ratio before inference:
```python
seq = tokenizer_obj.texts_to_sequences([cleaned])
total_tokens = len(cleaned.split())
oov_count = sum(1 for t in seq[0] if t == 0)
oov_ratio = oov_count / max(total_tokens, 1)
if oov_ratio > 0.5:
    # Return low-confidence Neutral, flag as unreliable
    return "Neutral", 0.5, "high_oov"
```

---

## Failure Mode 11: Sequence Truncation Asymmetry

**Severity:** Medium (affects ~15% of reviews)

**Component:** `pad_sequences(truncating="post")` in `app.py`

**Description:** `truncating="post"` cuts tokens from the end. For a 300-word review structured as "Great features in detail... but one critical flaw at the end," the flaw is in the truncated portion.

**Root cause:** Post-truncation is the default because it's simple and works for most reviews where sentiment is established early. Reviews where the key insight is at the end (a common "bury the lede" structure) suffer.

**Recommended fix:** Truncate from the middle, preserving opening and closing context:
```python
if len(seq[0]) > MAX_LENGTH:
    half = MAX_LENGTH // 2
    seq_truncated = seq[0][:half] + seq[0][-(MAX_LENGTH - half):]
else:
    seq_truncated = seq[0]
```
This preserves the first 75 and last 75 tokens, capturing both the initial framing and the conclusion.

---

## Mitigation Status Summary Table

| Failure Mode | Severity | Affected Component | Current Mitigation | Status | Recommended Fix |
|---|---|---|---|---|---|
| Sarcasm | High | BiLSTM | LLM path handles better | Partial | Sarcasm pre-classifier |
| Mixed sentiment | High | BiLSTM | ABSA provides per-aspect breakdown | Partial | Aspect-weighted headline |
| Long reviews (>150 tokens) | Medium | BiLSTM | MAX_REVIEW_LENGTH=2000 bounds worst case | Partial | Sliding window or middle-truncation |
| Ambiguous aspect references | Medium | spaCy extraction | LLM path resolves via inference | Partial | Entity linking for transformer path |
| LLM hallucinated aspects | Medium | LLM ABSA | System prompt instructions | Partial | Post-hoc verification against source text |
| Malformed LLM responses | Low | LLM modules | Retry + fallback chain | Mitigated | Stricter stop sequences |
| Ollama crash mid-session | High | LLM modules | Retry + fallback chain | Mitigated | Background health monitor thread |
| Request timeout (CPU) | Medium | LLM modules | MAX_REVIEW_LENGTH bounds max time | Partial | ThreadPoolExecutor timeout |
| NaN inference output | Low | BiLSTM | None | Open | `np.isnan()` guard → default 0.5 |
| OOV tokens | Low-Medium | Tokenizer | None | Open | OOV ratio check before inference |
| End-truncation | Medium | pad_sequences | None | Open | Middle-truncation strategy |

---

## Impact by Analysis Path

| Failure Mode | Transformer Path | LLM Path |
|---|---|---|
| Sarcasm | Fails (BiLSTM) | Handles (LLM context) |
| Mixed sentiment | Partial (ABSA helps) | Better (LLM ABSA) |
| Ambiguous aspects | Fails (spaCy) | Handles (LLM inference) |
| LLM hallucinations | N/A | Present |
| Malformed output | N/A | Mitigated (retry) |
| Ollama crash | N/A | Mitigated (fallback) |
| Timeout | Fast, no timeout risk | Slow CPU path at risk |

**Key insight:** The LLM path has better NLP quality (sarcasm, mixed sentiment, ambiguous aspects) but introduces new failure modes (hallucination, timeout, process crashes). The transformer path is more predictable but has lower ceiling quality. The two-path design with fallback provides the best of both: LLM quality when available, transformer reliability as fallback.
