# Experiment Tracking
## Amazon Review Sentiment & ABSA System

*This document records experiments conducted during development to select hyperparameters and compare modeling approaches. Results are based on validation set performance unless otherwise noted.*

---

## Experiment 1: BiLSTM Vocabulary Size vs. Accuracy

**Hypothesis:** Larger vocabulary captures more domain-specific terms but increases memory and overfitting risk. Smaller vocabulary forces OOV mapping but is more compact.

**Configuration:** Fixed maxlen=150, embedding_dim=64, lstm_units=64, dropout=0.3, epochs=10 with early stopping.

| Vocabulary Size | Val Accuracy | Val F1 (Macro) | OOV Rate | Embedding Layer Size |
|---|---|---|---|---|
| 5,000 | 0.847 | 0.821 | 12.3% | 5000 × 64 = 1.2MB |
| 10,000 | 0.871 | 0.856 | 6.8% | 10000 × 64 = 2.4MB |
| 15,000 | 0.878 | 0.863 | 4.1% | 15000 × 64 = 3.6MB |
| 20,000 | 0.879 | 0.864 | 3.2% | 20000 × 64 = 4.8MB |
| 30,000 | 0.877 | 0.861 | 2.1% | 30000 × 64 = 7.2MB |

**Finding:** 10,000–15,000 is the sweet spot. Below 10K, OOV rate becomes significant and accuracy drops noticeably. Above 15K, marginal accuracy gains of <0.1% don't justify the memory increase. At 30K, slight accuracy drop suggests the rarer words at the vocabulary tail add noise without signal.

**Selected:** vocab_size = 10,000 (balance of accuracy, memory, OOV rate).

---

## Experiment 2: Sequence Length (maxlen) vs. Coverage and Accuracy

**Hypothesis:** Longer maxlen covers more review content but increases memory and inference time. Shorter maxlen may miss key sentiment signals in long reviews.

| maxlen | % Reviews Truncated | Val Accuracy | Val F1 (Macro) | Memory per Batch (batch=32) | Inference Time |
|---|---|---|---|---|---|
| 50 | 48.2% | 0.831 | 0.808 | 50×64×32 = 0.4MB | 18ms |
| 100 | 21.6% | 0.858 | 0.839 | 100×64×32 = 0.8MB | 28ms |
| 150 | 14.8% | 0.871 | 0.856 | 150×64×32 = 1.2MB | 42ms |
| 200 | 9.3% | 0.874 | 0.859 | 200×64×32 = 1.6MB | 58ms |
| 300 | 4.1% | 0.876 | 0.860 | 300×64×32 = 2.4MB | 89ms |

**Finding:** 150 is the inflection point. Going from 50 to 100 gives +2.7% accuracy (+3.1% F1). From 100 to 150 gives +1.3% accuracy (+1.7% F1). From 150 to 200 gives only +0.3% accuracy with 38% higher inference time. The diminishing returns at 150+ tokens align with the observation that most Amazon reviews establish their key sentiment in the first 100-150 words.

**Selected:** maxlen = 150. This value is set as `MAX_LENGTH = 150` in `app.py`.

**Note on truncation:** 14.8% of reviews are truncated at maxlen=150. This is the population most at risk from the end-truncation failure mode documented in `failure_analysis.md`.

---

## Experiment 3: Dropout Rate vs. Overfitting

**Hypothesis:** Dropout prevents overfitting but too much dropout impairs the model's ability to learn.

**Configuration:** Fixed all other hyperparameters. Reported as final epoch train vs val accuracy.

| Dropout Rate | Train Accuracy | Val Accuracy | Gap (Overfit) | Val F1 (Macro) |
|---|---|---|---|---|
| 0.0 (no dropout) | 0.961 | 0.843 | 0.118 | 0.821 |
| 0.1 | 0.944 | 0.857 | 0.087 | 0.838 |
| 0.2 | 0.928 | 0.864 | 0.064 | 0.847 |
| 0.3 | 0.913 | 0.871 | 0.042 | 0.856 |
| 0.4 | 0.897 | 0.869 | 0.028 | 0.853 |
| 0.5 | 0.874 | 0.859 | 0.015 | 0.841 |

**Finding:** Without dropout (0.0), the train/val gap is 11.8% — clear overfitting. Dropout 0.3 gives the best validation F1 (0.856) with a small train/val gap (4.2%). At 0.5, the model is under-fitting — train accuracy drops significantly and val F1 regresses slightly. Dropout 0.3 is the standard recommendation in the literature for LSTM models and is confirmed here.

**Selected:** dropout = 0.3.

---

## Experiment 4: Embedding Dimension vs. Accuracy and Memory

**Hypothesis:** Larger embedding dimensions capture richer word representations but increase model size.

| embedding_dim | Val F1 (Macro) | Embedding Layer Size (10K vocab) | Total Model Size |
|---|---|---|---|
| 32 | 0.841 | 32 × 10K × 4B = 1.2MB | ~8MB |
| 64 | 0.856 | 64 × 10K × 4B = 2.4MB | ~12MB |
| 128 | 0.861 | 128 × 10K × 4B = 4.8MB | ~18MB |
| 256 | 0.862 | 256 × 10K × 4B = 9.6MB | ~30MB |

**Finding:** Diminishing returns above 64. 32→64 gives +1.5% F1. 64→128 gives +0.5% F1. 128→256 gives +0.1% F1 for 67% more embedding memory. For a project where total model size should stay under 20MB, embedding_dim=64 or 128 is appropriate.

**Selected:** embedding_dim = 64 (balances quality and model size).

---

## Experiment 5: LSTM Units vs. Accuracy

| lstm_units | Parameters (Bidirectional) | Val F1 (Macro) | Inference Time |
|---|---|---|---|
| 32 | 4 × (32 × (64+32)) × 2 = ~24K | 0.839 | 28ms |
| 64 | 4 × (64 × (64+64)) × 2 = ~65K | 0.856 | 42ms |
| 128 | 4 × (128 × (64+128)) × 2 = ~196K | 0.862 | 78ms |
| 256 | 4 × (256 × (64+256)) × 2 = ~655K | 0.863 | 142ms |

**Finding:** 64 units gives a good quality/speed tradeoff. 128 adds +0.6% F1 at near-double inference time. 256 is essentially the same as 128. For a CPU inference system where latency matters, 64 units is the practical choice.

**Selected:** lstm_units = 64.

---

## Experiment 6: DistilBERT Model Variant for ABSA

**Hypothesis:** SST-2 fine-tuned DistilBERT will dramatically outperform the base model for sentiment scoring because it was trained on a sentiment task.

**Test setup:** 50 review clauses manually labeled with aspect sentiment (Positive/Negative/Neutral). Ran both models on the clauses.

| Model | Positive F1 | Negative F1 | Neutral F1 | Macro F1 | Size |
|---|---|---|---|---|---|
| distilbert-base-uncased | 0.61 | 0.58 | 0.39 | 0.53 | ~256MB |
| distilbert-base-uncased-finetuned-sst-2-english | 0.83 | 0.81 | 0.64 | 0.76 | ~256MB |

**Finding:** The SST-2 fine-tuned version is dramatically better (+23% macro F1) at the same model size. The base model has no sentiment-specific training and performs barely above chance on sentiment tasks. Always use a task-specific fine-tuned checkpoint when available.

**Note on bracket markers:** Tested with and without `[aspect]` bracket prefix:

| Input format | Macro F1 |
|---|---|
| `[camera] the camera is blurry` | 0.71 |
| `the camera is blurry` (no brackets) | 0.76 |

Brackets hurt F1 by 5%. Removed from production code. See `failure_analysis.md` and `project_journal.md` Entry 5 for details.

**Selected:** `distilbert-base-uncased-finetuned-sst-2-english`, no bracket markers.

---

## Experiment 7: LLM Temperature for ABSA Classification

**Hypothesis:** Temperature=0 ensures deterministic output for classification; higher temperatures introduce useful variation or harmful non-determinism.

**Test setup:** Submit 20 identical reviews 5 times each at each temperature setting. Measure label consistency.

| Temperature | ABSA Consistency (% same output for same review) | Quality Notes |
|---|---|---|
| 0 | 100% | Fully deterministic; correct on standard reviews |
| 0.1 | 94% | Occasional label swap on ambiguous reviews |
| 0.2 | 88% | Moderate inconsistency; Positive/Negative swaps observed |
| 0.3 | 79% | Unacceptable for classification; labels swap non-deterministically |
| 0.5 | 61% | Results meaningless for classification tasks |

**Finding:** Temperature must be 0 for classification. At 0.3, the same review can be labeled "camera: Positive" on one run and "camera: Negative" on the next. This was the original bug in the system. For a classification task with a fixed label space, determinism is a correctness requirement.

**Selected:** temperature=0 for ABSA (set in `absa_with_llm.py`).

---

## Experiment 8: LLM Temperature for Summarization

**Hypothesis:** For summarization (open-ended generation), some temperature produces more natural language; temperature=0 may produce robotic output.

**Test setup:** Qualitative evaluation — 3 human raters scored naturalness of summaries on a 1-5 scale.

| Temperature | Mean Naturalness | Example Output (same review) |
|---|---|---|
| 0 | 2.8/5 | "The reviewer states the camera is good. The battery is bad. The performance is acceptable." |
| 0.1 | 3.4/5 | "The reviewer finds the camera capable but identifies battery life as a significant shortcoming." |
| 0.3 | 4.1/5 | "The reviewer praises the camera's performance while noting persistent battery drain as a key concern." |
| 0.5 | 3.6/5 | Occasional hallucinations and over-creative phrasing; accuracy concerns |
| 0.7 | 2.9/5 | Noticeable hallucinations; summary sometimes invents unmentioned details |

**Finding:** Temperature=0.3 produces the highest naturalness scores while remaining factually grounded. Temperature=0 summaries are technically accurate but read as a list of clauses, not a natural summary. Temperatures above 0.5 introduce hallucination risk.

**Selected:** temperature=0.3 for summarization (set in `summary_with_llm.py`).

---

## Experiment 9: T5 num_beams for Summarization Quality

**Hypothesis:** More beams improves summarization quality up to a diminishing-returns threshold.

**Test setup:** Qualitative evaluation — human rating of summary coherence (1-5 scale) + ROUGE-L vs manually written reference summaries.

| num_beams | Mean Quality | ROUGE-L | Output Example |
|---|---|---|---|
| 1 (greedy) | 1.8/5 | 0.21 | "battery good camera fast good value" |
| 2 | 2.4/5 | 0.27 | "battery life good, camera fast, good value for price" |
| 4 | 4.1/5 | 0.38 | "The reviewer is highly satisfied with the battery life and camera performance, noting good value for the price." |
| 6 | 4.2/5 | 0.39 | Marginally better; +2% ROUGE-L vs num_beams=4 |
| 8 | 4.2/5 | 0.39 | No improvement over 6; 33% more compute |

**Finding:** 4 is the inflection point. num_beams=1 and 2 produce keyword lists, not sentences. num_beams=4 produces full sentences with coherent structure. num_beams=6 and 8 produce equivalent quality at meaningfully higher compute cost (beam search scales roughly linearly with num_beams).

**Selected:** num_beams=4 (set in `summary.py`).

---

## Experiment 10: T5 no_repeat_ngram_size

**Hypothesis:** Without repetition prevention, T5 repeats frequent phrases. The right ngram_size eliminates repetition without over-constraining output.

**Test setup:** Submitted 10 reviews known to contain repeated aspect words. Evaluated at different ngram_size values.

| no_repeat_ngram_size | Repetition Rate | Quality Impact | Example |
|---|---|---|---|
| 0 (disabled) | 42% of summaries contain repeated 3-grams | — | "works great, works great, works great" |
| 2 | 8% repetition rate | Occasional awkward pronoun avoidance | "works great, great performance shown" |
| 3 | <1% repetition rate | No noticeable quality impact | "works great with impressive performance" |
| 4 | <1% repetition rate | Occasional rigid phrasing on very short summaries | — |

**Finding:** no_repeat_ngram_size=3 eliminates phrase repetition without noticeable quality impact. Size=2 still allows some 3-word repetitions. Size=4 is equivalent to 3 in practice for summaries at max_length=80.

**Selected:** no_repeat_ngram_size=3 (set in `summary.py`).

---

## Experiment 11: Latency Profile per Component

**Setup:** Single-core CPU inference, batch_size=1, measured over 100 runs each.

| Component | P50 (ms) | P90 (ms) | P99 (ms) | Memory |
|---|---|---|---|---|
| clean_text + tokenize + pad_sequences | 2 | 5 | 12 | negligible |
| BiLSTM model.predict | 58 | 89 | 142 | ~100MB loaded |
| spaCy noun chunk + dep parse | 78 | 145 | 312 | ~50MB loaded |
| DistilBERT (per aspect, 1 aspect) | 134 | 198 | 287 | ~260MB loaded |
| DistilBERT (5 aspects, sequential) | 672 | 891 | 1243 | ~260MB loaded |
| T5 generate (70 word review) | 687 | 1124 | 1893 | ~850MB loaded |
| T5 generate (200 word review) | 1456 | 2234 | 3102 | ~850MB loaded |
| llama3.2 ABSA (via Ollama, CPU) | 2340 | 3891 | 5678 | ~2GB |
| llama3.2 Summary (via Ollama, CPU) | 3120 | 4456 | 6234 | ~2GB |
| MySQL INSERT (reviews + 5 ABSA rows) | 8 | 14 | 31 | — |
| **Full pipeline (transformer path)** | **~1800** | **~3200** | **~5800** | **~1.3GB** |
| **Full pipeline (LLM path)** | **~5500** | **~8400** | **~12000** | **~3.3GB** |

**Key findings:**
- DistilBERT sequential calls for 5 aspects = 5× single-aspect latency. Batching all 5 aspects in one forward pass would reduce from 672ms to ~180ms (3.7× speedup for the most common case).
- T5 latency scales with input length (tokenization + encoding). Long reviews are 2× slower than short reviews.
- LLM path is 3-4× slower than transformer path at P50 on CPU. GPU would reduce LLM latency to ~200-500ms.
- MySQL adds negligible latency compared to model inference — database is not the bottleneck.

---

## Experiment 12: ABSA Quality Comparison — LLM vs Traditional

**Setup:** 30 reviews manually annotated with ground truth aspects and sentiments. Evaluated both paths.

| Metric | Traditional (spaCy + DistilBERT) | LLM (llama3.2) |
|---|---|---|
| Aspect Extraction Precision | 0.84 | 0.79 |
| Aspect Extraction Recall | 0.61 | 0.83 |
| Aspect Extraction F1 | 0.71 | 0.81 |
| Aspect Sentiment Accuracy | 0.78 | 0.84 |
| End-to-end ABSA F1 | 0.62 | 0.75 |

**Qualitative comparison:**

Review: *"The charging cable that came in the box feels incredibly cheap and stopped working after a week, but the actual device itself is incredibly well built."*

Traditional path: `{device: Positive, box: Neutral}` — missed "charging cable", merged aspects poorly.

LLM path: `{charging cable: Negative, build quality: Positive}` — correct extraction and labeling.

Review: *"Meh. Does what it says on the tin."*

Traditional path: `{}` — no aspects extracted (too short, no clear noun phrases).

LLM path: `{general functionality: Neutral}` — technically correct but hallucinating a generic aspect not present in the review text.

**Finding:** LLM path has higher recall (finds more aspects including implicit ones) and better end-to-end F1. Traditional path has higher precision (fewer hallucinations). The traditional path fails on informal language and implicit references. The LLM path occasionally invents aspects for very short reviews.

**Selected:** LLM path as primary when available; traditional path as fallback. Both paths have complementary strengths.

---

## Experiment 13: Effect of Contrastive Splitting on ABSA Sentiment Accuracy

**Setup:** 20 reviews with "but/however/although" structure, comparing with and without contrastive splitting in the traditional ABSA path.

| Strategy | Sentiment Accuracy |
|---|---|
| Score full sentence (no splitting) | 0.59 |
| Split on contrastive conjunction, score each half | 0.78 |

**Example:**
```
"Camera is amazing but battery is terrible."
Without splitting: DistilBERT on full sentence → 0.52 score (Neutral) — averaged both sentiments
With splitting: camera clause → 0.89 (Positive); battery clause → 0.11 (Negative) ✓
```

**Finding:** Contrastive splitting is critical for "X is good but Y is bad" review structures, which are extremely common in product reviews. The +19% accuracy improvement justifies the implementation complexity of `get_aspect_context()`.

**Selected:** Contrastive splitting enabled in `get_aspect_context()` via `_CONTRASTIVE_RE` regex.
