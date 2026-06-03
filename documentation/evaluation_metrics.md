# Evaluation Metrics
## Amazon Review Sentiment & ABSA System

---

## Overview

This document defines the evaluation metrics appropriate for each component of the system, explains why each metric was chosen, and identifies the current gaps in evaluation coverage. Understanding evaluation is as important as understanding the model — a model cannot be improved without knowing how to measure it.

---

## Part 1: Overall Sentiment Classification

### 1.1 Accuracy

**Definition:**

$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

**When it's useful:** Quick sanity check. Easy to compute without per-class breakdown.

**Why it's insufficient alone:** Consider a dataset where 70% of reviews are Positive. A model that predicts "Positive" for every single input achieves 70% accuracy without learning anything. This is called the majority-class baseline. Accuracy is a misleading metric whenever classes are imbalanced — and Amazon reviews are heavily imbalanced toward Positive.

**Example of accuracy gaming:**
```
Dataset: 700 Positive, 300 Negative (1000 total)
Model A: always predicts Positive → Accuracy = 70%, Recall(Negative) = 0%
Model B: 80% correct on Positive, 75% correct on Negative → Accuracy = 78.5%

Model A looks competitive with Model B on accuracy alone.
Precision/Recall reveals the difference immediately.
```

---

### 1.2 Precision, Recall, and F1

**Confusion matrix terminology:**
- TP (True Positive): model predicted Positive, review is Positive
- FP (False Positive): model predicted Positive, review is Negative
- FN (False Negative): model predicted Negative, review is Positive
- TN (True Negative): model predicted Negative, review is Negative

**Precision:**

$$\text{Precision}_{class} = \frac{TP}{TP + FP}$$

Interpretation: "Of all reviews the model labeled as Positive, what fraction are actually Positive?" Precision measures how trustworthy positive predictions are. Low precision means the model over-predicts the class (lots of false positives).

**Recall:**

$$\text{Recall}_{class} = \frac{TP}{TP + FN}$$

Interpretation: "Of all genuinely Positive reviews in the dataset, what fraction did the model correctly identify?" Recall measures completeness. Low recall means the model misses many true positives (lots of false negatives).

**F1 Score (harmonic mean):**

$$F1_{class} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Interpretation: The harmonic mean penalizes extreme imbalance between precision and recall. A model with Precision=1.0 and Recall=0.1 has F1=0.18 — the poor recall dominates. A model with Precision=0.8 and Recall=0.8 has F1=0.8. F1 is the standard metric for classification with class imbalance.

**Why harmonic mean (not arithmetic mean)?** The arithmetic mean of Precision=1.0 and Recall=0.1 would be 0.55 — misleadingly high for a model that misses 90% of true positives. The harmonic mean gives 0.18, accurately reflecting that the model fails on recall.

---

### 1.3 Macro vs Weighted F1

**Macro F1:** Average F1 across all classes, treating each class equally regardless of frequency.

$$\text{Macro F1} = \frac{F1_{Positive} + F1_{Negative} + F1_{Neutral}}{3}$$

Use when: You care equally about performance on rare and common classes. For this system, correctly classifying a rare Negative review is as important as classifying a common Positive review.

**Weighted F1:** Average F1 weighted by class frequency.

$$\text{Weighted F1} = \frac{N_{Positive} \times F1_{Positive} + N_{Negative} \times F1_{Negative} + N_{Neutral} \times F1_{Neutral}}{N_{total}}$$

Use when: You care more about performance on common classes (user-facing impact is proportional to class frequency).

**Recommendation for this system:** Report both. The gap between macro and weighted F1 reveals how much performance differs across classes. A macro F1 of 0.75 with weighted F1 of 0.85 means the model performs much better on common classes than rare ones.

---

### 1.4 Confidence Score as a Metric

The BiLSTM's sigmoid output is stored as `confidence_score` in the `reviews` table. This raw probability is more information-rich than the discrete label.

**Uses:**
- **Threshold tuning:** Plot precision-recall curves at different thresholds on a labeled test set. Choose the Neutral band boundaries (currently 0.15–0.5) to maximize F1 for the chosen task.
- **Low-confidence filtering:** Predictions near 0.5 (0.4–0.6) are near the decision boundary — genuinely uncertain. Applications that need high-precision predictions can filter for `confidence_score > 0.8` or `confidence_score < 0.2`.
- **Distribution monitoring:** Track the distribution of confidence scores over time. Drift toward 0.5 indicates the model is becoming less certain — a signal for retraining.
- **A/B comparison:** Query `analysis_source='llm'` and `analysis_source='transformer'` separately, compute mean confidence. Higher mean confidence on the LLM path (combined with human evaluation of correctness) would justify preferring the LLM path.

**Calibration:** A well-calibrated model's confidence score should match empirical accuracy. If reviews with `confidence_score=0.8` are actually Positive 80% of the time, the model is calibrated. BiLSTM models trained with binary cross-entropy tend to be reasonably calibrated.

---

## Part 2: Aspect-Based Sentiment Analysis

### 2.1 Aspect Extraction Metrics

ABSA evaluation requires evaluating two sub-tasks independently: aspect extraction and aspect sentiment classification.

**Aspect Extraction Precision:**

$$\text{Extraction Precision} = \frac{\text{Valid Aspects Extracted}}{\text{Total Aspects Extracted}}$$

"Of all aspects the system extracted, how many are valid (actually mentioned in the review)?" Low precision = hallucinated or incorrect aspects.

**Aspect Extraction Recall:**

$$\text{Extraction Recall} = \frac{\text{Valid Aspects Extracted}}{\text{Total Aspects Present in Review}}$$

"Of all aspects actually present in the review, how many did the system find?" Low recall = missed aspects. Recall requires knowing the ground truth — all aspects a human annotator would identify.

**Aspect F1:**

$$\text{Aspect F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

### 2.2 Aspect Sentiment Accuracy

After filtering to correctly extracted aspects (intersection of predicted and true aspects):

$$\text{Aspect Sentiment Accuracy} = \frac{\text{Correctly Classified Aspects}}{\text{Correctly Extracted Aspects}}$$

This measures how accurately the system assigns Positive/Negative/Neutral to aspects it correctly identified.

**Combined ABSA metric (Aspect Sentiment F1):**
For a strict end-to-end evaluation, an aspect-sentiment pair must be both correctly extracted AND correctly classified to count as a true positive.

---

### 2.3 Why ABSA Evaluation is Hard

**No single ground truth:** For the review "Camera is amazing but battery life is terrible, and the screen could be brighter," annotator A might extract [camera, battery life, screen brightness], while annotator B extracts [camera, battery, display]. Both are defensible. Aspect extraction has inherent subjectivity.

**Inter-Annotator Agreement:** Standard practice is to use multiple annotators per review and report Cohen's Kappa (κ) as a measure of agreement:

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

where $P_o$ is observed agreement and $P_e$ is expected chance agreement. κ > 0.6 is considered substantial agreement. For aspect extraction tasks, typical inter-annotator κ is 0.6–0.75.

**Recommended evaluation process:**
1. Select 100 diverse reviews covering short, long, simple, and complex
2. Have two annotators independently identify all aspects and their sentiments
3. Compute κ between annotators — if κ < 0.5, annotation guidelines need clarification
4. Resolve disagreements to create ground truth
5. Evaluate system against ground truth, report precision/recall/F1 for extraction and sentiment

---

## Part 3: Summarization

### 3.1 ROUGE Metrics

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** Compares n-gram overlap between a generated summary and one or more reference summaries.

**ROUGE-1 (unigram recall):**

$$\text{ROUGE-1} = \frac{\text{|Unigrams in Generated} \cap \text{Reference|}}{\text{|Unigrams in Reference|}}$$

Measures how many individual words from the reference appear in the generated summary.

**ROUGE-2 (bigram recall):**

$$\text{ROUGE-2} = \frac{\text{|Bigrams in Generated} \cap \text{Reference|}}{\text{|Bigrams in Reference|}}$$

Measures 2-word phrase overlap. A higher ROUGE-2 than ROUGE-1 indicates the summary preserves phrase structure from the reference.

**ROUGE-L (longest common subsequence):**

$$\text{ROUGE-L} = \frac{\text{LCS(Generated, Reference)}}{\text{|Reference|}}$$

Measures the longest common subsequence, capturing fluency and word order. More flexible than strict n-gram matching.

**ROUGE-1/2/L interpretation for this system:**
- T5-base on news summarization benchmarks (CNN/DailyMail): ROUGE-1 ≈ 0.42, ROUGE-2 ≈ 0.20, ROUGE-L ≈ 0.40
- On product reviews (shorter, different domain): expected lower scores due to domain shift
- Target: ROUGE-L > 0.35 on a held-out set of manually summarized reviews

---

### 3.2 BERTScore

**BERTScore:** Uses a pre-trained BERT model to compute semantic similarity between generated and reference summary via token-level cosine similarity of embeddings.

$$\text{BERTScore}_F = \frac{2 \times P_{BERT} \times R_{BERT}}{P_{BERT} + R_{BERT}}$$

where $P_{BERT}$ = average max cosine similarity for each generated token to a reference token, $R_{BERT}$ = average max cosine similarity for each reference token to a generated token.

**Why BERTScore matters here:** ROUGE penalizes paraphrases. "The reviewer is satisfied with the camera" vs "The user liked the camera quality" — ROUGE-1 gives low overlap despite identical meaning. BERTScore captures semantic similarity via embeddings, recognizing these as near-equivalent. For evaluating T5 vs LLM summaries (which often paraphrase differently), BERTScore is more appropriate than ROUGE.

---

### 3.3 BLEU

**BLEU (Bilingual Evaluation Understudy):** Precision-focused metric that measures n-gram overlap between generated text and reference, with a brevity penalty.

$$\text{BLEU} = BP \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

**Why BLEU is less appropriate here:** BLEU was designed for machine translation where there is often a single correct translation. Summarization has many acceptable paraphrases. BLEU penalizes summaries that correctly convey meaning using different words than the reference. ROUGE-L and BERTScore are preferred for summarization evaluation.

---

## Part 4: Metrics by Task — Quick Reference

| Task | Primary Metric | Secondary Metric | Why |
|---|---|---|---|
| Overall Sentiment | Macro F1 | Weighted F1 | Imbalanced classes |
| Overall Sentiment | Per-class F1 | Confusion matrix | Reveals class-specific weaknesses |
| ABSA Extraction | F1 (precision+recall) | Inter-annotator κ | No single ground truth |
| ABSA Sentiment | Aspect Sentiment Accuracy | End-to-end ABSA F1 | Conditional on correct extraction |
| Summarization | ROUGE-L | BERTScore | Paraphrase tolerance needed |
| All | Confidence Score distribution | — | Calibration + drift detection |

---

## Part 5: Current Evaluation Gaps

### Gap 1: No Held-Out Test Set Evaluation

The current system has no formal evaluation on labeled test data. The model is trained and deployed without a quantitative accuracy measurement. This is the most significant evaluation gap.

**Impact:** There is no number to cite when asked "how accurate is your sentiment model?" The qualitative answer is "it works well on typical reviews" — but this cannot be verified without a labeled test set.

**Recommended action:** Create a test set of 500 labeled reviews (manually annotated sentiment + aspects + reference summaries). Evaluate once before any model update. Report macro F1 for sentiment, aspect extraction F1, and ROUGE-L for summarization.

### Gap 2: No A/B Evaluation Between LLM and Transformer Paths

The `analysis_source` column records which path analyzed each review, but no automated comparison of quality between paths has been run. We know the LLM path handles sarcasm and ambiguous references better (from manual inspection), but we don't know by how much.

**Recommended action:** Sample 100 reviews from each path in the DB. Have a human annotator label sentiment and aspects. Compute F1 for each path separately. Report the quality gap.

### Gap 3: No Calibration Check

The sigmoid output `confidence_score` is assumed to be calibrated (0.8 probability = 80% empirical accuracy), but this has not been verified.

**Recommended action:** Group predictions by confidence buckets (0.5–0.6, 0.6–0.7, etc.). For each bucket, compute empirical accuracy on labeled reviews. Plot a calibration curve (reliability diagram). If the curve deviates from the diagonal, apply Platt scaling or isotonic regression to recalibrate.

### Gap 4: No Domain Drift Monitoring

The model was trained on Amazon reviews from a particular time period and product category distribution. If the deployed system receives reviews from different categories or time periods (post-training product language evolving), accuracy may degrade silently.

**Recommended action:** Track vocabulary distribution weekly. Alert if the fraction of OOV tokens increases significantly — this indicates domain drift. Re-evaluate on a fresh labeled sample quarterly.

---

## Part 6: Evaluation Formulas Summary

| Metric | Formula | Range | Higher = Better |
|---|---|---|---|
| Accuracy | TP+TN / Total | 0–1 | Yes |
| Precision | TP / (TP+FP) | 0–1 | Yes |
| Recall | TP / (TP+FN) | 0–1 | Yes |
| F1 | 2PR/(P+R) | 0–1 | Yes |
| Macro F1 | Mean F1 across classes | 0–1 | Yes |
| Weighted F1 | Freq-weighted F1 | 0–1 | Yes |
| Cohen's Kappa | (Po-Pe)/(1-Pe) | -1 to 1 | Yes (>0.6 target) |
| ROUGE-1 | Unigram recall vs reference | 0–1 | Yes |
| ROUGE-2 | Bigram recall vs reference | 0–1 | Yes |
| ROUGE-L | LCS recall vs reference | 0–1 | Yes |
| BERTScore-F1 | Token embedding cosine sim | 0–1 | Yes |
| BLEU | Precision n-gram + brevity | 0–1 | Yes (not recommended here) |
