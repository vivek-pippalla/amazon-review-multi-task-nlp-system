# API Reference

Amazon Review Sentiment ABSA — REST API Reference  
Base URL: `http://localhost:5000` (development)

---

## Global Constraints

| Constraint | Value |
|---|---|
| `MAX_REVIEW_LENGTH` | 2000 characters |
| `MAX_BATCH_SIZE` | 10 reviews per call |
| Content-Type (requests) | `application/json` |
| Content-Type (responses) | `application/json` |
| Authentication | None (current version) |
| Rate limiting | None (current version — see Production Gaps) |
| HTTP methods used | `GET`, `POST` |

### Common Error Format

All error responses use a single-key JSON object:

```json
{"error": "Human-readable description of the problem"}
```

### HTTP Status Codes

| Code | Meaning |
|---|---|
| `200 OK` | Request succeeded (including batch with partial failures) |
| `400 Bad Request` | Validation failed — missing field, wrong type, or constraint exceeded |
| `500 Internal Server Error` | Unhandled exception in the pipeline or database layer |

### Production Gaps (Noted for Engineering Review)

- No authentication layer — any client on the network can call any endpoint.
- No rate limiting — a single client can saturate the NLP pipeline.
- No request-ID header for distributed tracing across log lines.
- Batch endpoint returns HTTP 200 even on 100% item failure; callers must inspect `failed` count.

---

## Endpoints

### 1. `POST /analyze`

**Full analysis pipeline with database persistence.** Designed for the web UI. Runs sentiment
classification, summarization, and ABSA, then writes the result to MySQL. Duplicate reviews
(same text) are detected by SHA-256 hash and not re-inserted; the existing record's ID is
returned internally but is not surfaced to the client.

#### Request

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `review` | string | Yes | 1–2000 characters after strip | The raw product review text |

```json
{
  "review": "This phone has an amazing camera and excellent battery life. The display is crisp and bright. Build quality feels premium but it runs a bit warm under load."
}
```

#### Response — 200 OK

| Field | Type | Description |
|---|---|---|
| `Overall Sentiment` | string | `"Positive"`, `"Neutral"`, or `"Negative"` |
| `Confidence Score` | number | Raw BiLSTM sigmoid output, 4 decimal places (0.0–1.0) |
| `Summary` | string | Abstractive summary from T5-base or llama3.2 |
| `Aspect-based Sentiments` | object | Map of `{aspect_name: sentiment_label}` pairs |

```json
{
  "Overall Sentiment": "Positive",
  "Confidence Score": 0.8734,
  "Summary": "The reviewer praises the camera quality and battery life, noting a crisp and bright display. Build quality is considered premium, though the device runs warm under load.",
  "Aspect-based Sentiments": {
    "camera": "Positive",
    "battery life": "Positive",
    "display": "Positive",
    "build quality": "Positive",
    "performance": "Neutral"
  }
}
```

#### Error Responses

| Status | Condition | Body |
|---|---|---|
| `400` | `review` key missing or empty after strip | `{"error": "No review text provided"}` |
| `400` | Review length > 2000 chars | `{"error": "Review exceeds 2000 characters"}` |
| `500` | Pipeline or database exception | `{"error": "<exception message>"}` |

---

### 2. `POST /api/sentiment`

**Pipeline-only analysis, no database write.** Programmatic API for integrations that need
on-demand scoring without persisting data. Identical validation and pipeline to `/analyze`; the
only difference is that results are never written to MySQL.

#### Request

Same schema as `POST /analyze`.

```json
{
  "review": "The packaging was terrible — everything arrived damaged and the customer service was unhelpful."
}
```

#### Response — 200 OK

Same schema as `POST /analyze`. The `_source` field is stripped internally before the response
is serialized.

```json
{
  "Overall Sentiment": "Negative",
  "Confidence Score": 0.0312,
  "Summary": "The reviewer reports poor packaging that caused damaged goods on arrival, compounded by unhelpful customer service.",
  "Aspect-based Sentiments": {
    "packaging": "Negative",
    "customer service": "Negative"
  }
}
```

#### Error Responses

Same as `POST /analyze`.

---

### 3. `POST /api/batch/analyze`

**Batch analysis with persistence, up to 10 reviews per call.** Each review is processed
independently. A validation or pipeline failure for one item does not halt processing of the
remaining items. All items — successful and failed — are included in the `results` array.

HTTP status is always `200` when the batch-level validation passes. Per-item failure is
communicated via `status: "failed"` and the top-level `failed` counter.

#### Request

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `reviews` | array of strings | Yes | 1–10 items | List of raw review texts |

```json
{
  "reviews": [
    "Great product, fast delivery, works perfectly.",
    "Absolute garbage. Stopped working after 3 days.",
    ""
  ]
}
```

#### Response — 200 OK

| Field | Type | Description |
|---|---|---|
| `total` | integer | Total number of items submitted (length of input array) |
| `succeeded` | integer | Number of items that completed the pipeline and were saved |
| `failed` | integer | Number of items that failed at any stage |
| `results` | array | One entry per submitted review, in original order |

Each `results` entry:

| Field | Type | Present when | Description |
|---|---|---|---|
| `index` | integer | Always | Zero-based position in the input array |
| `status` | string | Always | `"success"` or `"failed"` |
| `data` | object | `status == "success"` | Same schema as single-review success response |
| `error` | string | `status == "failed"` | Human-readable reason for failure |

```json
{
  "total": 3,
  "succeeded": 2,
  "failed": 1,
  "results": [
    {
      "index": 0,
      "status": "success",
      "data": {
        "Overall Sentiment": "Positive",
        "Confidence Score": 0.9102,
        "Summary": "The reviewer found the product excellent with fast delivery and flawless performance.",
        "Aspect-based Sentiments": {
          "delivery": "Positive"
        }
      }
    },
    {
      "index": 1,
      "status": "success",
      "data": {
        "Overall Sentiment": "Negative",
        "Confidence Score": 0.0218,
        "Summary": "The reviewer describes the product as poor quality, ceasing to function within three days of purchase.",
        "Aspect-based Sentiments": {
          "durability": "Negative"
        }
      }
    },
    {
      "index": 2,
      "status": "failed",
      "error": "Empty or non-string review"
    }
  ]
}
```

#### Error Responses

| Status | Condition | Body |
|---|---|---|
| `400` | `reviews` key missing or not a list | `{"error": "'reviews' must be a non-empty list"}` |
| `400` | More than 10 items | `{"error": "Batch size exceeds the maximum of 10"}` |

Per-item failure reasons (in `results[n].error`):

| Reason | Condition |
|---|---|
| `"Empty or non-string review"` | Item is not a string, or is blank after strip |
| `"Exceeds 2000 character limit"` | Item length after strip > 2000 |
| Pipeline exception message | `complete_pipeline()` or `save_review_with_absa()` raises |

---

### 4. `GET /api/stats`

**System-wide aggregate statistics.** Queries three separate counts from the database. No
query parameters. Intended for dashboard/monitoring use.

#### Request

No body. No query parameters.

```
GET /api/stats
```

#### Response — 200 OK

| Field | Type | Description |
|---|---|---|
| `total_reviews` | integer | `COUNT(*)` from the `reviews` table |
| `sentiment_distribution` | object | Map of `{sentiment_label: count}` — only labels present in DB appear |
| `total_aspects_analyzed` | integer | `COUNT(*)` from the `absa_results` table |
| `most_recent_analysis` | string or null | ISO 8601 timestamp of the most recent review row; `null` if table is empty |

```json
{
  "total_reviews": 248,
  "sentiment_distribution": {
    "Positive": 142,
    "Negative": 74,
    "Neutral": 32
  },
  "total_aspects_analyzed": 1037,
  "most_recent_analysis": "2025-06-01T14:22:08"
}
```

#### Error Responses

| Status | Condition | Body |
|---|---|---|
| `500` | Database query fails | `{"error": "<exception message>"}` |

---

### 5. `GET /api/aspects/trending`

**Most-mentioned product aspects within a rolling date window.** Runs a JOIN between
`absa_results` and `reviews`, filtering by `created_at`, grouping by aspect name, and ordering
by total mention count descending.

#### Query Parameters

| Parameter | Type | Default | Min | Max | Description |
|---|---|---|---|---|---|
| `limit` | integer | `10` | `1` | `50` | Maximum number of aspects to return |
| `days` | integer | `30` | `1` | `365` | Look-back window in days (from `NOW()`) |

Out-of-range values are silently clamped to min/max rather than returning a 400.

```
GET /api/aspects/trending?limit=5&days=7
```

#### Response — 200 OK

| Field | Type | Description |
|---|---|---|
| `days` | integer | Effective look-back window used (after clamping) |
| `trending_aspects` | array | List of aspect objects, ordered by `total_mentions` descending |

Each aspect object:

| Field | Type | Description |
|---|---|---|
| `aspect` | string | Aspect name as stored in `absa_results.aspect` |
| `total_mentions` | integer | Total rows for this aspect in the date window |
| `positive` | integer | Count where `sentiment = 'Positive'` |
| `negative` | integer | Count where `sentiment = 'Negative'` |
| `neutral` | integer | Count where `sentiment = 'Neutral'` |

```json
{
  "days": 7,
  "trending_aspects": [
    {
      "aspect": "battery life",
      "total_mentions": 38,
      "positive": 24,
      "negative": 11,
      "neutral": 3
    },
    {
      "aspect": "camera",
      "total_mentions": 31,
      "positive": 28,
      "negative": 2,
      "neutral": 1
    },
    {
      "aspect": "display",
      "total_mentions": 27,
      "positive": 19,
      "negative": 5,
      "neutral": 3
    }
  ]
}
```

#### Error Responses

| Status | Condition | Body |
|---|---|---|
| `500` | Database query fails | `{"error": "<exception message>"}` |

---

### 6. `GET /api/reviews`

**Paginated review history with optional sentiment filter.** Returns stored reviews in reverse
chronological order. The `sentiment` filter uses the `idx_sentiment` index to avoid a full
table scan.

#### Query Parameters

| Parameter | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `page` | integer | `1` | Minimum 1 | Page number (1-indexed) |
| `limit` | integer | `20` | 1–100 | Reviews per page |
| `sentiment` | string | `all` | `Positive`, `Negative`, `Neutral`, or omit | Filter by sentiment label; any other value treated as no filter |

```
GET /api/reviews?page=2&limit=5&sentiment=Negative
```

#### Response — 200 OK

| Field | Type | Description |
|---|---|---|
| `reviews` | array | Page of review objects |
| `pagination` | object | Pagination metadata |

Each review object:

| Field | Type | Description |
|---|---|---|
| `id` | integer | Primary key from `reviews` table |
| `review_text` | string | Original review text as submitted |
| `summarized_review` | string or null | T5 or LLM-generated summary; null if not generated |
| `overall_sentiment` | string | `"Positive"`, `"Neutral"`, or `"Negative"` |
| `confidence_score` | number or null | BiLSTM sigmoid output (0.0–1.0), 4 decimal places |
| `created_at` | string | ISO 8601 timestamp of insertion |

Pagination object:

| Field | Type | Description |
|---|---|---|
| `page` | integer | Current page (as requested, after floor at 1) |
| `limit` | integer | Effective page size (after clamping to 1–100) |
| `total` | integer | Total matching rows (respects `sentiment` filter) |
| `pages` | integer | Total number of pages; minimum 1 even when table is empty |

```json
{
  "reviews": [
    {
      "id": 102,
      "review_text": "Terrible product. The screen cracked within a week and support was useless.",
      "summarized_review": "The reviewer reports screen failure within one week and describes customer support as unhelpful.",
      "overall_sentiment": "Negative",
      "confidence_score": 0.0411,
      "created_at": "2025-05-30T09:14:37"
    },
    {
      "id": 98,
      "review_text": "Horrible build quality. Feels like plastic junk.",
      "summarized_review": "The reviewer criticises the build quality, describing the product as cheaply constructed.",
      "overall_sentiment": "Negative",
      "confidence_score": 0.0189,
      "created_at": "2025-05-28T16:02:11"
    }
  ],
  "pagination": {
    "page": 2,
    "limit": 5,
    "total": 74,
    "pages": 15
  }
}
```

#### Error Responses

| Status | Condition | Body |
|---|---|---|
| `500` | Database query fails | `{"error": "<exception message>"}` |

---

## Appendix: Response Field Cross-Reference

| Field | `/analyze` | `/api/sentiment` | `/api/batch/analyze` (per item) | `/api/stats` | `/api/aspects/trending` | `/api/reviews` |
|---|---|---|---|---|---|---|
| `Overall Sentiment` | Yes | Yes | Yes (in `data`) | — | — | `overall_sentiment` |
| `Confidence Score` | Yes | Yes | Yes (in `data`) | — | — | `confidence_score` |
| `Summary` | Yes | Yes | Yes (in `data`) | — | — | `summarized_review` |
| `Aspect-based Sentiments` | Yes | Yes | Yes (in `data`) | — | — | — |
| `total_reviews` | — | — | — | Yes | — | — |
| `trending_aspects` | — | — | — | — | Yes | — |
| `pagination` | — | — | — | — | — | Yes |
