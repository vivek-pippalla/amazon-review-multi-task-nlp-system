import os
import re
import pickle
import hashlib
import traceback

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from db_connection import get_cursor, commit, get_connection, close_connection

load_dotenv()

app = Flask(__name__)

MAX_REVIEW_LENGTH = 2000
MAX_BATCH_SIZE = 10

# ----------------------------------------
# LLM detection + module selection
# ----------------------------------------
from llm_check import LLM_AVAILABLE, initialize_llm
initialize_llm()

if LLM_AVAILABLE:
    from absa_with_llm import aspect_based_sentiment_llm as aspect_based_sentiment
    from summary_with_llm import generate_summary
    ANALYSIS_SOURCE = "llm"
else:
    from absa import aspect_based_sentiment_improved as aspect_based_sentiment
    from summary import generate_summary
    ANALYSIS_SOURCE = "transformer"

# ----------------------------------------
# ML model + tokenizer
# ----------------------------------------
MODEL_PATH = "best_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)  # type: ignore

if not os.path.exists(TOKENIZER_PATH):
    raise RuntimeError(f"Tokenizer file not found: {TOKENIZER_PATH}")
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer_obj = pickle.load(f)

MAX_LENGTH = 150

# ----------------------------------------
# Per-request DB teardown
# ----------------------------------------
@app.teardown_appcontext
def teardown_db(exception):
    try:
        close_connection()
    except Exception:
        pass

# ----------------------------------------
# Helpers
# ----------------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def hash_review(text: str) -> str:
    """SHA-256 of the normalised review — used as a fast unique key."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def complete_pipeline(review: str) -> dict:
    cleaned = clean_text(review)
    seq = tokenizer_obj.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding="post", truncating="post")

    pred = float(model.predict(padded, verbose=0)[0][0])

    if pred > 0.5:
        overall_sentiment = "Positive"
    elif pred >= 0.15:
        overall_sentiment = "Neutral"
    else:
        overall_sentiment = "Negative"

    summary = generate_summary(review)
    aspects = aspect_based_sentiment(review)

    return {
        "Overall Sentiment": overall_sentiment,
        "Confidence Score": round(pred, 4),
        "Summary": summary,
        "Aspect-based Sentiments": aspects,
        "_source": ANALYSIS_SOURCE,
    }


# ----------------------------------------
# DB write
# ----------------------------------------
def save_review_with_absa(review, summary, overall_sentiment, confidence_score, aspects, source):
    """
    Insert a new review + its ABSA results.
    If the review hash already exists the write is skipped entirely —
    avoids the duplicate-ABSA-row bug that existed in the original code.
    """
    review_hash = hash_review(review)
    cursor = get_cursor()
    try:
        cursor.execute("SELECT id FROM reviews WHERE review_hash = %s", (review_hash,))
        existing = cursor.fetchone()

        if existing:
            return existing["id"]

        cursor.execute(
            """
            INSERT INTO reviews
                (review_hash, review_text, summarized_review, overall_sentiment, confidence_score, analysis_source)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (review_hash, review, summary, overall_sentiment, confidence_score, source),
        )
        review_id = cursor.lastrowid

        for aspect, sentiment in aspects.items():
            cursor.execute(
                """
                INSERT IGNORE INTO absa_results (review_id, aspect, sentiment)
                VALUES (%s, %s, %s)
                """,
                (review_id, aspect, sentiment),
            )

        commit()
        return review_id

    except Exception as e:
        print(f"[ERROR] Failed to save review: {e}")
        get_connection().rollback()
        raise
    finally:
        cursor.close()


# ----------------------------------------
# Routes — core analysis
# ----------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    review = (data.get("review") or "").strip()
    if not review:
        return jsonify({"error": "No review text provided"}), 400
    if len(review) > MAX_REVIEW_LENGTH:
        return jsonify({"error": f"Review exceeds {MAX_REVIEW_LENGTH} characters"}), 400
    try:
        result = complete_pipeline(review)
        source = result.pop("_source")
        save_review_with_absa(
            review,
            result["Summary"],
            result["Overall Sentiment"],
            result["Confidence Score"],
            result["Aspect-based Sentiments"],
            source,
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/sentiment", methods=["POST"])
def sentiment_api():
    data = request.get_json()
    review = (data.get("review") or "").strip()
    if not review:
        return jsonify({"error": "No review text provided"}), 400
    if len(review) > MAX_REVIEW_LENGTH:
        return jsonify({"error": f"Review exceeds {MAX_REVIEW_LENGTH} characters"}), 400
    try:
        result = complete_pipeline(review)
        result.pop("_source", None)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ----------------------------------------
# Routes — batch
# ----------------------------------------
@app.route("/api/batch/analyze", methods=["POST"])
def batch_analyze():
    """
    Process up to MAX_BATCH_SIZE reviews in one call.

    Request body:
        { "reviews": ["text1", "text2", ...] }

    Response:
        {
            "total": 3,
            "succeeded": 2,
            "failed": 1,
            "results": [
                {"index": 0, "status": "success", "data": { ...pipeline output... }},
                {"index": 1, "status": "failed",  "error": "reason"},
                ...
            ]
        }
    """
    data = request.get_json()
    reviews = data.get("reviews") if data else None

    if not reviews or not isinstance(reviews, list):
        return jsonify({"error": "'reviews' must be a non-empty list"}), 400
    if len(reviews) > MAX_BATCH_SIZE:
        return jsonify({"error": f"Batch size exceeds the maximum of {MAX_BATCH_SIZE}"}), 400

    results = []
    succeeded = 0
    failed = 0

    for i, review in enumerate(reviews):
        if not isinstance(review, str) or not review.strip():
            results.append({"index": i, "status": "failed", "error": "Empty or non-string review"})
            failed += 1
            continue

        review = review.strip()

        if len(review) > MAX_REVIEW_LENGTH:
            results.append({
                "index": i,
                "status": "failed",
                "error": f"Exceeds {MAX_REVIEW_LENGTH} character limit",
            })
            failed += 1
            continue

        try:
            result = complete_pipeline(review)
            source = result.pop("_source")
            save_review_with_absa(
                review,
                result["Summary"],
                result["Overall Sentiment"],
                result["Confidence Score"],
                result["Aspect-based Sentiments"],
                source,
            )
            results.append({"index": i, "status": "success", "data": result})
            succeeded += 1
        except Exception as e:
            results.append({"index": i, "status": "failed", "error": str(e)})
            failed += 1

    return jsonify({"total": len(reviews), "succeeded": succeeded, "failed": failed, "results": results})


# ----------------------------------------
# Routes — analytics
# ----------------------------------------
@app.route("/api/stats", methods=["GET"])
def get_stats():
    """
    Overall system statistics.

    Response:
        {
            "total_reviews": 150,
            "sentiment_distribution": {"Positive": 90, "Negative": 40, "Neutral": 20},
            "total_aspects_analyzed": 620,
            "most_recent_analysis": "2025-01-15T10:30:00"
        }
    """
    cursor = get_cursor()
    try:
        cursor.execute("SELECT COUNT(*) AS total FROM reviews")
        total_reviews = cursor.fetchone()["total"]

        cursor.execute(
            "SELECT overall_sentiment, COUNT(*) AS cnt FROM reviews GROUP BY overall_sentiment"
        )
        distribution = {row["overall_sentiment"]: row["cnt"] for row in cursor.fetchall()}

        cursor.execute("SELECT COUNT(*) AS total FROM absa_results")
        total_aspects = cursor.fetchone()["total"]

        cursor.execute("SELECT MAX(created_at) AS last FROM reviews")
        last_row = cursor.fetchone()
        most_recent = last_row["last"].isoformat() if last_row["last"] else None

        return jsonify({
            "total_reviews": total_reviews,
            "sentiment_distribution": distribution,
            "total_aspects_analyzed": total_aspects,
            "most_recent_analysis": most_recent,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()


@app.route("/api/aspects/trending", methods=["GET"])
def get_trending_aspects():
    """
    Most-mentioned aspects across all reviews in the last N days.

    Query params:
        limit (int, 1-50, default 10)
        days  (int, 1-365, default 30)

    Response:
        {
            "days": 30,
            "trending_aspects": [
                {"aspect": "battery life", "total_mentions": 45,
                 "positive": 30, "negative": 12, "neutral": 3},
                ...
            ]
        }
    """
    limit = min(max(request.args.get("limit", 10, type=int), 1), 50)
    days = min(max(request.args.get("days", 30, type=int), 1), 365)

    cursor = get_cursor()
    try:
        cursor.execute(
            """
            SELECT
                a.aspect,
                COUNT(*)                                                      AS total_mentions,
                SUM(CASE WHEN a.sentiment = 'Positive' THEN 1 ELSE 0 END)   AS positive,
                SUM(CASE WHEN a.sentiment = 'Negative' THEN 1 ELSE 0 END)   AS negative,
                SUM(CASE WHEN a.sentiment = 'Neutral'  THEN 1 ELSE 0 END)   AS neutral
            FROM absa_results a
            JOIN reviews r ON a.review_id = r.id
            WHERE r.created_at >= NOW() - INTERVAL %s DAY
            GROUP BY a.aspect
            ORDER BY total_mentions DESC
            LIMIT %s
            """,
            (days, limit),
        )
        rows = cursor.fetchall()
        return jsonify({
            "days": days,
            "trending_aspects": [
                {
                    "aspect": row["aspect"],
                    "total_mentions": row["total_mentions"],
                    "positive": row["positive"],
                    "negative": row["negative"],
                    "neutral": row["neutral"],
                }
                for row in rows
            ],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()


@app.route("/api/reviews", methods=["GET"])
def get_reviews():
    """
    Paginated review history with optional sentiment filter.

    Query params:
        page      (int, default 1)
        limit     (int, 1-100, default 20)
        sentiment (Positive | Negative | Neutral)

    Response:
        {
            "reviews": [ {id, review_text, summarized_review, overall_sentiment,
                          confidence_score, created_at}, ... ],
            "pagination": {"page": 1, "limit": 20, "total": 150, "pages": 8}
        }
    """
    page = max(request.args.get("page", 1, type=int), 1)
    limit = min(max(request.args.get("limit", 20, type=int), 1), 100)
    sentiment = request.args.get("sentiment")
    if sentiment not in ("Positive", "Negative", "Neutral"):
        sentiment = None

    offset = (page - 1) * limit

    cursor = get_cursor()
    try:
        if sentiment:
            cursor.execute(
                "SELECT COUNT(*) AS total FROM reviews WHERE overall_sentiment = %s",
                (sentiment,),
            )
        else:
            cursor.execute("SELECT COUNT(*) AS total FROM reviews")
        total = cursor.fetchone()["total"]

        if sentiment:
            cursor.execute(
                """
                SELECT id, review_text, summarized_review, overall_sentiment,
                       confidence_score, created_at
                FROM reviews
                WHERE overall_sentiment = %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (sentiment, limit, offset),
            )
        else:
            cursor.execute(
                """
                SELECT id, review_text, summarized_review, overall_sentiment,
                       confidence_score, created_at
                FROM reviews
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )

        rows = cursor.fetchall()
        reviews = [
            {
                "id": row["id"],
                "review_text": row["review_text"],
                "summarized_review": row["summarized_review"],
                "overall_sentiment": row["overall_sentiment"],
                "confidence_score": (
                    float(row["confidence_score"]) if row["confidence_score"] is not None else None
                ),
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]

        return jsonify({
            "reviews": reviews,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total,
                "pages": max((total + limit - 1) // limit, 1),
            },
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()


# ----------------------------------------
# Run
# ----------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
