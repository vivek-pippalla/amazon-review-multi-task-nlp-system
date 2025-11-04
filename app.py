# ========================================
# Librariesssssssss
# ========================================

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
import pickle
import re
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # TensorFlow workaround
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences # type:ignore
from db_connection import get_cursor, commit , get_connection 
load_dotenv()
import traceback

# ----------------------------------------
# Flask app initialization
# ----------------------------------------
app = Flask(__name__)

# ----------------------------------------
# Import appropriate modules based on LLM
# ----------------------------------------

from llm_check import LLM_AVAILABLE, USE_DOCKER_MODEL_RUNNER, initialize_llm
initialize_llm()

try:
    if LLM_AVAILABLE:
        from absa_with_llm import aspect_based_sentiment_llm as aspect_based_sentiment
        from summary_with_llm import generate_summary
    else:
        from absa import aspect_based_sentiment_improved as aspect_based_sentiment
        from summary import generate_summary
except ImportError as e:
    raise RuntimeError(f"Required module missing: {e}")

# ----------------------------------------
# Load ML model and tokenizer
# ----------------------------------------

MODEL_PATH = "best_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False) # type: ignore

if not os.path.exists(TOKENIZER_PATH):
    raise RuntimeError(f"Tokenizer file not found: {TOKENIZER_PATH}")
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

MAX_LENGTH = 150

# ----------------------------------------
# Helper functions
# ----------------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def complete_pipeline(review: str, max_length=MAX_LENGTH) -> dict:
    cleaned = clean_text(review)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    
    pred = model.predict(padded, verbose=0)[0][0]
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
        "Summary": summary,
        "Aspect-based Sentiments": aspects
    }

# ----------------------------------------
# Save to DB
# ----------------------------------------
def save_review_with_absa(review, summary, overall_sentiment, aspects):
    cursor = get_cursor()
    try:
        # Step 1: Check if the review already exists
        cursor.execute("SELECT id FROM reviews WHERE review_text = %s", (review,))
        existing = cursor.fetchone()

        if existing:
            review_id = existing["id"]
            print(f"[INFO] Review already exists (id={review_id}). Linking ABSA results.")
        else:
            # Step 2: Insert new review
            cursor.execute("""
                INSERT INTO reviews (review_text, summarized_review, sentiment)
                VALUES (%s, %s, %s)
            """, (review, summary, overall_sentiment))
            review_id = cursor.lastrowid
            print(f"[INFO] Inserted new review (id={review_id}).")

        # Step 3: Insert ABSA aspects
        for aspect, sentiment in aspects.items():
            cursor.execute("""
                INSERT INTO absa_results (review_id, aspect, sentiment)
                VALUES (%s, %s, %s)
            """, (review_id, aspect, sentiment))

        commit()

    except Exception as e:
        print(f"[ERROR] Failed to save review: {e}")
        get_connection().rollback()
    finally:
        cursor.close()  # ensure cursor is closed every time

# ----------------------------------------
# Routes
# ----------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    review = data.get('review', '')
    if not review:
        return jsonify({"error": "No review text provided"}), 400
    try:
        result = complete_pipeline(review)
        save_review_with_absa(
            review,
            result["Summary"],
            result["Overall Sentiment"],
            result["Aspect-based Sentiments"]
        )
        return jsonify(result)
    except Exception as e:
        print("[ERROR] Exception during analysis:")
        traceback.print_exc()  # shows full stack trace in terminal
        return jsonify({"error": str(e)}), 500

@app.route('/api/sentiment', methods=['POST'])
def sentiment_api():
    data = request.get_json()
    review = data.get('review', '')
    if not review:
        return jsonify({"error": "No review text provided"}), 400
    try:
        result = complete_pipeline(review)
        return jsonify(result)
    except Exception as e:
        print("[ERROR] Exception during analysis:")
        traceback.print_exc()  # shows full stack trace in terminal
        return jsonify({"error": str(e)}), 500

# ----------------------------------------
# Run Flask
# ----------------------------------------

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port= 5000)
