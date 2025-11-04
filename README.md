# Amazon Review Sentiment & ABSA Analysis System

---

## 🚀 Overview

This project is an advanced NLP pipeline for Amazon-style product review analysis. It performs:

- **Sentiment Classification:** Predicts the overall sentiment of reviews with high accuracy.
- **Aspect-Based Sentiment Analysis (ABSA):** Extracts product aspects and determines sentiment for each aspect.
- **Review Summarization:** Generates concise summaries of review content.

The system is modular, robust, and supports real-time analysis backed by both LLMs and efficient classic models.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **NLP Libraries:** Transformers, TensorFlow, Keras, Scikit-learn, SpaCy, Torch
- **Database:** MySQL
- **LLM Orchestration:** Ollama (serving LLaMA 3.2 and other models)
- **Deployment:** Docker (optional, for reproducible dev/prod), .env-based config

---

## ✨ Features

- **Hybrid Model Pipeline:**  
  Uses LLaMA 3.2 via Ollama for ABSA and summarization when available.  
  Falls back automatically to **DistilBERT**, **SpaCy**, and **T5** models if LLM is unavailable.

- **High-Accuracy Sentiment Classifier:**  
  Achieves **92% accuracy** on test data using a custom BiLSTM neural network.

- **Real-Time Processing:**  
  Fast, API-driven architecture suitable for production and demo environments.

- **Aspect Extraction:**  
  Extracts fine-grained aspects (e.g., battery, camera) and evaluates per-aspect sentiment with both neural and rule-based logic.

- **Summarization:**  
  Summarizes reviews into concise, readable summaries using either T5 or LLaMA 3.2.

- **Persistence & Analytics:**  
  All reviews, summaries, and aspect sentiments are stored in MySQL for analytics, dashboards, and reporting.

- **Modular & Configurable:**  
  Easily switch between LLMs and classic models by environment variable, Docker Compose, or automatic detection.

---

## 🏗️ Architecture

User/API Client
│
▼
Flask Web Server
│
├──> Sentiment Model (BiLSTM/TensorFlow)
├──> Aspect & Summary (LLaMA 3.2 via Ollama → T5/DistilBERT/SpaCy fallback)
└──> MySQL DB Storage (for results/analytics)


- **LLM Availability detection**: At runtime, checks if LLaMA 3.2 is available via Ollama; falls back instantly if not.
- **Docker optional**: Can be run locally or via Docker Compose. Modular service layout.

---

## ⚡ Quickstart (Local)

1. **Clone this repo and set up Python env**
2. Create `.env` (see `.env.example`)
3. Install requirements:
    ```
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```
4. Start MySQL (`localhost:3306`) and update `.env` credentials accordingly
5. Run the app:
    ```
    python app.py
    ```
6. (Optional) Start Ollama, pull and serve LLaMA 3.2 for LLM-based ABSA & summarization

---

## 🔌 LLM Usage

- If LLaMA 3.2 is detected via Ollama (Docker or local), ABSA/summarization uses it for best accuracy.
- If not, falls back to DistilBERT/T5/SpaCy pipeline—no manual intervention required.

---

## 🧪 API Example

curl -X POST http://localhost:5000/analyze
-H "Content-Type: application/json"
-d '{"review": "Excellent battery life but the camera is mediocre."}'

Response:

{
"Overall Sentiment": "Positive",
"Summary": "Excellent battery life, average camera.",
"Aspect-based Sentiments": {
"battery life": "Positive",
"camera": "Neutral"
}
}

---

## 📊 Analytics

- All processed reviews and aspect/sentiment results are stored in MySQL.
- Build dashboards and advanced analytics on top of this data for insights and reporting.

## 🙌 Credits

- Developed with HuggingFace Transformers, Ollama, TensorFlow, and the open-source NLP community.

---

## 🏷️ License

[MIT License](LICENSE) <!-- Or your preferred license -->
