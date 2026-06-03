CREATE DATABASE IF NOT EXISTS nlp_app_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE nlp_app_db;

-- --------------------------------------------------------
-- reviews
--   review_hash : SHA-256 of the raw review text.
--                 Used for O(1) duplicate detection instead of
--                 the broken UNIQUE(review_text(255)) prefix index.
--   confidence_score : raw sigmoid output of the BiLSTM model
--                      (0.0 – 1.0). Lets us audit low-confidence
--                      predictions later.
--   analysis_source  : which inference path produced the result
--                      so we can compare LLM vs transformer quality.
-- --------------------------------------------------------
CREATE TABLE reviews (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    review_hash      CHAR(64)      NOT NULL,
    review_text      TEXT          NOT NULL,
    summarized_review TEXT,
    overall_sentiment ENUM('Positive', 'Neutral', 'Negative') NOT NULL,
    confidence_score  DECIMAL(5, 4),
    analysis_source   ENUM('llm', 'transformer') NOT NULL DEFAULT 'transformer',
    created_at        TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE  KEY uq_review_hash  (review_hash),
    INDEX   idx_sentiment       (overall_sentiment),
    INDEX   idx_created_at      (created_at)
);

-- --------------------------------------------------------
-- absa_results
--   UNIQUE KEY uq_review_aspect prevents duplicate aspect rows
--   even if the application layer has a bug.
--   INSERT IGNORE on the application side safely skips
--   re-insertion for an already-processed review.
-- --------------------------------------------------------
CREATE TABLE absa_results (
    id        INT AUTO_INCREMENT PRIMARY KEY,
    review_id INT          NOT NULL,
    aspect    VARCHAR(255) NOT NULL,
    sentiment ENUM('Positive', 'Neutral', 'Negative') NOT NULL,

    UNIQUE  KEY uq_review_aspect (review_id, aspect),
    INDEX   idx_aspect           (aspect),
    INDEX   idx_absa_sentiment   (sentiment),

    FOREIGN KEY (review_id) REFERENCES reviews(id) ON DELETE CASCADE
);
