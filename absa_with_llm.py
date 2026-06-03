import re
import ollama

# -----------------------------------------------------------------------
# Prompt constants
# -----------------------------------------------------------------------

_SYSTEM = """\
You are an Aspect-Based Sentiment Analysis (ABSA) engine for product reviews.

WHAT COUNTS AS AN ASPECT:
- Concrete product features or components: "battery life", "camera quality",
  "build material", "screen brightness", "charging speed", "fingerprint sensor"
- User-experience dimensions: "setup process", "customer service", "packaging"
- Physical attributes: "weight", "size", "grip", "fit"

WHAT TO EXCLUDE:
- Vague pronouns and generic nouns: "product", "item", "thing", "it", "this",
  "one", "device", "unit", "gadget"
- Standalone opinion words used as subjects: "quality" alone, "overall"
- Time and quantity references: "days", "months", "times", "hours"
- The review itself or the reviewer: "review", "reviewer", "I", "me"

SENTIMENT RULES:
- Positive  → clearly good, satisfying, working well
- Negative  → clearly bad, failing, disappointing
- Neutral   → mixed signal, factual without clear opinion, "okay" / "decent"
- Negation  : "not great" → Negative | "not bad" → Positive | "not terrible" → Positive
- Implication: "stopped working after a week" → Negative for that aspect
- Comparative: "better than expected" → Positive | "worse than before" → Negative
- When genuinely uncertain or mixed → Neutral (not Positive or Negative)

NORMALISATION:
- Lowercase aspect names, singular form ("batteries" → "battery")
- Remove leading articles: "the display" → "display"
- Merge duplicates: "camera" and "the camera" → one entry "camera"

OUTPUT FORMAT — strictly one line per aspect, nothing else:
aspect name: Sentiment
"""

_USER_TEMPLATE = """\
EXAMPLES:

Review: "Battery lasts all day but the camera is blurry in low light."
battery life: Positive
camera: Negative

Review: "Build quality feels cheap. Display is absolutely stunning."
build quality: Negative
display: Positive

Review: "Charging speed is not great. Price is fair for what you get. Sound is decent."
charging speed: Negative
price: Neutral
sound: Neutral

Review: "Works perfectly right out of the box. Bluetooth keeps disconnecting randomly."
setup: Positive
bluetooth: Negative

Review: "Screen brightness could be better. Speakers are surprisingly loud and clear. \
Battery is not bad for a budget phone."
screen brightness: Neutral
speaker: Positive
battery: Positive

---

Review: "{review_text}"
"""

# -----------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------

# Tightened from the original (.+?): pattern, which matched any text before
# a colon, including echoed lines like "Review: Positive".
# This regex requires:
#   - aspect starts with a letter
#   - 2–59 characters total (rejects single-char garbage and runaway phrases)
#   - only letters, digits, spaces, hyphens, slashes allowed in aspect name
#   - sentiment word must end the line (no trailing text)
_LINE_RE = re.compile(
    r"^([a-zA-Z][a-zA-Z0-9 \-/]{1,58}):\s*(Positive|Negative|Neutral)\s*$",
    re.IGNORECASE,
)


def _parse_response(text: str) -> dict:
    result = {}
    for line in text.strip().splitlines():
        m = _LINE_RE.match(line.strip())
        if m:
            aspect = m.group(1).strip().lower()
            result[aspect] = m.group(2).capitalize()
    return result


# -----------------------------------------------------------------------
# LLM call
# -----------------------------------------------------------------------

def _call_llm(review_text: str) -> dict:
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _USER_TEMPLATE.format(review_text=review_text.strip())},
        ],
        options={
            "temperature": 0,    # deterministic output for a classification task
            "num_predict": 400,  # cap tokens — aspect lists are short
        },
    )
    return _parse_response(response["message"]["content"])


# -----------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------

def aspect_based_sentiment_llm(review_text: str) -> dict:
    """
    Extract aspect-sentiment pairs using llama3.2 via Ollama.
    Retries once on empty parse result, then falls back to the
    transformer-based pipeline so the caller always gets a response.
    """
    for attempt in range(2):
        try:
            result = _call_llm(review_text)
            if result:
                return result
            # Empty parse on attempt 0 — retry with the same prompt;
            # the model sometimes outputs a preamble on the first try.
        except Exception as e:
            print(f"[ABSA LLM] attempt {attempt + 1} failed: {e}")

    # Both attempts produced nothing — fall back to transformer ABSA
    # so the API never returns an empty aspect dict due to an LLM hiccup.
    print("[ABSA LLM] falling back to transformer ABSA")
    try:
        from absa import aspect_based_sentiment_improved
        return aspect_based_sentiment_improved(review_text)
    except Exception as e:
        print(f"[ABSA LLM] fallback also failed: {e}")
        return {}
