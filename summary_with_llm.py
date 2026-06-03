import ollama

_SHORT_REVIEW_THRESHOLD = 20  # words — pass short reviews through unchanged

_SYSTEM = """\
You are a product review summarizer. Write a concise, factual summary in 2-3 sentences.

Your summary must cover:
1. The reviewer's overall impression (positive, negative, or mixed)
2. The 2-3 most important specific points mentioned (features, quality, value, etc.)
3. Any significant downside or caveat, if one exists

Rules:
- Use third-person, neutral language: "The reviewer finds...", "According to the review..."
- Do not invent details or opinions not present in the original review
- Do not include phrases like "In summary" or "Overall" as openers
- Output only the summary — no labels, no bullet points, no headers
- Maximum 60 words
"""

_USER_TEMPLATE = """\
Review: "{review_text}"
"""


def _call_llm(review_text: str) -> str:
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _USER_TEMPLATE.format(review_text=review_text.strip())},
        ],
        options={
            "temperature": 0.3,  # low but not zero — allows natural phrasing variety
            "num_predict": 150,  # ~60 words, generous cap to avoid cutoffs
        },
    )
    return response["message"]["content"].strip()


def generate_summary(text: str) -> str:
    """
    Summarise a product review using llama3.2 via Ollama.

    Short reviews are returned as-is — there is nothing to compress.
    Retries once on failure, then falls back to the T5-base pipeline
    so callers always receive a string, never an exception or error text.
    """
    text = (text or "").strip()
    if not text:
        return "No review text provided."

    if len(text.split()) < _SHORT_REVIEW_THRESHOLD:
        return text

    for attempt in range(2):
        try:
            result = _call_llm(text)
            if result:
                return result
        except Exception as e:
            print(f"[Summary LLM] attempt {attempt + 1} failed: {e}")

    # Both attempts failed — fall back to T5 so the DB never stores an
    # error string as a review's summary (which the original code did).
    print("[Summary LLM] falling back to T5 summarizer")
    try:
        from summary import generate_summary as t5_summary
        return t5_summary(text)
    except Exception as e:
        print(f"[Summary LLM] T5 fallback also failed: {e}")
        return "Summary not available."
