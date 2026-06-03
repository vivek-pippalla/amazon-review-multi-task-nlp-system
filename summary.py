from transformers import T5Tokenizer, T5ForConditionalGeneration

_tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
_model = T5ForConditionalGeneration.from_pretrained("t5-base")
_model.eval()

_SHORT_REVIEW_THRESHOLD = 20   # words — below this, summarising adds no value
_MAX_INPUT_TOKENS = 512


def generate_summary(text: str, max_length: int = 80, min_length: int = 20) -> str:
    """
    Abstractive summary of a product review using T5-base.

    Returns the original text for very short reviews because T5 produces
    worse output than the source when there is almost nothing to compress.
    Falls back to a safe string on any model error so callers always get
    a string, never an exception.
    """
    text = (text or "").strip()
    if not text:
        return "No review text provided."

    if len(text.split()) < _SHORT_REVIEW_THRESHOLD:
        return text

    try:
        inputs = _tokenizer(
            "summarize: " + text,
            return_tensors="pt",
            max_length=_MAX_INPUT_TOKENS,
            truncation=True,
        )

        ids = _model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,             # was 2; 4 is the standard minimum for quality
            early_stopping=True,
            no_repeat_ngram_size=3,  # prevents T5's tendency to repeat phrases
            length_penalty=2.0,      # penalises length → more concise output
        )

        return _tokenizer.decode(ids[0], skip_special_tokens=True)

    except Exception as e:
        print(f"[Summary] T5 inference error: {e}")
        return "Summary not available."
