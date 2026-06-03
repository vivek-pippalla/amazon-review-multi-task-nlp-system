import re
import numpy as np
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en.stop_words import STOP_WORDS

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Renamed from `tokenizer` to avoid collision with the BiLSTM tokenizer in app.py
try:
    _bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    _bert_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    _bert_model.eval()
    _bert_model.to(torch.device("cpu"))
    has_transformer = True
except Exception:
    print("[ABSA] Transformer not available, falling back to rule-based scoring")
    has_transformer = False

# -----------------------------------------------------------------------
# Lexicons
# -----------------------------------------------------------------------

# "not" / "no" removed from here — they are negation words, not sentiment
# words. Keeping them caused "not bad" to score -2 instead of 0.
negative_indicators = {
    "hard", "difficult", "poor", "flaw", "problem", "issue", "bad", "wrong",
    "terrible", "awful", "disappointing", "disappointed", "subpar", "horrible",
    "unsatisfactory", "defective", "broken", "ugly", "overpriced", "waste",
    "wasteful", "lousy", "unreliable", "unresponsive", "inferior", "dislike",
    "hate", "annoying", "dismal", "crappy", "sucks", "abysmal", "pitiful",
    "drain", "drains", "lacking", "fails", "fail", "failing", "useless",
    "flimsy", "slow", "laggy", "blurry", "dim", "weak", "noisy", "scratchy",
    "uncomfortable", "inaccurate", "inconsistent", "stopped", "died", "dead",
    "damaged", "worthless", "fragile", "mediocre", "cheap",
}

positive_indicators = {
    "great", "good", "excellent", "best", "amazing", "wonderful", "fantastic",
    "awesome", "superb", "incredible", "impressive", "outstanding", "perfect",
    "brilliant", "exceptional", "phenomenal", "love", "loved", "delightful",
    "enjoyable", "marvelous", "spectacular", "fabulous", "stunning", "terrific",
    "splendid", "enjoy", "enjoys", "enjoying", "enjoyed", "recommend",
    "recommended", "worth", "satisfied", "happy", "pleased", "easy", "clear",
    "sharp", "crisp", "smooth", "comfortable", "sturdy", "durable", "fast",
    "quick", "accurate", "reliable", "responsive", "bright", "vivid", "vibrant",
    "solid", "stable", "convenient", "intuitive", "sleek", "elegant",
    "lightweight", "effective", "efficient", "seamless", "snappy",
}

neutral_indicators = {
    "okay", "ok", "decent", "average", "fair", "acceptable", "moderate",
    "regular", "standard", "usual", "typical", "normal", "ordinary",
    "so-so", "alright",
}

negation_words = {
    "not", "no", "never", "neither", "nor", "none",
    "isn't", "aren't", "wasn't", "weren't",
    "don't", "doesn't", "didn't",
    "cannot", "can't", "couldn't", "shouldn't", "wouldn't", "won't",
    "without", "lack", "lacks", "lacking",
}

intensity_modifiers = {
    "strengthen": ["very", "extremely", "incredibly", "remarkably", "exceptionally",
                   "absolutely", "really", "truly", "insanely", "super"],
    "weaken":     ["somewhat", "slightly", "a bit", "a little", "kind of",
                   "sort of", "rather", "fairly", "pretty"],
}

# -----------------------------------------------------------------------
# Aspect categories — expanded beyond smartphones to cover common Amazon
# product types (clothing, furniture, food, books, accessories, etc.)
# -----------------------------------------------------------------------
aspect_categories = {
    "display":      ["screen", "display", "resolution", "brightness", "clarity",
                     "color", "lcd", "oled", "panel", "touch"],
    "battery":      ["battery", "charge", "charging", "power", "life", "drain",
                     "draining", "endurance", "backup"],
    "camera":       ["camera", "picture", "photo", "video", "megapixel", "lens",
                     "zoom", "focus", "aperture", "selfie"],
    "performance":  ["performance", "speed", "processor", "cpu", "gpu", "lag",
                     "responsive", "loading", "snappy"],
    "design":       ["design", "look", "build", "construction", "material",
                     "aesthetic", "style", "finish", "texture"],
    "audio":        ["sound", "audio", "speaker", "volume", "bass", "treble",
                     "noise", "headphone", "earphone", "microphone", "mic"],
    "storage":      ["storage", "memory", "space", "capacity", "expandable"],
    "software":     ["software", "os", "android", "ios", "app", "application",
                     "update", "firmware", "interface", "ui"],
    "connectivity": ["wifi", "bluetooth", "signal", "reception", "network",
                     "connection", "nfc", "usb", "port", "wireless"],
    "price":        ["price", "cost", "value", "worth", "expensive", "affordable", "budget"],
    "packaging":    ["packaging", "box", "packing", "unboxing"],
    "shipping":     ["shipping", "delivery", "arrived", "arrival"],
    "durability":   ["durability", "durable", "lasting", "rugged", "robust", "sturdy"],
    "comfort":      ["comfort", "comfortable", "fit", "grip", "ergonomic",
                     "lightweight", "portable"],
}

aspect_qualifiers = {
    "camera":      ["low light", "night", "portrait", "landscape", "video", "selfie"],
    "display":     ["sunlight", "viewing angle", "brightness", "color accuracy"],
    "battery":     ["heavy use", "standby", "screen on", "gaming"],
    "performance": ["gaming", "multitasking", "background", "loading"],
    "audio":       ["bass", "treble", "vocal", "call quality"],
}

generic_aspects = {
    "product", "replacement", "order", "phone", "it", "this", "that", "one",
    "item", "unit", "thing", "object", "stuff", "package", "pack", "box",
    "container", "shipment", "service", "delivery", "device", "gadget", "model",
    "version", "specification", "specs", "general", "standard", "common",
    "quality", "support", "lot", "kind", "type", "sorts", "part", "bits",
    "piece", "condition",
}

excluded_words = {
    "day", "days", "week", "weeks", "month", "months", "year", "years", "time",
    "sometimes", "often", "always", "never", "today", "tomorrow", "yesterday",
    "then", "now", "later", "soon", "eventually", "finally", "initially",
    "aspect", "review", "opinion", "thought", "idea", "view", "fact", "thing",
    "case", "few", "hours", "hour", "minutes", "minute", "seconds", "second",
    "moment", "period", "duration", "span", "interval", "phase", "stage",
    "instance", "situation", "context", "environment", "setting", "background",
    "circumstance", "condition", "state", "place", "location", "area", "region",
    "zone", "space", "site", "venue", "locale", "point", "spot", "position",
}

_CONTRASTIVE_RE = re.compile(
    r"\b(but|however|although|though|yet|nevertheless|nonetheless)\b",
    re.IGNORECASE,
)

# -----------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------

def normalize_text(text):
    return re.sub(r"\s+", " ", text.lower().strip())


def normalize_aspect(aspect):
    aspect = aspect.lower().strip()
    for article in ("a ", "an ", "the "):
        if aspect.startswith(article):
            aspect = aspect[len(article):]
    return aspect


def is_generic(aspect):
    aspect_lower = aspect.lower()
    return (
        aspect_lower in generic_aspects
        or aspect_lower in excluded_words
        or any(ch.isdigit() for ch in aspect)
        or len(aspect) <= 2
        or all(w in STOP_WORDS for w in aspect_lower.split())
    )


def get_aspect_category(aspect):
    aspect_lower = aspect.lower()
    for category, terms in aspect_categories.items():
        if any(term in aspect_lower for term in terms):
            return category
    for category, qualifiers in aspect_qualifiers.items():
        if any(q in aspect_lower for q in qualifiers):
            return category
    return "general"


def is_qualifier_not_aspect(text, aspect):
    for category, qualifiers in aspect_qualifiers.items():
        if any(q in aspect for q in qualifiers):
            for term in aspect_categories[category]:
                if term in text and term not in aspect:
                    return True, category
    return False, None


# -----------------------------------------------------------------------
# Deduplication helpers
# -----------------------------------------------------------------------

def _deduplicate_by_substring(aspects):
    """
    When one aspect is a sub-phrase of another, keep the more specific one.
    E.g. ["battery", "battery life"] → ["battery life"]
    """
    by_length = sorted(aspects, key=len, reverse=True)
    kept = []
    for aspect in by_length:
        if not any(aspect in other and aspect != other for other in kept):
            kept.append(aspect)
    return kept


def get_related_aspects(aspects, threshold=0.7):
    """Group aspects by character n-gram similarity."""
    if len(aspects) < 2:
        return {a: [a] for a in aspects}

    try:
        vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 3))
        matrix = vectorizer.fit_transform(aspects)
        sim = cosine_similarity(matrix)
    except Exception:
        return {a: [a] for a in aspects}

    used = set()
    groups = {}
    for i, aspect in enumerate(aspects):
        if aspect in used:
            continue
        group = [aspect]
        used.add(aspect)
        for j, other in enumerate(aspects):
            if i != j and other not in used and sim[i, j] > threshold:
                group.append(other)
                used.add(other)
        groups[max(group, key=len)] = group
    return groups


# -----------------------------------------------------------------------
# Aspect extraction
# -----------------------------------------------------------------------

def extract_aspects_improved(text):
    doc = nlp(normalize_text(text))
    candidates = set()

    # Strategy 1: Noun chunks
    for chunk in doc.noun_chunks:
        if 1 <= len(chunk.text.split()) <= 4:
            candidate = normalize_aspect(chunk.text)
            if candidate and not is_generic(candidate):
                candidates.add(candidate)

    # Strategy 2: Dependency parsing — opinion verb → object noun
    # and subject/object nouns (without including the opinion adjective as
    # part of the aspect name; e.g. "terrible camera" → "camera" not "terrible camera")
    opinion_verbs = positive_indicators | negative_indicators
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_.lower() in opinion_verbs:
            for child in token.children:
                if child.pos_ == "NOUN" and not is_generic(child.text):
                    candidates.add(child.lemma_.lower())

        if token.pos_ == "NOUN" and token.dep_ in {"nsubj", "dobj", "pobj"}:
            if not is_generic(token.text):
                candidates.add(token.lemma_.lower())

    # Strategy 3: Compound nouns (e.g. "battery life", "charging speed")
    for token in doc:
        if token.pos_ == "NOUN" and not is_generic(token.text):
            compounds = [c.text for c in token.children if c.dep_ == "compound"]
            if compounds:
                phrase = " ".join([c.lower() for c in compounds] + [token.text.lower()])
                if not is_generic(phrase):
                    candidates.add(phrase)

    # Strategy 4: Regex patterns for common multi-word aspects
    for pattern in [
        r"\b(\w+ quality)\b",
        r"\b(\w+ life)\b",
        r"\b(\w+ performance)\b",
        r"\b(low light \w+)\b",
        r"\b(\w+ resolution)\b",
        r"\b(\w+ speed)\b",
        r"\b(\w+ sensor)\b",
    ]:
        for match in re.findall(pattern, text.lower()):
            if not is_generic(match):
                candidates.add(match)

    # Strategy 5: Qualifier-aware filtering — keep qualifiers only when they
    # provide unique context (e.g. "low light" when "camera" is already present)
    final = set()
    for aspect in candidates:
        is_qual, _ = is_qualifier_not_aspect(text, aspect)
        if not is_qual:
            final.add(aspect)
        elif any(q in aspect for q in ["low light", "fast charging", "high quality"]):
            final.add(aspect)

    # Lemmatise to reduce inflected duplicates
    lemmatised = set()
    for aspect in final:
        doc_a = nlp(aspect)
        lemmatised.add(" ".join(t.lemma_ for t in doc_a).lower())

    return list(lemmatised)


def filter_and_deduplicate_aspects(aspects, review_text, fuzzy_threshold=0.75):
    if not aspects:
        return []   # was returning {} — type inconsistency fixed

    filtered = [a for a in aspects if not is_generic(a)]
    if not filtered:
        return []

    # Remove aspects that are sub-phrases of more specific aspects first
    filtered = _deduplicate_by_substring(filtered)

    # Group remaining aspects by character similarity
    groups = get_related_aspects(filtered, threshold=fuzzy_threshold)

    # Build hierarchy: if A is contained in B, prefer B
    hierarchy = {}
    for main, related in groups.items():
        for a1 in related:
            for a2 in related:
                if a1 != a2 and a1 in a2:
                    hierarchy[a1] = a2

    unique = set()
    for main, related in groups.items():
        best = main
        for aspect in related:
            if len(aspect) > len(best):
                best = aspect
            if aspect in hierarchy:
                best = hierarchy[aspect]
        unique.add(best)

    final = set()
    for aspect in unique:
        is_qual, _ = is_qualifier_not_aspect(review_text, aspect)
        if not is_qual and len(aspect.split()) <= 4:
            final.add(aspect)

    return list(final)


# -----------------------------------------------------------------------
# Sentiment analysis
# -----------------------------------------------------------------------

def check_negation_context(text, target_word, window_size=5):
    # Strip punctuation so "great," matches "great"
    words = [re.sub(r"[^\w']", "", w) for w in text.lower().split()]
    target_clean = re.sub(r"[^\w']", "", target_word.lower())

    for idx, word in enumerate(words):
        if word != target_clean:
            continue
        window = words[max(0, idx - window_size):idx]
        if any(neg in window for neg in negation_words):
            return True
    return False


def analyze_clause_sentiment(clause, aspect):
    clause_lower = clause.lower()

    # Renamed local vars: pos_count / neg_count / neutral_count
    # The original code used the same names as the module-level sets,
    # which caused UnboundLocalError in Python 3 (name pre-classified as local).
    pos_count = sum(1 for w in positive_indicators if w in clause_lower)
    neg_count = sum(1 for w in negative_indicators if w in clause_lower)
    neutral_count = sum(1 for w in neutral_indicators if w in clause_lower)

    negated_pos = sum(
        1 for w in positive_indicators
        if w in clause_lower and check_negation_context(clause_lower, w)
    )
    negated_neg = sum(
        1 for w in negative_indicators
        if w in clause_lower and check_negation_context(clause_lower, w)
    )

    pos_count = pos_count - negated_pos + negated_neg
    neg_count = neg_count - negated_neg + negated_pos

    factor = 1.0
    if any(w in clause_lower for w in intensity_modifiers["strengthen"]):
        factor = 1.5
    elif any(w in clause_lower for w in intensity_modifiers["weaken"]):
        factor = 0.75

    pos_count *= factor
    neg_count *= factor

    if pos_count > neg_count:
        return min(0.5 + 0.1 * pos_count, 1.0)
    if neg_count > pos_count:
        return max(0.5 - 0.1 * neg_count, 0.0)
    if neutral_count > 0:
        return 0.5
    return 0.5


def analyze_sentiment_with_transformer(context_text, aspect):
    """
    Run DistilBERT SST-2 on the relevant clause.

    Removed the [aspect] bracket markers from the original code — SST-2 was
    never trained to interpret those brackets, so they added noise rather
    than focus.  Passing the clause directly gives cleaner results.
    """
    if not has_transformer:
        return None
    try:
        inputs = _bert_tokenizer(
            context_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(torch.device("cpu"))
        with torch.no_grad():
            logits = _bert_model(**inputs).logits
            score = torch.softmax(logits, dim=1)[0][1].item()
        return score
    except Exception as e:
        print(f"[ABSA] Transformer error: {e}")
        return None


def get_aspect_context(review, aspect):
    """
    Extract the clause(s) in which the aspect appears, isolating
    the relevant side of any contrastive structure.
    """
    doc = nlp(review)
    aspect_sentences = [
        sent.text for sent in doc.sents if aspect.lower() in sent.text.lower()
    ]
    if not aspect_sentences:
        return [review]

    relevant = []
    for sentence in aspect_sentences:
        # Split on contrastive conjunctions (capturing group keeps the markers)
        parts = _CONTRASTIVE_RE.split(sentence)
        aspect_parts = [
            p.strip() for p in parts
            if aspect.lower() in p.lower() and not _CONTRASTIVE_RE.match(p.strip())
        ]
        if aspect_parts:
            relevant.extend(aspect_parts)
        else:
            relevant.append(sentence)

    return relevant if relevant else [review]


def get_aspect_sentiment_improved(review, aspect):
    contexts = get_aspect_context(review, aspect)

    if has_transformer:
        scores = [
            s for s in (analyze_sentiment_with_transformer(c, aspect) for c in contexts)
            if s is not None
        ]
        if scores:
            avg = float(np.mean(scores))
            if avg > 0.6:
                return "Positive", avg
            if avg < 0.4:
                return "Negative", avg
            return "Neutral", avg

    scores = [analyze_clause_sentiment(c, aspect) for c in contexts]
    avg = float(np.mean(scores))
    if avg > 0.6:
        return "Positive", avg
    if avg < 0.4:
        return "Negative", avg
    return "Neutral", avg


# -----------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------

def aspect_based_sentiment_improved(review):
    if not review or not review.strip():
        return {}

    normalized = normalize_text(review)
    candidates = extract_aspects_improved(normalized)
    aspects = filter_and_deduplicate_aspects(candidates, normalized)

    result = {}
    for aspect in aspects:
        # Use the original (un-normalised) review for sentiment so the
        # transformer sees proper punctuation and capitalisation.
        sentiment, _ = get_aspect_sentiment_improved(review, aspect)
        result[aspect] = sentiment

    return result
