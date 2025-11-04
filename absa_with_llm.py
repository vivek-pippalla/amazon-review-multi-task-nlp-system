import ollama
import re

def aspect_based_sentiment_llm(review_text):
 
    prompt = f"""
            You are an expert in product review analysis using Aspect-Based Sentiment Analysis (ABSA).

            Your goal is to extract specific product ASPECTS (distinct features or components) and classify each one's SENTIMENT.

            ### REVIEW
            "{review_text}"

            ---

            ### GUIDELINES
            1. Identify only concrete product aspects — these should be specific nouns or noun phrases that describe clear product
               features or components (e.g., “battery life”, “screen brightness”, “camera quality”).
            2. Exclude vague or generic terms such as “product”, “item”, “thing”, “it”, “this”, or any reference that doesn’t point
               to a distinct feature.
            3. For each aspect you identify:
                Determine its sentiment as Positive, Negative, or Neutral.
                Consider negations and comparisons carefully (e.g., “not great” → Negative, “better than before” → Positive).
                If sentiment is uncertain, mixed, or implied without clarity, mark it as Neutral.
            4. Present each aspect–sentiment pair on a separate line using the format:
                aspect: sentiment
            5. Do not include explanations, reasoning, or extra formatting — output only the aspect–sentiment pairs.
            6. Normalize and deduplicate aspect names:
                Remove leading articles like “the”, “a”, or “an”.
                Merge aspects that refer to the same concept (e.g., “laptop” and “the laptop” → “laptop”).
                Use singular, base form nouns (e.g., “batteries” → “battery”, “screens” → “screen”).
                Treat multi-word noun phrases as one aspect (e.g., extract “heart rate reading”, not both “heart rate reading” and “read”).


            ---

            ### EXAMPLES
            Input: "The battery lasts long, but the camera is poor."
            Output:
            battery: Positive
            camera: Negative

            Input: "Build quality feels cheap, but the display is excellent."
            Output:
            build quality: Negative
            display: Positive

            Input: "Price is okay, not too expensive."
            Output:
            price: Neutral

            ---

            ### FORMAT
            aspect: sentiment
            aspect: sentiment
            ...

            Now analyze the review below:

            BEGIN OUTPUT:
            """


    
    try:
        response = ollama.chat(
            model='llama3.2',
            messages=[{"role": "user", "content": prompt}]
        )
        lines = response['message']['content'].strip().splitlines()
        aspect_sentiment = {}

        for line in lines:
            match = re.match(r"(.+?):\s*(Positive|Negative|Neutral)", line, re.IGNORECASE)
            if match:
                aspect = match.group(1).strip().lower()
                sentiment = match.group(2).capitalize()
                aspect_sentiment[aspect] = sentiment

        return aspect_sentiment

    except Exception as e:
        print("ABSA LLM error:", e)
        return {}
