import re

ASTRO_KEYWORDS = [
    "finastro", "financial astrology", "financial astro", "astrology",
    "astrological", "astro", "planet", "planetary", "mercury", "venus",
    "mars", "jupiter", "saturn", "rahu", "ketu", "moon", "full moon",
    "new moon", "eclipse", "retrograde", "nakshatra", "zodiac", "transit",
    "conjunction", "aspect", "horoscope", "vedic", "jyotish", "tithi",
    "amavasya", "purnima"
]

MARKET_KEYWORDS = [
    "nifty", "banknifty", "bank nifty", "sensex", "stock", "stocks",
    "market", "equity", "gold", "silver", "crude", "oil", "bitcoin", "btc",
    "crypto", "commodity", "bullish", "bearish", "rally", "correction",
    "support", "resistance", "breakout", "breakdown", "index", "futures",
    "options", "dow", "nasdaq", "s&p"
]

BULLISH_TERMS = [
    "bullish", "rally", "rise", "upside", "positive", "breakout", "higher",
    "buy", "strength", "strong", "upmove", "gain"
]

BEARISH_TERMS = [
    "bearish", "fall", "correction", "downside", "negative", "breakdown",
    "lower", "sell", "weakness", "weak", "decline", "drop"
]


def _found(text: str, words: list[str]) -> list[str]:
    low = (text or "").lower()
    return sorted({w for w in words if w.lower() in low})


def classify(text: str) -> dict:
    astro = _found(text, ASTRO_KEYWORDS)
    market = _found(text, MARKET_KEYWORDS)
    low = (text or "").lower()
    bull = sum(term in low for term in BULLISH_TERMS)
    bear = sum(term in low for term in BEARISH_TERMS)

    if bull > bear:
        direction = "Bullish"
    elif bear > bull:
        direction = "Bearish"
    else:
        direction = "Neutral / unclear"

    numbers = re.findall(r"(?<!\w)(?:₹|\$)?\d{1,3}(?:,\d{2,3})*(?:\.\d+)?%?", text or "")

    return {
        "is_finastro": bool(astro),
        "astro_terms": ", ".join(astro),
        "market_terms": ", ".join(market),
        "direction": direction,
        "numbers_or_levels": ", ".join(numbers),
    }
