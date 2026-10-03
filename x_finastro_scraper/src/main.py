import os
import time
import re
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
USERNAME = os.getenv("X_USERNAME", "anandrathi12")

ASTRO_KEYWORDS = [
    "finastro", "financial astrology", "astrology", "astrological", "astro",
    "planet", "planetary", "mercury", "venus", "mars", "jupiter", "saturn",
    "rahu", "ketu", "moon", "full moon", "new moon", "eclipse",
    "retrograde", "nakshatra", "zodiac", "transit", "conjunction"
]

MARKET_KEYWORDS = [
    "nifty", "banknifty", "bank nifty", "sensex", "stock", "stocks",
    "market", "equity", "gold", "silver", "crude", "oil", "bitcoin",
    "btc", "crypto", "commodity", "bullish", "bearish", "rally",
    "correction", "support", "resistance", "breakout", "breakdown"
]

BULLISH_TERMS = [
    "bullish", "rally", "rise", "upside", "positive", "breakout",
    "higher", "buy", "strength", "strong"
]

BEARISH_TERMS = [
    "bearish", "fall", "correction", "downside", "negative", "breakdown",
    "lower", "sell", "weakness", "weak"
]

HEADERS = {"Authorization": f"Bearer {BEARER_TOKEN}"}


def require_token():
    if not BEARER_TOKEN:
        raise RuntimeError(
            "X_BEARER_TOKEN is missing. Set it as an environment variable or GitHub secret."
        )


def api_get(url, params=None, retries=4):
    for attempt in range(retries):
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)

        if r.status_code == 429:
            wait = min(60, 5 * (attempt + 1))
            print(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            continue

        r.raise_for_status()
        return r.json()

    raise RuntimeError("X API rate limit persisted after retries.")


def get_user_id(username):
    data = api_get(f"https://api.x.com/2/users/by/username/{username}")
    return data["data"]["id"]


def detect_keywords(text, keywords):
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lower]


def classify_direction(text):
    text_lower = text.lower()
    bull = sum(term in text_lower for term in BULLISH_TERMS)
    bear = sum(term in text_lower for term in BEARISH_TERMS)

    if bull > bear:
        return "Bullish"
    if bear > bull:
        return "Bearish"
    return "Neutral / unclear"


def extract_numbers(text):
    pattern = r"(?<!\w)(?:₹|\$)?\d{1,3}(?:,\d{2,3})*(?:\.\d+)?%?"
    return ", ".join(re.findall(pattern, text))


def get_posts(user_id):
    url = f"https://api.x.com/2/users/{user_id}/tweets"
    params = {
        "max_results": 100,
        "tweet.fields": "created_at,public_metrics,entities,lang",
        "exclude": "retweets",
    }

    all_posts = []
    next_token = None

    while True:
        if next_token:
            params["pagination_token"] = next_token
        else:
            params.pop("pagination_token", None)

        payload = api_get(url, params=params)
        batch = payload.get("data", [])
        all_posts.extend(batch)
        print(f"Downloaded {len(all_posts)} posts")

        next_token = payload.get("meta", {}).get("next_token")
        if not next_token:
            break

        time.sleep(0.25)

    return all_posts


def analyse_posts(posts):
    rows = []

    for post in posts:
        text = post.get("text", "")
        astro_terms = detect_keywords(text, ASTRO_KEYWORDS)
        market_terms = detect_keywords(text, MARKET_KEYWORDS)

        if not astro_terms:
            continue

        metrics = post.get("public_metrics", {})

        rows.append(
            {
                "Date": post.get("created_at"),
                "Tweet": text,
                "Astrology_Factor": ", ".join(astro_terms),
                "Market_Terms": ", ".join(market_terms),
                "Direction": classify_direction(text),
                "Numbers_or_Levels": extract_numbers(text),
                "Likes": metrics.get("like_count", 0),
                "Retweets": metrics.get("retweet_count", 0),
                "Replies": metrics.get("reply_count", 0),
                "Quotes": metrics.get("quote_count", 0),
                "Tweet_ID": post.get("id"),
                "URL": f"https://x.com/{USERNAME}/status/{post.get('id')}",
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        df = df.sort_values("Date", ascending=False)

    return df


def main():
    require_token()

    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    user_id = get_user_id(USERNAME)
    posts = get_posts(user_id)
    df = analyse_posts(posts)

    csv_path = out_dir / f"{USERNAME}_financial_astrology.csv"
    xlsx_path = out_dir / f"{USERNAME}_financial_astrology.xlsx"

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    print(f"\nFinancial-astrology posts found: {len(df)}")
    print(f"CSV:   {csv_path}")
    print(f"Excel: {xlsx_path}")


if __name__ == "__main__":
    main()
