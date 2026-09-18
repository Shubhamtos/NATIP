from __future__ import annotations

import io
import os
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from finastro import classify

st.set_page_config(
    page_title="X Financial Astrology Scraper",
    page_icon="📈",
    layout="wide",
)

DEFAULT_USERNAME = "anandrathi12"


def get_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name)


def detect_keywords_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    if "text" not in out.columns:
        text_col = next(
            (c for c in out.columns if c.lower() in {"tweet", "post", "content", "text"}),
            None,
        )
        if text_col:
            out = out.rename(columns={text_col: "text"})

    if "text" not in out.columns:
        raise ValueError("Dataset must contain a text/tweet/post/content column.")

    tags = out["text"].fillna("").map(classify).apply(pd.Series)
    for c in tags.columns:
        out[c] = tags[c]

    return out


def api_get(url: str, token: str, params: dict | None = None, retries: int = 5) -> dict:
    headers = {"Authorization": f"Bearer {token}"}

    for attempt in range(retries):
        response = requests.get(url, headers=headers, params=params, timeout=40)

        if response.status_code == 429:
            wait = min(60, 5 * (attempt + 1))
            time.sleep(wait)
            continue

        response.raise_for_status()
        return response.json()

    raise RuntimeError("X API rate limit persisted after retries.")


def fetch_x_timeline(username: str, token: str, progress_cb=None) -> pd.DataFrame:
    user_payload = api_get(
        f"https://api.x.com/2/users/by/username/{username}",
        token,
    )

    user_id = user_payload["data"]["id"]
    url = f"https://api.x.com/2/users/{user_id}/tweets"

    params = {
        "max_results": 100,
        "tweet.fields": "created_at,public_metrics,entities,lang",
        "exclude": "retweets",
    }

    rows = []
    next_token = None

    while True:
        if next_token:
            params["pagination_token"] = next_token
        else:
            params.pop("pagination_token", None)

        payload = api_get(url, token, params=params)
        batch = payload.get("data", [])

        for post in batch:
            metrics = post.get("public_metrics", {})
            rows.append(
                {
                    "tweet_id": post.get("id"),
                    "created_at": post.get("created_at"),
                    "text": post.get("text", ""),
                    "reply_count": metrics.get("reply_count", 0),
                    "repost_count": metrics.get("retweet_count", 0),
                    "like_count": metrics.get("like_count", 0),
                    "quote_count": metrics.get("quote_count", 0),
                    "url": f"https://x.com/{username}/status/{post.get('id')}",
                }
            )

        if progress_cb:
            progress_cb(len(rows))

        next_token = payload.get("meta", {}).get("next_token")
        if not next_token:
            break

        time.sleep(0.3)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        df = df.drop_duplicates("tweet_id").sort_values("created_at", ascending=False)

    return detect_keywords_dataframe(df)


def excel_bytes(all_df: pd.DataFrame, filtered_df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        filtered_df.to_excel(writer, sheet_name="Financial_Astrology", index=False)
        all_df.to_excel(writer, sheet_name="All_Scraped", index=False)
    bio.seek(0)
    return bio.read()


def normalize_upload(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)
    raise ValueError("Upload CSV or Excel only.")


def show_dashboard(df: pd.DataFrame, username: str):
    enriched = detect_keywords_dataframe(df)
    filtered = enriched[enriched["is_finastro"] == True].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Posts collected", f"{len(enriched):,}")
    c2.metric("Financial astrology", f"{len(filtered):,}")

    if len(enriched):
        pct = 100 * len(filtered) / len(enriched)
        c3.metric("Relevant share", f"{pct:.1f}%")
    else:
        c3.metric("Relevant share", "0%")

    if "direction" in filtered.columns and not filtered.empty:
        top_direction = filtered["direction"].value_counts().index[0]
    else:
        top_direction = "—"
    c4.metric("Most common direction", top_direction)

    st.subheader("Financial astrology posts")

    controls = st.columns([2, 2, 2])

    astro_options = sorted(
        {
            x.strip()
            for cell in filtered.get("astro_terms", pd.Series(dtype=str)).fillna("")
            for x in str(cell).split(",")
            if x.strip()
        }
    )
    selected_astro = controls[0].multiselect("Astrology factor", astro_options)

    market_options = sorted(
        {
            x.strip()
            for cell in filtered.get("market_terms", pd.Series(dtype=str)).fillna("")
            for x in str(cell).split(",")
            if x.strip()
        }
    )
    selected_market = controls[1].multiselect("Market / asset", market_options)

    directions = sorted(filtered["direction"].dropna().unique()) if "direction" in filtered.columns else []
    selected_direction = controls[2].multiselect("Direction", directions)

    view = filtered.copy()

    if selected_astro:
        view = view[
            view["astro_terms"].fillna("").apply(
                lambda x: any(term in str(x).split(", ") for term in selected_astro)
            )
        ]

    if selected_market:
        view = view[
            view["market_terms"].fillna("").apply(
                lambda x: any(term in str(x).split(", ") for term in selected_market)
            )
        ]

    if selected_direction:
        view = view[view["direction"].isin(selected_direction)]

    display_cols = [
        c for c in [
            "created_at",
            "text",
            "astro_terms",
            "market_terms",
            "direction",
            "numbers_or_levels",
            "like_count",
            "repost_count",
            "reply_count",
            "url",
        ] if c in view.columns
    ]

    st.dataframe(
        view[display_cols],
        use_container_width=True,
        hide_index=True,
        height=520,
        column_config={
            "url": st.column_config.LinkColumn("Post"),
            "text": st.column_config.TextColumn("Post text", width="large"),
        },
    )

    st.subheader("Downloads")
    d1, d2, d3 = st.columns(3)

    filtered_csv = filtered.to_csv(index=False).encode("utf-8")
    all_csv = enriched.to_csv(index=False).encode("utf-8")
    xlsx = excel_bytes(enriched, filtered)

    d1.download_button(
        "Download filtered CSV",
        filtered_csv,
        file_name=f"{username}_financial_astrology.csv",
        mime="text/csv",
        use_container_width=True,
    )

    d2.download_button(
        "Download all scraped CSV",
        all_csv,
        file_name=f"{username}_all_scraped.csv",
        mime="text/csv",
        use_container_width=True,
    )

    d3.download_button(
        "Download Excel workbook",
        xlsx,
        file_name=f"{username}_financial_astrology.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    with st.expander("See complete raw dataset"):
        st.dataframe(enriched, use_container_width=True, hide_index=True)


st.title("X Financial Astrology Scraper")
st.caption(
    "Collect and analyse financial-astrology posts from an X account. "
    "Default target: @anandrathi12"
)

with st.sidebar:
    st.header("Scraper settings")
    username = st.text_input("X username", DEFAULT_USERNAME).strip().lstrip("@")

    st.markdown("---")
    st.markdown("**Hosted scraping**")
    token_available = bool(get_secret("X_BEARER_TOKEN"))

    if token_available:
        st.success("X API secret detected")
    else:
        st.warning("X API secret is not configured")

    st.caption(
        "On Streamlit Cloud, the reliable hosted mode uses an X API bearer token stored "
        "as a Streamlit secret. The local desktop version continues to use Playwright."
    )

tab1, tab2, tab3 = st.tabs(
    ["Scrape account", "Upload existing scrape", "How it works"]
)

with tab1:
    st.subheader(f"Scrape @{username}")

    if token_available:
        st.info(
            "The scraper paginates until X returns no next page. "
            "There is no arbitrary post-count stop in this application."
        )

        if st.button("Start hosted scrape", type="primary", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()

            def update_progress(count):
                status.info(f"Collected {count:,} posts so far...")
                # Indeterminate timeline length, so pulse the progress bar.
                progress.progress(min(95, 5 + (count % 90)))

            try:
                token = get_secret("X_BEARER_TOKEN")
                result = fetch_x_timeline(username, token, update_progress)
                st.session_state["finastro_df"] = result
                st.session_state["finastro_username"] = username
                progress.progress(100)
                status.success(f"Scrape complete — {len(result):,} unique posts collected.")
            except Exception as exc:
                progress.empty()
                status.empty()
                st.error(f"Scrape failed: {exc}")

        if "finastro_df" in st.session_state:
            show_dashboard(
                st.session_state["finastro_df"],
                st.session_state.get("finastro_username", username),
            )
    else:
        st.warning(
            "Hosted scraping is not enabled yet because X_BEARER_TOKEN is missing."
        )
        st.markdown(
            """
            Add the secret in **Streamlit Cloud → App settings → Secrets**:

            ```toml
            X_BEARER_TOKEN = "your-token-here"
            ```

            After saving it, restart the app and the **Start hosted scrape** button will appear.
            """
        )

with tab2:
    st.subheader("Analyse an existing scraper file")
    uploaded = st.file_uploader(
        "Upload CSV or Excel produced by the desktop scraper",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded is not None:
        try:
            uploaded_df = normalize_upload(uploaded)
            show_dashboard(uploaded_df, username or DEFAULT_USERNAME)
        except Exception as exc:
            st.error(str(exc))

with tab3:
    st.markdown(
        """
        ### Hosted mode
        The Streamlit-hosted scraper uses the X API and requests pages repeatedly until X
        stops returning a pagination token. What historical posts are accessible depends on
        the X API access attached to your account.

        ### Desktop mode
        The existing Playwright desktop scraper remains in this repository. It opens a
        persistent local browser, scrolls the account timeline, checkpoints continuously,
        retries timeline stalls, and stops after X stops yielding additional posts.

        ### Financial-astrology filter
        Every collected post is checked for astrology terminology such as Saturn, Jupiter,
        Mercury, Rahu, Ketu, retrograde, transit, eclipse, nakshatra and #finastro, along
        with market terms such as NIFTY, Bank Nifty, gold, crude, Bitcoin, stocks and
        bullish/bearish language.

        The classification is a text-extraction aid, not a claim that astrology causes
        financial-market movements.
        """
    )
