from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

from finastro import classify


@dataclass
class ScrapeConfig:
    username: str
    output_dir: Path
    profile_dir: Path
    max_stall_cycles: int = 12
    recovery_rounds: int = 3
    scroll_pause_seconds: float = 1.5


def _parse_metric(value: str | None) -> int | None:
    if not value:
        return None
    s = value.strip().replace(",", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*([KMB])?", s, re.I)
    if not m:
        return None
    n = float(m.group(1))
    suffix = (m.group(2) or "").upper()
    mult = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[suffix]
    return int(n * mult)


def _button_metric(article, testid: str) -> int | None:
    try:
        btn = article.locator(f'[data-testid="{testid}"]').first
        if btn.count() == 0:
            return None
        aria = btn.get_attribute("aria-label")
        if aria:
            return _parse_metric(aria)
        txt = btn.inner_text(timeout=1000)
        return _parse_metric(txt)
    except Exception:
        return None


def _extract_visible(article, username: str) -> dict | None:
    try:
        time_el = article.locator("time").first
        if time_el.count() == 0:
            return None

        dt = time_el.get_attribute("datetime")
        anchor = time_el.locator("xpath=..")
        href = anchor.get_attribute("href") or ""

        match = re.search(r"/status/(\d+)", href)
        if not match:
            html = article.inner_html(timeout=1500)
            match = re.search(r"/status/(\d+)", html)
        if not match:
            return None

        tweet_id = match.group(1)
        url = f"https://x.com/{username}/status/{tweet_id}"

        text_loc = article.locator('[data-testid="tweetText"]').first
        text = text_loc.inner_text(timeout=1500) if text_loc.count() else ""

        return {
            "tweet_id": tweet_id,
            "created_at": dt,
            "text": text,
            "reply_count": _button_metric(article, "reply"),
            "repost_count": _button_metric(article, "retweet"),
            "like_count": _button_metric(article, "like"),
            "bookmark_count": _button_metric(article, "bookmark"),
            "url": url,
        }
    except Exception:
        return None


def _write_outputs(rows: dict[str, dict], cfg: ScrapeConfig, final: bool = False) -> tuple[Path, Path, Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.DataFrame(rows.values())
    if not raw.empty:
        raw["created_at"] = pd.to_datetime(raw["created_at"], errors="coerce", utc=True)
        raw = raw.sort_values("created_at", ascending=False)

    enriched_rows = []
    for row in raw.to_dict("records"):
        tags = classify(row.get("text", ""))
        enriched_rows.append({**row, **tags})

    enriched = pd.DataFrame(enriched_rows)
    filtered = enriched[enriched["is_finastro"] == True].copy() if not enriched.empty else enriched.copy()

    raw_csv = cfg.output_dir / f"{cfg.username}_all_scraped.csv"
    filtered_csv = cfg.output_dir / f"{cfg.username}_financial_astrology.csv"
    excel_path = cfg.output_dir / f"{cfg.username}_financial_astrology.xlsx"

    enriched.to_csv(raw_csv, index=False)
    filtered.to_csv(filtered_csv, index=False)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        filtered.to_excel(writer, sheet_name="Financial_Astrology", index=False)
        enriched.to_excel(writer, sheet_name="All_Scraped", index=False)

    checkpoint = cfg.output_dir / f"{cfg.username}_checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "username": cfg.username,
                "tweet_count": len(enriched),
                "finastro_count": len(filtered),
                "final": final,
                "tweet_ids": list(rows.keys()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return filtered_csv, excel_path, raw_csv


def scrape_profile(
    cfg: ScrapeConfig,
    progress: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    log = progress or (lambda msg: None)
    stop_requested = should_stop or (lambda: False)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.profile_dir.mkdir(parents=True, exist_ok=True)

    rows: dict[str, dict] = {}

    # Resume from any prior raw CSV.
    raw_csv = cfg.output_dir / f"{cfg.username}_all_scraped.csv"
    if raw_csv.exists():
        try:
            old = pd.read_csv(raw_csv)
            for rec in old.to_dict("records"):
                tid = str(rec.get("tweet_id", "")).replace(".0", "")
                if tid:
                    rows[tid] = {
                        "tweet_id": tid,
                        "created_at": rec.get("created_at"),
                        "text": rec.get("text", ""),
                        "reply_count": rec.get("reply_count"),
                        "repost_count": rec.get("repost_count"),
                        "like_count": rec.get("like_count"),
                        "bookmark_count": rec.get("bookmark_count"),
                        "url": rec.get("url"),
                    }
            log(f"Resumed {len(rows)} previously scraped posts.")
        except Exception:
            pass

    with sync_playwright() as p:
        log("Opening X in a persistent browser session...")

        context = p.chromium.launch_persistent_context(
            user_data_dir=str(cfg.profile_dir),
            headless=False,
            viewport={"width": 1400, "height": 1000},
            args=["--disable-blink-features=AutomationControlled"],
        )

        page = context.pages[0] if context.pages else context.new_page()
        target = f"https://x.com/{cfg.username}"
        page.goto(target, wait_until="domcontentloaded", timeout=60_000)

        # If X asks for login, leave the browser open until the user completes it.
        login_wait_started = time.time()
        while True:
            if stop_requested():
                context.close()
                return {"cancelled": True}

            if page.locator('article[data-testid="tweet"]').count() > 0:
                break

            if "login" in page.url or "flow/login" in page.url:
                log("Please log in to X in the opened browser. Scraping will resume automatically.")

            if time.time() - login_wait_started > 600:
                context.close()
                raise RuntimeError("Login/profile timeline did not become available within 10 minutes.")

            try:
                page.wait_for_timeout(1500)
            except Exception:
                pass

        log(f"Timeline loaded for @{cfg.username}. Scraping until no more posts are available...")

        stalls = 0
        recoveries = 0
        last_height = 0
        last_count = len(rows)

        while True:
            if stop_requested():
                _write_outputs(rows, cfg, final=False)
                context.close()
                return {"cancelled": True, "scraped": len(rows)}

            articles = page.locator('article[data-testid="tweet"]')
            visible_count = articles.count()
            added = 0

            for i in range(visible_count):
                rec = _extract_visible(articles.nth(i), cfg.username)
                if rec and rec["tweet_id"] not in rows:
                    rows[rec["tweet_id"]] = rec
                    added += 1

            if added:
                stalls = 0
                recoveries = 0
                _write_outputs(rows, cfg, final=False)
                finastro_n = sum(1 for r in rows.values() if classify(r.get("text", ""))["is_finastro"])
                log(f"Scraped {len(rows)} unique posts | {finastro_n} financial-astrology matches")
            else:
                stalls += 1
                log(f"No new posts this pass ({stalls}/{cfg.max_stall_cycles}). Retrying...")

            try:
                current_height = page.evaluate("document.body.scrollHeight")
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(int(cfg.scroll_pause_seconds * 1000))
                new_height = page.evaluate("document.body.scrollHeight")
            except PlaywrightTimeoutError:
                current_height = last_height
                new_height = last_height

            at_bottom = False
            try:
                at_bottom = bool(
                    page.evaluate(
                        "() => (window.innerHeight + window.scrollY) >= (document.body.scrollHeight - 1200)"
                    )
                )
            except Exception:
                pass

            height_static = new_height == last_height == current_height
            count_static = len(rows) == last_count

            if stalls >= cfg.max_stall_cycles and height_static and count_static and at_bottom:
                if recoveries < cfg.recovery_rounds:
                    recoveries += 1
                    stalls = 0
                    log(f"Timeline stalled. Recovery {recoveries}/{cfg.recovery_rounds}: reloading at current position...")
                    try:
                        page.reload(wait_until="domcontentloaded", timeout=60_000)
                        page.wait_for_timeout(3000)
                        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    except Exception:
                        pass
                else:
                    log("No additional posts are being returned by X. Scrape is complete.")
                    break

            last_height = new_height
            last_count = len(rows)

        filtered_csv, excel_path, raw_csv = _write_outputs(rows, cfg, final=True)
        finastro_n = sum(1 for r in rows.values() if classify(r.get("text", ""))["is_finastro"])

        context.close()

        return {
            "cancelled": False,
            "scraped": len(rows),
            "finastro": finastro_n,
            "filtered_csv": str(filtered_csv),
            "excel": str(excel_path),
            "raw_csv": str(raw_csv),
        }
