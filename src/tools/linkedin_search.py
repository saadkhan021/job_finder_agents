from __future__ import annotations

import json
import re
import time

import requests
from bs4 import BeautifulSoup

from src.utils.freshness import parse_relative_age_hours

LINKEDIN_GUEST_SEARCH_URL = (
    "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.linkedin.com/jobs/search",
}


def _find_posted_text(card) -> str:
    """
    LinkedIn has used a few different markups for the "posted" timestamp
    over time (e.g. time.job-search-card__listdate vs
    time.job-search-card__listdate--new, or no class at all). Try the
    known selectors first, then fall back to any <time> tag, then to a
    regex scan of the card's own text for a relative-time phrase.
    """

    for selector in (
        "time.job-search-card__listdate",
        "time.job-search-card__listdate--new",
        "time",
    ):
        tag = card.select_one(selector)
        if tag:
            text = tag.get_text(strip=True)
            if text:
                return text

    match = re.search(
        r"(\d+\s*(?:minute|hour|day|week|month)s?\s*ago|just now|today)",
        card.get_text(" ", strip=True),
        re.IGNORECASE,
    )
    return match.group(1) if match else ""


def fetch_job_snippet(url: str, max_chars: int = 400, debug: bool = False) -> str:
    """
    Fetch a job's own detail page and pull a plain-text snippet of its
    actual description. Runs entirely in code (no LLM tokens).

    Tries structured data (schema.org JobPosting JSON-LD) FIRST -- this
    is what LinkedIn embeds in the raw, un-rendered HTML for search-engine
    indexing, so it's present even though `requests` doesn't execute
    JavaScript. The visible description div is often client-rendered by
    JS and therefore missing from a plain requests.get() response, which
    is why relying on it alone silently returned empty snippets for
    almost every job. The HTML class selectors are kept only as a
    fallback for pages where JSON-LD isn't present.
    """

    try:
        response = requests.get(url, headers=HEADERS, timeout=8)
        if response.status_code != 200:
            if debug:
                print(f"  [LinkedIn] snippet fetch non-200 ({response.status_code}) for {url}")
            return ""

        soup = BeautifulSoup(response.text, "lxml")

        # 1) Structured data (schema.org JobPosting) -- most reliable.
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
            except (json.JSONDecodeError, TypeError):
                continue

            candidates = data if isinstance(data, list) else [data]
            for item in candidates:
                if isinstance(item, dict) and item.get("@type") == "JobPosting":
                    desc_html = item.get("description", "")
                    if desc_html:
                        text = BeautifulSoup(desc_html, "lxml").get_text(" ", strip=True)
                        if text:
                            if debug:
                                print(f"  [LinkedIn] snippet via JSON-LD for {url}")
                            return text[:max_chars]

        # 2) Fallback: server-rendered HTML description containers, if any.
        desc_tag = (
            soup.select_one("div.show-more-less-html__markup")
            or soup.select_one("div.description__text")
        )
        if desc_tag:
            text = desc_tag.get_text(" ", strip=True)
            if text:
                if debug:
                    print(f"  [LinkedIn] snippet via HTML fallback for {url}")
                return text[:max_chars]

        if debug:
            print(f"  [LinkedIn] no snippet found (no JSON-LD, no HTML match) for {url}")
        return ""

    except requests.RequestException as e:
        if debug:
            print(f"  [LinkedIn] snippet fetch failed for {url}: {e}")
        return ""


def search_linkedin_jobs(
    query: str,
    location: str = "",
    max_age_hours: int = 24,
    pages: int = 2,
    debug: bool = True,
) -> list[dict]:
    """
    Search LinkedIn's public job feed directly for `query` / `location`,
    restricted server-side to `max_age_hours` via f_TPR, then
    double-checked client-side against each card's relative "posted" text.

    If a card's posted-time text can't be parsed at all, the job is kept
    rather than dropped -- the server-side f_TPR filter already did the
    real filtering, so an unparsed date just means "unknown label", not
    "old job".
    """

    jobs: list[dict] = []
    f_tpr_seconds = int(max_age_hours * 3600)

    for page in range(pages):
        params = {
            "keywords": query,
            "location": location,
            "f_TPR": f"r{f_tpr_seconds}",
            "start": page * 25,
        }

        try:
            response = requests.get(
                LINKEDIN_GUEST_SEARCH_URL,
                headers=HEADERS,
                params=params,
                timeout=10,
            )
        except requests.RequestException as e:
            if debug:
                print(f"  [LinkedIn] request error for '{query}': {e}")
            break

        if debug:
            print(
                f"  [LinkedIn] query='{query}' page={page} "
                f"status={response.status_code} bytes={len(response.content)}"
            )

        if response.status_code != 200:
            if debug:
                print(f"  [LinkedIn] non-200 response, body preview: {response.text[:300]!r}")
            break

        soup = BeautifulSoup(response.text, "lxml")
        cards = soup.select("li")

        if debug:
            print(f"  [LinkedIn] found {len(cards)} <li> card(s) on this page")

        if not cards:
            break

        page_jobs_kept = 0

        for card in cards:
            link_tag = card.select_one("a.base-card__full-link")
            title_tag = card.select_one("h3.base-search-card__title")
            company_tag = card.select_one("h4.base-search-card__subtitle")
            location_tag = card.select_one("span.job-search-card__location")

            if not (link_tag and title_tag):
                continue

            posted_text = _find_posted_text(card)
            age_hours = parse_relative_age_hours(posted_text)

            # Server-side f_TPR already restricted results to max_age_hours.
            # Only drop a card here if we found a parseable age AND it
            # exceeds the window -- an unparsed/unknown label is kept.
            if age_hours is not None and age_hours > max_age_hours:
                continue

            jobs.append(
                {
                    "source": "LinkedIn",
                    "title": title_tag.get_text(strip=True),
                    "company": company_tag.get_text(strip=True) if company_tag else "",
                    "location": location_tag.get_text(strip=True) if location_tag else "",
                    "posted": posted_text or f"within last {max_age_hours}h (label unavailable)",
                    "url": link_tag.get("href", "").split("?")[0],
                }
            )
            page_jobs_kept += 1

        if debug:
            print(f"  [LinkedIn] kept {page_jobs_kept} job(s) within {max_age_hours}h from this page")

        if page_jobs_kept == 0 and len(cards) > 0 and debug:
            sample = cards[0].get_text(" ", strip=True)[:200]
            print(f"  [LinkedIn] debug sample card text: {sample!r}")

        time.sleep(1)

    return jobs