from __future__ import annotations

import requests
from bs4 import BeautifulSoup

from src.utils.freshness import parse_relative_age_hours

INDEED_SEARCH_URL = "https://www.indeed.com/jobs"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def search_indeed_jobs(
    query: str,
    location: str = "",
    max_age_hours: int = 72,
    debug: bool = True,
) -> list[dict]:
    """
    Search Indeed directly using its native `fromage` (days since posted)
    filter, then double-check each card's relative "posted" text
    client-side.
    """

    fromage_days = max(1, round(max_age_hours / 24))

    params = {
        "q": query,
        "l": location,
        "fromage": fromage_days,
    }

    try:
        response = requests.get(
            INDEED_SEARCH_URL,
            headers=HEADERS,
            params=params,
            timeout=10,
        )
    except requests.RequestException as e:
        if debug:
            print(f"  [Indeed] request error for '{query}': {e}")
        return []

    if debug:
        print(
            f"  [Indeed] query='{query}' status={response.status_code} "
            f"bytes={len(response.content)}"
        )

    if response.status_code != 200:
        if debug:
            print(f"  [Indeed] non-200 response, body preview: {response.text[:300]!r}")
        return []

    soup = BeautifulSoup(response.text, "lxml")
    cards = soup.select("div.job_seen_beacon") or soup.select("td.resultContent")

    if debug:
        print(f"  [Indeed] found {len(cards)} job card(s) on this page")

    jobs: list[dict] = []

    for card in cards:
        title_tag = card.select_one("h2.jobTitle span") or card.select_one("h2.jobTitle a")
        company_tag = card.select_one("span.companyName")
        location_tag = card.select_one("div.companyLocation")
        date_tag = card.select_one("span.date")
        link_tag = card.select_one("a")

        if not (title_tag and link_tag):
            continue

        posted_text = date_tag.get_text(strip=True) if date_tag else ""
        age_hours = parse_relative_age_hours(posted_text)

        if age_hours is not None and age_hours > max_age_hours:
            continue

        href = link_tag.get("href", "")
        if href.startswith("/"):
            href = f"https://www.indeed.com{href}"

        jobs.append(
            {
                "source": "Indeed",
                "title": title_tag.get_text(strip=True),
                "company": company_tag.get_text(strip=True) if company_tag else "",
                "location": location_tag.get_text(strip=True) if location_tag else "",
                "posted": posted_text or f"within last {fromage_days}d",
                "url": href,
            }
        )

    if debug:
        print(f"  [Indeed] kept {len(jobs)} job(s) within {max_age_hours}h")

    return jobs