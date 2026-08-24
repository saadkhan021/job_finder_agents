from __future__ import annotations

import time

from src.tools.linkedin_search import search_linkedin_jobs, fetch_job_snippet

# Indeed is dropped for now: it returns a Cloudflare bot-challenge page
# (HTTP 403) to plain HTTP requests, which no header tweaking gets past.
# The Indeed scraper code is still in src/tools/indeed_search.py if you
# want to revisit this later -- it's just not called here anymore.


def search_job_boards(
    queries: list[str],
    location: str = "",
    max_age_hours: int = 72,
    fetch_snippets: bool = True,
) -> list[dict]:
    """
    Deterministic (zero-LLM) job search across LinkedIn only, restricted
    to postings within `max_age_hours`.

    If `fetch_snippets` is True, each deduped job's own detail page is
    fetched once to pull a short plain-text description snippet -- this
    is pure code (no LLM tokens), and exists because matching on title
    alone misses genuinely relevant jobs with generic titles and can't
    reliably reject ambiguous ones either.
    """

    all_jobs: list[dict] = []
    seen_urls: set[str] = set()

    for query in queries:
        for job in search_linkedin_jobs(query, location, max_age_hours):
            if job["url"] and job["url"] not in seen_urls:
                seen_urls.add(job["url"])
                all_jobs.append(job)

    if fetch_snippets:
        print(f"  [LinkedIn] fetching description snippets for {len(all_jobs)} job(s)...")
        for job in all_jobs:
            job["snippet"] = fetch_job_snippet(job["url"])
            time.sleep(0.5)  # be polite between per-job page fetches

    return all_jobs