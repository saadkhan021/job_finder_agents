"""
Streamlit UI for the Multi-Agent Job Finder.

Design goal (mirrors the reference UI the user shared): ONE LLM call
reads the resume (Analyzer). Everything after that -- searching
LinkedIn, scraping snippets, date filtering, deduping, seniority
filtering, and match-score ranking -- is deterministic Python. No
per-job LLM calls, no writer LLM call. This keeps token usage fixed
and minimal regardless of how many jobs are found.
"""

from __future__ import annotations

import os
import re
import tempfile
import time

import streamlit as st

from src.tools.resume_reader import read_resume
from src.chains.analyzer import analyzer_chain
from src.tools.linkedin_search import search_linkedin_jobs, fetch_job_snippet
from src.utils.freshness import parse_relative_age_hours


# ---------------------------------------------------------
# Page setup + white/blue theme (component-level styling;
# base colors also set in .streamlit/config.toml)
# ---------------------------------------------------------

st.set_page_config(
    page_title="Job Finder — Resume to Ranked Jobs",
    page_icon="🎯",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.hero-eyebrow {
    color: #2563eb;
    letter-spacing: 2px;
    font-weight: 700;
    font-size: 0.8rem;
}
.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    color: #0f172a;
    margin-top: 4px;
}
.hero-title .accent { color: #2563eb; }
.hero-sub { color: #475569; margin-top: 8px; font-size: 0.95rem; }

.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0f172a;
    margin-top: 1.6rem;
    margin-bottom: 0.6rem;
}

.profile-card {
    background: #f8fafc;
    border: 1px solid #dbeafe;
    border-radius: 10px;
    padding: 14px 18px;
}
.profile-label { font-size: 0.78rem; color: #64748b; font-weight: 600; }
.profile-value { font-size: 1.25rem; font-weight: 700; color: #0f172a; }

.skill-pill {
    display: inline-block;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 999px;
    padding: 4px 12px;
    margin: 3px 4px 3px 0;
    font-size: 0.82rem;
    font-weight: 600;
}

.metric-card {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 14px 8px;
    text-align: center;
}
.metric-value { font-size: 1.55rem; font-weight: 800; color: #2563eb; }
.metric-label { font-size: 0.75rem; color: #475569; margin-top: 2px; }

.job-card {
    border: 1px solid #dbeafe;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 16px;
    background: #ffffff;
    box-shadow: 0 1px 3px rgba(37, 99, 235, 0.08);
}
.job-title { font-size: 1.08rem; font-weight: 700; color: #0f172a; }
.job-meta { color: #475569; font-size: 0.85rem; margin-top: 4px; }
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    margin-left: 6px;
    vertical-align: middle;
}
.badge-match-high { background: #dcfce7; color: #15803d; }
.badge-match-mid { background: #fef9c3; color: #a16207; }
.badge-match-low { background: #fee2e2; color: #b91c1c; }
.badge-type { background: #e0e7ff; color: #3730a3; }
.badge-source { background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }
.job-snippet { color: #334155; font-size: 0.87rem; margin-top: 10px; line-height: 1.4; }
.apply-link {
    display: inline-block;
    margin-top: 12px;
    color: #2563eb;
    font-weight: 700;
    text-decoration: none;
}
.pipeline-line { color: #15803d; font-size: 0.92rem; margin: 2px 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Deterministic helpers -- pure Python, zero LLM tokens
# ---------------------------------------------------------

SENIOR_KEYWORDS = ["senior", "lead", "principal", "staff", "manager", "head of", "director"]
JUNIOR_SIGNALS = ["fresher", "entry", "junior", "intern", "fresh grad", "associate"]

FRESHNESS_HOURS = {
    "Last 24 hours": 24,
    "Last 3 days": 72,
    "Last 7 days": 168,
}


def infer_employment_type(text: str) -> str:
    text = text.lower()
    if "intern" in text:
        return "Internship"
    if "part-time" in text or "part time" in text:
        return "Part-time"
    if "contract" in text:
        return "Contract"
    return "Full-time"


def compute_match_score(job: dict, skills: list[str], technologies: list[str]) -> int:
    """
    Deterministic, whole-word skill score (0-100).

    LinkedIn frequently blocks the detail-page request, so a job may have
    only its title available.  A single clear skill match in that title is
    useful evidence and must not be diluted by every other skill on the
    resume.  Short tokens such as ``C`` and ``R`` are ignored because they
    produce false substring matches in unrelated titles.
    """
    text = f"{job.get('title', '')} {job.get('snippet', '')}".lower()
    keywords = {
        k.strip().lower()
        for k in (skills + technologies)
        if k and len(k.strip()) >= 2
    }
    if not text or not keywords:
        return 0

    hits = sum(
        1
        for keyword in keywords
        if re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text)
    )
    return min(100, hits * 25)


def is_too_senior(title: str, experience_level: str) -> bool:
    if not any(signal in experience_level.lower() for signal in JUNIOR_SIGNALS):
        return False
    title_lower = title.lower()
    return any(word in title_lower for word in SENIOR_KEYWORDS)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Search Settings")
    results_per_query = st.slider("Search Results per Query", 5, 25, 15)
    max_jobs = st.slider("Maximum Jobs (final list)", 10, 50, 30)

    st.markdown("---")
    st.markdown("### 📅 Freshness / Posting Date")
    freshness_choice = st.radio(
        "Show jobs posted:",
        ["Last 24 hours", "Last 3 days", "Last 7 days", "Custom range", "Any time"],
        index=1,
    )

    if freshness_choice == "Custom range":
        max_age_hours = st.number_input("Custom window (hours)", min_value=1, value=72)
    elif freshness_choice == "Any time":
        max_age_hours = None
    else:
        max_age_hours = FRESHNESS_HOURS[freshness_choice]

    include_unknown_dates = st.checkbox("Include jobs with unknown/unverifiable dates")

    st.markdown("---")
    st.markdown("### 🌍 Job Search Location")
    search_location = st.text_input(
        "Search jobs in",
        value="Pakistan",
        placeholder="e.g. Pakistan, Lahore, Karachi, Islamabad",
        help="Type any city, or leave as 'Pakistan' to search the whole country -- not locked to one city.",
    )

    st.markdown("---")
    st.markdown("### 📍 Filters")
    location_contains = st.text_input("Location contains", placeholder="e.g. Karachi, Remote")
    employment_type_filter = st.selectbox(
        "Employment Type", ["Any", "Full-time", "Part-time", "Internship", "Contract"]
    )
    min_match = st.slider("Minimum Match %", 0, 100, 20)

    st.markdown("---")
    st.caption(
        "Minimum LLM usage: only the resume analysis stage calls Gemini. "
        "Search, scraping, date filtering, dedup, and ranking are pure Python."
    )


# ---------------------------------------------------------
# Hero
# ---------------------------------------------------------

st.markdown('<div class="hero-eyebrow">RESUME → RANKED JOBS</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-title">Skip the noise. <span class="accent">Get jobs that fit.</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-sub">One LLM call reads your resume. Everything after that — search, '
    "scraping, date filtering, deduping, and ranking — is deterministic Python.</div>",
    unsafe_allow_html=True,
)

st.write("")
uploaded_file = st.file_uploader("📄 Upload Resume (PDF / DOCX)", type=["pdf", "docx"])
run_clicked = st.button("🚀 Find Matching Jobs", use_container_width=True)


# ---------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------

if run_clicked:
    if not uploaded_file:
        st.warning("Please upload a resume first.")
    else:
        stage_labels = [
            "Resume uploaded & text extracted",
            "Candidate profile analyzed",
            "Search queries generated",
            "Searching jobs",
            "Scraping job pages",
            "Applying date filter",
            "Filtering, deduplicating & ranking results",
            "Final results ready",
        ]
        done_stages: list[str] = []
        progress_box = st.empty()

        def render_progress() -> None:
            lines = "\n".join(f'<div class="pipeline-line">✅ {s}</div>' for s in done_stages)
            progress_box.markdown(
                f'<div class="section-title">⚡ Pipeline Progress</div>{lines}',
                unsafe_allow_html=True,
            )

        # Stage 1: save + read resume
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        resume_text = read_resume.invoke({"file_path": tmp_path})
        done_stages.append(stage_labels[0])
        render_progress()

        # Stage 2: analyze -- the ONLY LLM call in the whole pipeline
        try:
            analysis = analyzer_chain.invoke({"resume": resume_text})
        except Exception as e:
            error_text = str(e)
            if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
                st.error(
                    "⚠️ Gemini free-tier daily quota is exhausted for the current model.\n\n"
                    "Fix: open your `.env` file and change `MODEL_NAME` to a different "
                    "current model (e.g. `gemini-3.7-flash` if you were on `gemini-3.6-flash`, "
                    "or vice versa) -- each model has its own separate daily quota. "
                    "A new API key under the same Google account will NOT help; the quota "
                    "is tied to your account/project, not the key."
                )
            else:
                st.error(f"Resume analysis failed: {error_text}")
            st.stop()

        candidate_profile = analysis.get("candidate_profile", {})
        skills = analysis.get("skills", [])
        technologies = analysis.get("technologies", [])
        search_queries = analysis.get("search_queries", [])
        done_stages.append(stage_labels[1])
        done_stages.append(stage_labels[2])
        render_progress()

        # Stage 3: search LinkedIn directly (pure requests, zero LLM tokens)
        # Location comes from the sidebar (any city, or whole country) --
        # NOT locked to any single city, and NOT taken from the resume.
        location = search_location.strip() or "Pakistan"
        server_side_hours = max_age_hours if max_age_hours else 24 * 30  # "Any time" -> wide window
        pages = max(1, -(-results_per_query // 10))  # ceil(results_per_query / 10)

        all_jobs: list[dict] = []
        seen_urls: set[str] = set()
        for query in search_queries:
            query_jobs = search_linkedin_jobs(
                query, location, server_side_hours, pages=pages, debug=False
            )[:results_per_query]  # Search Results per Query slider applied here
            for job in query_jobs:
                if job["url"] not in seen_urls:
                    seen_urls.add(job["url"])
                    all_jobs.append(job)
        done_stages.append(stage_labels[3])
        render_progress()

        # Stage 4: date filter -- pure Python, uses the "posted" text already
        # returned by search, no network calls, so this is nearly instant.
        found_count = len(all_jobs)
        no_verifiable_date = 0
        date_out_of_range = 0
        date_filtered_jobs: list[dict] = []

        for job in all_jobs:
            age = parse_relative_age_hours(job.get("posted", ""))
            if age is None:
                no_verifiable_date += 1
                if include_unknown_dates or max_age_hours is None:
                    date_filtered_jobs.append(job)
                continue
            if max_age_hours is not None and age > max_age_hours:
                date_out_of_range += 1
                continue
            date_filtered_jobs.append(job)
        done_stages.append(stage_labels[5])
        render_progress()

        # Stage 5: dedupe + seniority filter -- also pure Python, no network.
        # Doing this BEFORE snippet fetching means we only make a detail-page
        # request for jobs that survive these cheap checks, instead of every
        # single scraped job -- this is what keeps the wait time reasonable.
        experience_level = candidate_profile.get("experience_level", "")
        seen_key: set[tuple[str, str]] = set()
        duplicates = 0
        candidate_jobs: list[dict] = []

        for job in date_filtered_jobs:
            key = (job.get("title", "").lower(), job.get("company", "").lower())
            if key in seen_key:
                duplicates += 1
                continue
            seen_key.add(key)

            if is_too_senior(job.get("title", ""), experience_level):
                continue

            candidate_jobs.append(job)

        # Hard cap so total wait time stays bounded even with very broad
        # slider settings (lots of queries x high results-per-query).
        snippet_cap = max(max_jobs * 3, 60)
        candidate_jobs = candidate_jobs[:snippet_cap]

        # Stage 6: scrape each surviving job's own detail page for a snippet
        # (pure Python, zero LLM tokens). Live counter so it's clear this is
        # actually progressing rather than appearing stuck.
        snippet_progress = st.empty()
        for i, job in enumerate(candidate_jobs, 1):
            snippet_progress.caption(f"Fetching job descriptions: {i}/{len(candidate_jobs)}...")
            job["snippet"] = fetch_job_snippet(job["url"])
            time.sleep(0.3)
        snippet_progress.empty()
        done_stages.append(stage_labels[4])
        render_progress()

        # Stage 6: dedupe + seniority filter + score + rank
        experience_level = candidate_profile.get("experience_level", "")
        seen_key: set[tuple[str, str]] = set()
        duplicates = 0
        low_relevance = 0
        scored_jobs: list[dict] = []
        all_scored_jobs: list[dict] = []  # every job that passed dedupe/seniority, incl. low-score ones -- for debugging

        for job in date_filtered_jobs:
            key = (job.get("title", "").lower(), job.get("company", "").lower())
            if key in seen_key:
                duplicates += 1
                continue
            seen_key.add(key)

            if is_too_senior(job.get("title", ""), experience_level):
                continue

            score = compute_match_score(job, skills, technologies)
            job["match_score"] = score
            job["employment_type"] = infer_employment_type(
                f"{job.get('title', '')} {job.get('snippet', '')}"
            )

            all_scored_jobs.append(job)

            if score < min_match:
                low_relevance += 1
                continue

            scored_jobs.append(job)

        def ranking_key(job: dict) -> tuple[int, float]:
            """Prefer relevance, then newest verifiable posting."""
            age = parse_relative_age_hours(job.get("posted", ""))
            return (-job["match_score"], age if age is not None else float("inf"))

        all_scored_jobs.sort(key=ranking_key)
        relaxed_match_threshold = False
        if not scored_jobs and all_scored_jobs:
            # The jobs were returned for resume-derived LinkedIn queries.
            # Do not turn a temporary snippet-scraping failure into an empty
            # screen; show the best recent candidates and label that clearly.
            scored_jobs = all_scored_jobs[:max_jobs]
            relaxed_match_threshold = True
        else:
            scored_jobs.sort(key=ranking_key)
            scored_jobs = scored_jobs[:max_jobs]
        done_stages.append(stage_labels[6])
        done_stages.append(stage_labels[7])
        render_progress()

        st.session_state["result"] = {
            "candidate_profile": candidate_profile,
            "search_location": location,
            "skills": skills,
            "technologies": technologies,
            "search_queries": search_queries,
            "jobs": scored_jobs,
            "all_scored_jobs": all_scored_jobs,
            "found_count": found_count,
            "no_verifiable_date": no_verifiable_date,
            "date_out_of_range": date_out_of_range,
            "duplicates": duplicates,
            "low_relevance": low_relevance,
            "relaxed_match_threshold": relaxed_match_threshold,
        }

        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

result = st.session_state.get("result")

if result:
    profile = result["candidate_profile"]

    st.markdown('<div class="section-title">👤 Candidate Profile</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in (
        (c1, "Target Role", profile.get("primary_role", "N/A")),
        (c2, "Experience", profile.get("experience_level", "N/A")),
        (c3, "Location", result.get("search_location") or profile.get("location", "N/A")),
        (c4, "Employment Type", profile.get("employment_type", "Full-time")),
    ):
        col.markdown(
            f'<div class="profile-card"><div class="profile-label">{label}</div>'
            f'<div class="profile-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("**Extracted Skills:**")
    pills = "".join(
        f'<span class="skill-pill">{s}</span>' for s in result["skills"] + result["technologies"]
    )
    st.markdown(pills or "_None extracted_", unsafe_allow_html=True)

    with st.expander("🔍 Optimized search queries used"):
        for q in result["search_queries"]:
            st.write(f"- {q}")

    st.markdown('<div class="section-title">📊 Match Insights</div>', unsafe_allow_html=True)
    jobs = result["jobs"]
    companies = len({j.get("company", "") for j in jobs if j.get("company")})
    avg_match = round(sum(j["match_score"] for j in jobs) / len(jobs)) if jobs else 0

    metrics = [
        ("Final Jobs", len(jobs)),
        ("Companies", companies),
        ("Avg Match", f"{avg_match}%"),
        ("Duplicates", result["duplicates"]),
        ("No Verifiable Date", result["no_verifiable_date"]),
        ("Date Out of Range", result["date_out_of_range"]),
        ("Low Relevance", result["low_relevance"]),
    ]
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    # Client-side filters -- pure Python, no re-search, no LLM
    filtered_jobs = jobs
    if location_contains:
        filtered_jobs = [
            j for j in filtered_jobs
            if location_contains.lower() in j.get("location", "").lower()
        ]
    if employment_type_filter != "Any":
        filtered_jobs = [
            j for j in filtered_jobs if j.get("employment_type") == employment_type_filter
        ]

    st.markdown(
        f'<div class="section-title">🎯 Job Cards ({len(filtered_jobs)})</div>',
        unsafe_allow_html=True,
    )

    if not filtered_jobs:
        st.info("No jobs match the current filters. Try loosening Minimum Match % or Location.")
    elif result.get("relaxed_match_threshold"):
        st.warning(
            "No job met the selected match threshold, so these are the closest recent "
            "results from your resume-derived LinkedIn searches."
        )

    with st.expander("🔍 Debug: all scraped jobs & their scores (before Min Match % filter)"):
        st.caption(
            "This shows every job LinkedIn returned for your search, with its computed "
            "match score, so you can see whether low scores mean 'genuinely irrelevant "
            "jobs' or something else. No LLM is involved in this scoring -- it's plain "
            "keyword overlap between your skills and each job's title + description snippet."
        )
        debug_jobs = result.get("all_scored_jobs", [])
        if not debug_jobs:
            st.write("No jobs were scraped at all in this run.")
        else:
            for job in debug_jobs:
                snippet_preview = (job.get("snippet") or "(no snippet fetched)")[:150]
                st.markdown(
                    f"**{job.get('match_score')}%** — {job.get('title', 'Untitled')} "
                    f"— {job.get('company') or 'Unknown company'}  \n"
                    f"_{snippet_preview}_"
                )

    for i, job in enumerate(filtered_jobs, 1):
        score = job["match_score"]
        badge_class = (
            "badge-match-high" if score >= 60 else "badge-match-mid" if score >= 35 else "badge-match-low"
        )

        st.markdown(
            f"""
            <div class="job-card">
                <div class="job-title">{i}. {job.get('title', 'Untitled')}
                    <span class="badge {badge_class}">{score}% match</span>
                    <span class="badge badge-type">{job.get('employment_type', 'Full-time')}</span>
                </div>
                <div class="job-meta">🏢 {job.get('company') or 'Company not available'}
                    &nbsp;•&nbsp; 📍 {job.get('location') or 'Location not available'}</div>
                <div class="job-meta">🕒 {job.get('posted', 'n/a')}
                    <span class="badge badge-source">LinkedIn</span></div>
                <div class="job-snippet">{job.get('snippet') or 'No description preview available.'}</div>
                <a class="apply-link" href="{job.get('url', '#')}" target="_blank">🔗 Apply Now ↗</a>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("Upload a resume and click **Find Matching Jobs** to get started.")
