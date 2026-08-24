from src.tools.resume_reader import read_resume, read_resumes_from_folder
from src.chains.analyzer import analyzer_chain
from src.tools.job_board_search import search_job_boards
from src.chains.writer import writer_chain
from src.config import settings


def _print_direct_links(jobs: list[dict]) -> None:
    """
    Print raw LinkedIn/Indeed links straight to the terminal, exactly as
    scraped -- no LLM involved, so these can never be hallucinated.
    """

    if not jobs:
        print("No direct job links found in this window.")
        return

    print(f"\n{'-' * 70}")
    print(f"DIRECT JOB LINKS ({len(jobs)} found)")
    print(f"{'-' * 70}")

    for job in jobs:
        print(f"[{job['source']}] {job['title']} — {job['company']}")
        print(f"  Posted: {job.get('posted', 'n/a')} | Location: {job.get('location', 'n/a')}")
        print(f"  {job['url']}")
        print()


def ask_company_preference() -> str:
    """
    Ask in the terminal whether to target small startups, multinational
    companies, or both. Returns one of: "startup", "multinational", "both".
    """

    print("\nDo you want to target:")
    print("  [1] Small startups")
    print("  [2] Multinational companies")
    print("  [3] Both / no preference")

    choice = input("Enter 1, 2, or 3 (default 3): ").strip()

    mapping = {"1": "startup", "2": "multinational", "3": "both"}

    return mapping.get(choice, "both")


def ask_location() -> str:
    """
    Ask in the terminal which city (or the whole country) to search.
    Not locked to any single city -- type any Pakistani city, or press
    Enter to search all of Pakistan.
    """

    location = input(
        f"\nWhich city to search? (default: {settings.DEFAULT_LOCATION}, "
        f"or type any city e.g. Lahore, Islamabad): "
    ).strip()

    return location or settings.DEFAULT_LOCATION


def run_pipeline(
    resume_path: str | None = None,
    cv_folder: str | None = None,
    company_preference: str = "both",
    location: str | None = None,
) -> str:
    """
    Run the Resume -> Job Matcher pipeline.

    Pass either `resume_path` (single file) or `cv_folder` (a folder of
    CV versions -- all get read and merged before analysis).

    `company_preference` is one of "startup", "multinational", "both" --
    it's a post-scrape filter based on a heuristic company-name match
    (see src/utils/company_classifier.py), since job boards don't expose
    a reliable employee-count field.

    This project is currently scoped to Karachi jobs only
    (see settings.DEFAULT_LOCATION) -- job roles still come from the
    candidate's CV(s), only the location is fixed.

    Flow:
        Resume(s) -> Analyzer (1 LLM call)
                  -> LinkedIn direct search, Karachi only (0 LLM calls)
                  -> company-type filter (0 LLM calls)
                  -> Writer / ranker (1 LLM call)
                  -> Final report + raw direct links printed to terminal
    """

    if not resume_path and not cv_folder:
        raise ValueError("Provide either resume_path or cv_folder.")

    if company_preference not in ("startup", "multinational", "both"):
        company_preference = "both"

    # -----------------------------------------------------
    # Stage 1: Resume Reader
    # -----------------------------------------------------

    print("\n[1/4] Reading resume(s)...")

    if cv_folder:
        resume_text = read_resumes_from_folder(cv_folder)
    else:
        resume_text = read_resume.invoke({"file_path": resume_path})

    if not resume_text.strip():
        raise ValueError("Resume reader returned empty text.")

    print("Resume(s) extracted successfully.")

    # -----------------------------------------------------
    # Stage 2: Resume Analyzer (LLM call 1 of 2)
    # -----------------------------------------------------

    print("\n[2/4] Analyzing resume(s) and extracting key points...")

    try:
        analyzer_result = analyzer_chain.invoke({"resume": resume_text})
    except Exception as e:
        error_text = str(e)
        if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
            print(
                "\nGemini free-tier daily quota is exhausted for the current model.\n"
                "Fix: open .env and change MODEL_NAME to a different current model\n"
                "(e.g. gemini-3.7-flash if you were on gemini-3.6-flash, or vice versa) --\n"
                "each model has its own separate daily quota. A new API key under the\n"
                "same Google account will NOT help; the quota is tied to your\n"
                "account/project, not the key."
            )
        else:
            print(f"\nResume analysis failed: {error_text}")
        raise

    candidate_profile = analyzer_result.get("candidate_profile", {})
    skills = analyzer_result.get("skills", [])
    technologies = analyzer_result.get("technologies", [])
    search_queries = analyzer_result.get("search_queries", [])

    # This project searches whatever location is passed in (any city, or
    # the whole country) -- roles/queries still come from the CV, only
    # location is chosen separately (via the terminal prompt or caller).
    location = location or settings.DEFAULT_LOCATION

    print("Resume analysis completed.")
    print(f"  Role: {candidate_profile.get('primary_role', 'n/a')}")
    print(f"  Skills: {', '.join(skills[:8])}{' ...' if len(skills) > 8 else ''}")
    print(f"  Location (fixed): {location}")
    print(f"  Company preference: {company_preference}")

    # -----------------------------------------------------
    # Stage 3: Direct LinkedIn + Indeed search (0 LLM calls)
    # -----------------------------------------------------

    print(
        f"\n[3/4] Searching LinkedIn directly in {location} "
        f"(last {settings.MAX_JOB_AGE_HOURS}h / ~3 days)..."
    )

    jobs = search_job_boards(
        queries=search_queries,
        location=location,
        max_age_hours=settings.MAX_JOB_AGE_HOURS,
    )

    print(f"Found {len(jobs)} jobs posted within the last {settings.MAX_JOB_AGE_HOURS}h.")

    _print_direct_links(jobs)

    if not jobs:
        return "No jobs found matching the candidate's profile in this window."

    # -----------------------------------------------------
    # Stage 4: Writer / ranker (LLM call 2 of 2)
    #
    # Company-type preference is passed through as a ranking signal, not
    # a pre-filter -- a hard pre-filter on company name previously threw
    # out relevant jobs just because their employer wasn't on a static
    # "known multinational" list, leaving the LLM with nothing useful.
    # -----------------------------------------------------

    print("[4/4] Generating final ranked report...")

    final_report = writer_chain.invoke(
        {
            "candidate_profile": candidate_profile,
            "skills": skills,
            "technologies": technologies,
            "jobs": jobs,
            "max_age_hours": settings.MAX_JOB_AGE_HOURS,
            "company_preference": company_preference,
        }
    )

    print("Final report generated.")

    return final_report


if __name__ == "__main__":

    # Point this at a folder containing one or more CV files
    # (PDF/DOCX) to have them all read and merged automatically.
    cv_folder = "data/cvs"

    preference = ask_company_preference()
    location = ask_location()

    report = run_pipeline(cv_folder=cv_folder, company_preference=preference, location=location)

    print("\n")
    print("=" * 70)
    print("FINAL JOB MATCHING REPORT")
    print("=" * 70)
    print("\n")

    print(report)