import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model=settings.MODEL_NAME,
    google_api_key=settings.GOOGLE_API_KEY,
    temperature=0.1,
)


# ---------------------------------------------------------
# Writer Prompt
#
# Jobs arrive here already scraped directly from LinkedIn/Indeed and
# already date-filtered in code (see src/tools/job_board_search.py).
# This step no longer extracts or invents fields -- it only ranks,
# dedupes, and formats a compact JSON list. That keeps the prompt size
# roughly constant regardless of how many jobs were found, instead of
# scaling with full scraped page text.
# ---------------------------------------------------------

writer_prompt = ChatPromptTemplate.from_template(
    """
You are an expert technical recruiter.

You are given a candidate profile and a JSON array of job postings
already scraped directly from LinkedIn (all posted within the last
{max_age_hours} hours). Every field below came straight from the job
board -- do not add, invent, or change any field.

Candidate Profile:
{candidate_profile}

Candidate Skills:
{skills}

Candidate Technologies:
{technologies}

Company-type preference: {company_preference}
(one of "startup", "multinational", or "both". This is a PREFERENCE for
ranking/ordering only -- there is no reliable employee-count data in the
JSON, so judge company size only from the company name itself if it's
obviously a well-known large/multinational brand. NEVER use this
preference to drop a job that is otherwise clearly relevant to the
candidate's skills -- relevance to the candidate's profile always comes
first. If "both", ignore this preference entirely.)

Jobs (JSON array, fields: source, title, company, location, posted, url,
snippet -- snippet is a short excerpt of the job's real description,
may be empty if it couldn't be fetched):
{jobs_json}

TASK:
1. Discipline match (GENERIC -- works for any field, not just tech):
   Keep a job if it genuinely belongs to the SAME discipline as
   candidate_profile.primary_role -- whatever that role is (AI
   Engineer, Data Scientist, Software Engineer, Mechanical Engineer,
   Civil Engineer, Accountant, anything). Judge this using BOTH the
   title AND the snippet (if the snippet is non-empty, it is strong
   evidence -- e.g. a generically-titled "Software Engineer" job whose
   snippet describes ML/AI work DOES count for an AI Engineer
   candidate). If the snippet is empty, judge from the title alone
   using the same standard as before: the title itself must plausibly
   belong to the candidate's field, and a shared generic tool alone
   (Python, SQL, AutoCAD, etc.) does not make two different disciplines
   the same one.
2. Seniority match (strict): Compare the title (and snippet, if it
   states a level) to candidate_profile.experience_level. If
   experience_level is "Fresher / Entry-Level" (or similar), REJECT any
   posting that reads as Senior/Lead/Principal/Staff/Manager/Head of --
   no exceptions, even if the discipline matches perfectly. Only
   Junior/Entry-Level/Associate/Internship/Fresh Graduate postings (or
   plain postings with no seniority signal) are acceptable in that case.
3. When genuinely unsure on either check AND the snippet is empty,
   EXCLUDE the job rather than force a justification. When the snippet
   gives clear evidence either way, trust the snippet over a guess from
   the title alone.
4. Among the jobs that pass both checks, if company-type preference is
   "startup" or "multinational", list jobs that clearly match that
   preference first, then the rest after -- do not omit the rest.
5. Sort within each group by freshness, most recently posted first.
6. Drop duplicates (same title + company).
7. For each kept job, write ONE short line citing a SPECIFIC overlapping
   skill/technology from the candidate's own list, or from the snippet
   if that's where the evidence came from -- no vague or forced
   connections.

OUTPUT FORMAT (use exactly this structure, one block per job):

# Job Matches (last {max_age_hours} hours)

## {{title}} — {{company}}

**Source:** {{source}}
**Location:** {{location}}
**Posted:** {{posted}}
**Match:** {{one short sentence}}
**Link:** {{url}}

After all jobs:

## Summary
- Total relevant jobs: X
- Matching company-type preference ({company_preference}): X

Do not add any other commentary. Never fabricate a URL, company, or
title that isn't in the JSON above. If zero jobs in the JSON are
relevant, say so plainly instead of inventing anything.
"""
)


def _to_jobs_json(jobs: list[dict]) -> str:
    """
    Strip to only the fields the writer needs. This is what keeps the
    prompt (and therefore token cost) small and roughly constant, no
    matter how many jobs were scraped.
    """

    compact = [
        {
            "source": j.get("source", ""),
            "title": j.get("title", ""),
            "company": j.get("company", ""),
            "location": j.get("location", ""),
            "posted": j.get("posted", ""),
            "url": j.get("url", ""),
            "snippet": j.get("snippet", ""),
        }
        for j in jobs
    ]
    return json.dumps(compact, ensure_ascii=False)


writer_chain = (
    {
        "candidate_profile": lambda x: x["candidate_profile"],
        "skills": lambda x: x["skills"],
        "technologies": lambda x: x["technologies"],
        "jobs_json": lambda x: _to_jobs_json(x["jobs"]),
        "max_age_hours": lambda x: x.get("max_age_hours", settings.MAX_JOB_AGE_HOURS),
        "company_preference": lambda x: x.get("company_preference", "both"),
    }
    | writer_prompt
    | llm
    | StrOutputParser()
)