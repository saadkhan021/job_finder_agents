from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings


llm = ChatGoogleGenerativeAI(
    model=settings.MODEL_NAME,
    google_api_key=settings.GOOGLE_API_KEY,
    temperature=settings.TEMPERATURE,
)


prompt = ChatPromptTemplate.from_template(
    """
You are an expert technical recruiter and search optimization specialist.

Your job is NOT only to summarize the resume.

Your main objective is to analyze the candidate's resume(s) and generate
highly effective search queries that will maximize the quality and
relevance of job search results.

The generated search queries will be sent directly to LinkedIn's and
Indeed's own job search (not general web search) to find real, very
recent job openings, so each query should read like a short job-search
phrase a person would type into LinkedIn/Indeed's search box.

Resume:
{resume}

Analyze the resume and return ONLY valid JSON.

Use exactly this structure:

{{
    "candidate_profile": {{
        "primary_role": "",
        "experience_level": "",
        "years_of_experience": "",
        "location": "",
        "employment_type": ""
    }},

    "skills": [],

    "technologies": [],

    "search_queries": []
}}

Experience-level rules (IMPORTANT):

- Only ON-SITE / physical / in-person work experience counts toward
  seniority. Remote work, remote internships, and remote freelance
  gigs do NOT count as qualifying experience for "experience_level" --
  treat them as if they weren't there when judging seniority.
- If the resume has NO on-site/physical work experience (even if it
  lists remote internships, remote freelance, or personal/academic
  projects), set "experience_level" to "Fresher / Entry-Level" and
  generate search queries for Internship and Fresh Graduate / Entry-Level
  roles ONLY -- never Mid-Level or Senior queries in this case.
- Only use Mid-Level or Senior in "experience_level" and in search
  queries if the resume clearly shows real on-site/physical job
  experience of that level.
- Never generate "Remote jobs" search queries -- remote postings are
  out of scope for this project entirely.

Search query rules:

- Match the candidate's actual experience level as determined above.
- Identify the most suitable primary job role.
- Consider closely related job titles that match the candidate.
- Include important technologies and skills in the search queries.
- Include the determined experience level (Internship / Entry-Level /
  Fresher, or Mid-Level / Senior only if genuinely earned) in each query.
- Do NOT include "Remote" in any query.
- Include the candidate's location when it is available in the resume.
- Do NOT create queries using skills that are not present in the resume.
- Avoid overly generic queries such as "software engineer jobs".
- Prefer specific job-search phrases such as:
  "Python Backend Developer Internship"
  "Fresh Graduate FastAPI Developer jobs"
  "Junior Python Developer jobs"
- Generate 6-10 high-quality search queries.
- Each query should have a clear job-search intent.
- Queries should be suitable for web search and should help find actual job
  postings rather than tutorials, documentation, courses, or general articles.
- Prefer queries containing terms such as:
  jobs, careers, vacancies, hiring, openings, internship, position
  when appropriate.
- Do not include unnecessary explanations.
- Do not fabricate candidate information.
"""
)


parser = JsonOutputParser()

analyzer_chain = prompt | llm | parser