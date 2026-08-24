from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings


load_dotenv()


class Settings(BaseSettings):

    # Gemini
    GOOGLE_API_KEY: str
    MODEL_NAME: str = "gemini-3.6-flash"
    TEMPERATURE: float = 0.7

    # Job search (LinkedIn + Indeed direct, no web search / no Tavily)
    MAX_JOB_AGE_HOURS: int = 72  # "most recent" window: last 3 days
    JOBS_PER_QUERY_PAGES: int = 1  # LinkedIn pagination depth per query
    DEFAULT_LOCATION: str = "Pakistan"  # whole country by default; overridden by the terminal prompt each run

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    SAMPLE_RESUME_DIR: Path = DATA_DIR / "sample_resumes"
    CV_FOLDER_DIR: Path = DATA_DIR / "cvs"

    # Resume formats
    SUPPORTED_EXTENSIONS: tuple[str, ...] = (
        ".pdf",
        ".docx",
    )

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()