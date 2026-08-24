from pathlib import Path
import re

from docx import Document
from langchain_core.tools import tool
from pypdf import PdfReader

from src.config import settings


def clean_text(text: str) -> str:
    """
    Clean extracted resume text.
    """

    text = text.replace("\r", "")
    text = text.replace("\t", " ")

    text = re.sub(r"[ ]{2,}", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf(path: Path) -> str:

    try:

        reader = PdfReader(path)

        pages = []

        for page in reader.pages:

            pages.append(page.extract_text() or "")

        return "\n".join(pages)

    except Exception as e:
        raise RuntimeError(f"Unable to read PDF : {e}")


def extract_docx(path: Path) -> str:

    try:

        document = Document(path)

        paragraphs = []

        for para in document.paragraphs:

            if para.text.strip():

                paragraphs.append(para.text)

        return "\n".join(paragraphs)

    except Exception as e:
        raise RuntimeError(f"Unable to read DOCX : {e}")


def _extract_one(path: Path) -> str:

    extension = path.suffix.lower()

    if extension not in settings.SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type : {extension}"
        )

    if extension == ".pdf":
        text = extract_pdf(path)
    else:
        text = extract_docx(path)

    return clean_text(text)


@tool
def read_resume(file_path: str) -> str:
    """
    Read a resume file (PDF/DOCX) and return extracted text.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"{file_path} not found.")

    return _extract_one(path)


def read_resumes_from_folder(folder_path: str) -> str:
    """
    Read every supported resume file (PDF/DOCX) in a folder and merge
    them into a single combined text block, so a candidate can keep
    several CV versions (e.g. resume_general.pdf, resume_backend.docx)
    and have all of them considered together.

    Each file's extracted text is clearly separated so the Analyzer can
    still tell which points came from which document if needed.
    """

    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"{folder_path} is not a valid folder.")

    resume_files = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in settings.SUPPORTED_EXTENSIONS
    )

    if not resume_files:
        raise ValueError(
            f"No supported resume files ({', '.join(settings.SUPPORTED_EXTENSIONS)}) "
            f"found in {folder_path}."
        )

    combined_sections = []

    for path in resume_files:
        try:
            text = _extract_one(path)
        except (RuntimeError, ValueError) as e:
            print(f"Skipping {path.name}: {e}")
            continue

        if text.strip():
            combined_sections.append(f"--- Resume: {path.name} ---\n{text}")

    if not combined_sections:
        raise ValueError("All resume files in the folder were empty or unreadable.")

    return "\n\n".join(combined_sections)