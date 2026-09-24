"""
AI Candidate Search API
-----------------------
FastAPI server that receives free-text Hebrew job-seeking posts,
extracts structured candidate data using OpenAI (gpt-4o-mini),
generates a vector embedding (text-embedding-3-small),
and stores everything in a Supabase `candidates` table.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, Field, field_validator
from supabase import AsyncClient, acreate_client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()  # Loads a local .env file if present (optional in production)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("candidates-api")

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions
CANDIDATES_TABLE = "candidates"


def get_required_env(name: str) -> str:
    """Read a required environment variable or fail fast on startup."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class CandidatePostRequest(BaseModel):
    """Incoming request body."""

    raw_text: str = Field(
        ...,
        min_length=5,
        max_length=5000,
        description="Free-text job-seeking post in Hebrew",
        examples=["אהלן, בן 24 מרמת גן, עם קטנוע 125, מחפש עבודה בשליחויות בערבים, רוצה לפחות 45 שח שעה"],
    )

    @field_validator("raw_text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("raw_text cannot be empty")
        return value


class CandidateExtraction(BaseModel):
    """
    Structured data extracted by the LLM.
    All fields are required-but-nullable so the schema is compatible
    with OpenAI Structured Outputs (strict mode).
    """

    full_name: str | None = Field(
        description="The candidate's full name if explicitly mentioned, otherwise null"
    )
    city: str | None = Field(
        description="City of residence, in Hebrew, normalized (e.g. 'רמת גן'), otherwise null"
    )
    desired_roles: list[str] = Field(
        description="List of desired job roles in Hebrew (e.g. ['שליח']). Empty list if none"
    )
    licenses: list[str] = Field(
        description="Driving licenses / vehicles mentioned (e.g. ['A2 - קטנוע 125', 'B']). Empty list if none"
    )
    availability: str | None = Field(
        description="Availability / preferred shifts in Hebrew (e.g. 'ערבים'), otherwise null"
    )
    desired_hourly_rate: float | None = Field(
        description="Desired hourly wage in ILS as a number (e.g. 45), otherwise null"
    )


class CandidateResponse(CandidateExtraction):
    """Response returned to the client after a successful save."""

    id: Any | None = None
    raw_text: str


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """
You are an information-extraction engine for an Israeli job-matching platform.
You receive a free-text job-seeking post written in Hebrew (often informal, with slang and typos).

Extract the following fields:
- full_name: only if a name is explicitly written. Never invent a name.
- city: the city where the candidate lives, in standard Hebrew spelling.
- desired_roles: job roles the candidate is looking for, as short Hebrew role names
  (e.g. "מחפש עבודה בשליחויות" -> ["שליח"]).
- licenses: driving licenses or vehicles relevant to work. Infer the license class when obvious
  (e.g. "קטנוע 125" -> "A2", "רכב פרטי" -> "B"), keeping a short Hebrew description.
- availability: when the candidate can work (days, shifts, hours), in Hebrew.
- desired_hourly_rate: the requested hourly wage in ILS as a number only
  (e.g. "לפחות 45 שח שעה" -> 45). If a monthly salary is given instead, return null.

Rules:
- If a field is not mentioned, return null (or an empty list for list fields).
- Do not guess information that is not present in the text.
""".strip()


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create shared API clients once on startup."""
    app.state.openai = AsyncOpenAI(api_key=get_required_env("OPENAI_API_KEY"))
    app.state.supabase = await acreate_client(
        get_required_env("SUPABASE_URL"),
        get_required_env("SUPABASE_KEY"),
    )
    logger.info("OpenAI and Supabase clients initialized")
    yield
    await app.state.openai.close()
    logger.info("Shutting down")


app = FastAPI(
    title="AI Candidate Search API",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Service functions
# ---------------------------------------------------------------------------


async def extract_candidate_data(client: AsyncOpenAI, raw_text: str) -> CandidateExtraction:
    """Use gpt-4o-mini with Structured Outputs to extract candidate fields."""
    completion = await client.chat.completions.parse(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": raw_text},
        ],
        response_format=CandidateExtraction,
    )

    message = completion.choices[0].message
    if message.refusal:
        raise ValueError(f"Model refused to process the text: {message.refusal}")
    if message.parsed is None:
        raise ValueError("Model returned no structured data")

    return message.parsed


async def create_embedding(client: AsyncOpenAI, text: str) -> list[float]:
    """Generate a vector embedding for the raw text."""
    response = await client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


async def save_candidate(
    supabase: AsyncClient,
    raw_text: str,
    extracted: CandidateExtraction,
    embedding: list[float],
) -> dict[str, Any]:
    """Insert the candidate row into Supabase and return the stored record."""
    record = {
        "raw_text": raw_text,
        **extracted.model_dump(),
        "embedding": embedding,
    }
    result = await supabase.table(CANDIDATES_TABLE).insert(record).execute()
    if not result.data:
        raise RuntimeError("Supabase insert returned no data")
    return result.data[0]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/api/v1/candidates",
    response_model=CandidateResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["candidates"],
)
async def create_candidate(payload: CandidatePostRequest, request: Request) -> CandidateResponse:
    openai_client: AsyncOpenAI = request.app.state.openai
    supabase_client: AsyncClient = request.app.state.supabase

    # 1 + 2. Extraction and embedding are independent, so run them in parallel
    try:
        extracted, embedding = await asyncio.gather(
            extract_candidate_data(openai_client, payload.raw_text),
            create_embedding(openai_client, payload.raw_text),
        )
    except ValueError as exc:
        logger.warning("Extraction failed: %s", exc)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except OpenAIError as exc:
        logger.exception("OpenAI API error")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="AI service error") from exc

    # 3. Persist to Supabase
    try:
        saved = await save_candidate(supabase_client, payload.raw_text, extracted, embedding)
    except Exception as exc:
        logger.exception("Supabase insert failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error") from exc

    logger.info("Candidate saved (id=%s)", saved.get("id"))

    return CandidateResponse(
        id=saved.get("id"),
        raw_text=payload.raw_text,
        **extracted.model_dump(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
