from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select
from ..database import get_db
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

load_dotenv()

router = APIRouter(prefix="/modelcall", tags=["Call Local or External Models"])


gemApikey = os.getenv("GEMINI_APIKEY")

client = genai.Client(api_key=gemApikey)


async def geminiChat(user_prompt: str):
    modelResponse = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction="You are an assistant who helps the user get ready for the navigator('해기사') exam. Always answer in Korean. Keep the response below 700 characters long.",
            temperature=0.3,
        ),
        contents=user_prompt,
    )
    for chunk in modelResponse:
        yield f"data: {chunk.text}\n\n"


@router.get("/gemini")
async def modelcall_root(user_prompt: str):
    return StreamingResponse(geminiChat(user_prompt), media_type="text/event-stream")
