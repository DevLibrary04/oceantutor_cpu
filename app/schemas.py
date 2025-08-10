from typing import Optional, List, Literal, Dict
from pydantic import BaseModel, EmailStr, ConfigDict

# main.py
class RootResponse(BaseModel):
    message: str
    endpoints: str

# RAG
class RAGRequest(BaseModel):
    question: str
    image_b64: Optional[str] = None

class RAGResponse(BaseModel):
    answer: str