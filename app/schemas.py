from typing import Optional, List
from pydantic import BaseModel
from .models import GichulQnaBase


# main.py
class RootResponse(BaseModel):
    message: str
    endpoints: str


# solve
class QnaWithImgPaths(GichulQnaBase):
    imgPaths: Optional[List[str]] = None


class SolveResponse(BaseModel):
    qnas: List[QnaWithImgPaths]
