from typing import Optional, List, Literal
from pydantic import BaseModel, EmailStr, ConfigDict
from sqlmodel import SQLModel, Field
from .models import GichulQnaBase, UserBase


# main.py
class RootResponse(BaseModel):
    message: str
    endpoints: str


# solve
class QnaWithImgPaths(GichulQnaBase):
    imgPaths: Optional[List[str]] = None


class SolveResponse(BaseModel):
    odapset_id: Optional[int] = None
    qnas: List[QnaWithImgPaths]


# auth
class CreateUser(UserBase):
    password: str = Field(min_length=8)


class CreateUserResponse(BaseModel):
    email: str
    name: str
    message: str = "User successfully registered!"


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


# odap
class UserSolvedQna(BaseModel):
    choice: Literal["가", "나", "사", "아"]
    gichulqna_id: int
    odapset_id: int
