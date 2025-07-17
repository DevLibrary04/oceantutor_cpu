from typing import Optional, List
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
    qnas: List[QnaWithImgPaths]


# auth


class CreateUser(UserBase):
    password: str


class CreateUserResponse(BaseModel):
    email: str
    name: str
    message: str = "User successfully registered!"


class Token(BaseModel):
    access_token: str
    token_type: str
