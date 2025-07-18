from typing import Annotated, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, SQLModel, Field
from ..schemas import UserSolvedQna, UserBase
from ..models import Something, User, OdapSet, ExamType
from .user_crud import read_one_user


def create_one_odap(smth: Something, db: Session):
    print(smth)
    print("something object reached till the crud function!")
    return smth
