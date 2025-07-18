from typing import Annotated, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, SQLModel, Field
from ..schemas import UserSolvedQna, UserBase
from ..models import Something, User, OdapSet, ExamType
from .user_crud import read_one_user


def create_one_odapset(examtype: str, user_id: int, db: Session):
    new_odapset = OdapSet(examtype=ExamType(examtype), user_id=user_id)
    db.add(new_odapset)
    db.commit()
    db.refresh(new_odapset)
    return new_odapset
