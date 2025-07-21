from typing import Annotated, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, SQLModel, Field, select
from ..schemas import UserSolvedQna, UserBase
from ..models import User, OdapSet, ExamType
from .user_crud import read_one_user


def create_one_odapset(examtype: str, user_id: int, db: Session):
    new_odapset = OdapSet(examtype=ExamType(examtype), user_id=user_id)
    db.add(new_odapset)
    db.commit()
    db.refresh(new_odapset)
    return new_odapset


def read_many_odapsets(user_id: int, db: Session):
    return db.exec(select(OdapSet).where(OdapSet.user_id == user_id)).all()
