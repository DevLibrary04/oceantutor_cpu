from typing import Annotated, Optional, List
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, SQLModel, Field
from ..schemas import UserSolvedQna, UserBase, ManyOdaps
from ..models import Odap, User, OdapSet, ExamType
from .user_crud import read_one_user


def create_one_odap(new_odap: Odap, db: Session):
    db.add(new_odap)
    return new_odap


def create_many_odaps(odaplist: List[Odap], db: Session):
    db.add_all(odaplist)
    return odaplist
