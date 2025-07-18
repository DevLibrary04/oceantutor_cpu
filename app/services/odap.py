from typing import Annotated, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, SQLModel, Field
from ..schemas import UserSolvedQna, UserBase
from ..models import Something, Odap
from ..crud.user_crud import read_one_user
from ..crud import odap_crud


def save_user_solved_qna(
    odap_data: UserSolvedQna, current_user: UserBase, db: Session
) -> Something:
    user = read_one_user(current_user.username, db)
    if user is None:
        raise HTTPException(status_code=400)
    smth = Something(
        choice=odap_data.choice, user_id=user.id, gichulqna_id=odap_data.gichulqna_id
    )
    return odap_crud.create_one_odap(smth, db)
