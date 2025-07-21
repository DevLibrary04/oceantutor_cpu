from typing import Annotated
from fastapi import APIRouter, Depends
from sqlmodel import Session
from ..schemas import UserSolvedQna, ManyOdaps
from ..dependencies import get_current_active_user
from ..database import get_db
from ..services.odap import (
    save_user_solved_qna,
    save_user_solved_many_qnas,
    retrieve_user_saved_many_qnas,
)
from ..models import Odap, User


router = APIRouter(prefix="/odap", tags=["Save gichul qnas"])


@router.post("/save")
async def save_one_qna(
    odap_data: UserSolvedQna,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[Session, Depends(get_db)],
):
    return save_user_solved_qna(odap_data, current_user, db)


@router.post("/savemany")
async def save_many_qnas(
    odaps: ManyOdaps,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[Session, Depends(get_db)],
):
    return save_user_solved_many_qnas(odaps, current_user, db)


@router.get("/list")
async def get_many_odaps(
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[Session, Depends(get_db)],
):
    odaps = retrieve_user_saved_many_qnas(current_user, db)
    return [odap.odaps for odap in odaps]
