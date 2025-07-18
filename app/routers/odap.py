from typing import Annotated
from fastapi import APIRouter, Depends
from sqlmodel import Session
from ..schemas import UserSolvedQna, UserBase
from ..dependencies import get_current_active_user
from ..database import get_db
from ..services.odap import save_user_solved_qna
from ..models import Something, Odap


router = APIRouter(prefix="/odap", tags=["Save gichul qnas"])


@router.post("/save", response_model=Something)
async def save_one_qna(
    odap_data: UserSolvedQna,
    current_user: Annotated[UserBase, Depends(get_current_active_user)],
    db: Annotated[Session, Depends(get_db)],
):
    return save_user_solved_qna(odap_data, current_user, db)
