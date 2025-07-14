from typing import Annotated, Literal
from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import Session
from ..core.config import settings
from ..database import get_db
from ..schemas import SolveResponse
from ..models import (
    GichulSetType,
    GichulSetInning,
    GichulSetGrade,
)
from app.services import solve as solve_service

router = APIRouter(prefix="/solve", tags=["Transfer Gichul QnAs"])


@router.get("/", response_model=SolveResponse)
def get_one_inning(
    year: Literal["2021", "2022", "2023"],
    license: GichulSetType,
    level: GichulSetGrade,
    round: GichulSetInning,
    db: Annotated[Session, Depends(get_db)],
):
    return solve_service.retrieve_one_inning(year, license, level, round, db)


@router.get("/img/{endpath:path}", response_class=FileResponse)
def get_one_image(endpath: str):
    base_path = settings.BASE_PATH
    path = (base_path / endpath).resolve()
    if not str(path).startswith(str(base_path.resolve())):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="접근이 허용되지 않은 경로입니다.",
        )
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="이미지를 찾을 수 없습니다.",
        )
    return FileResponse(path=path, media_type="image/png")
