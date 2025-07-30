# from typing import Annotated, Literal, Optional
# from fastapi import APIRouter, Depends, status, HTTPException
# from fastapi.responses import FileResponse
# from sqlmodel import Session
# from app.dependencies import get_optional_current_activate_user
# from app.core.config import settings
# from app.database import get_db
# from app.schemas import SolveResponse
# from app.models import GichulSetType, GichulSetInning, GichulSetGrade, ExamType, User
# from app.services.solve import retrieve_one_inning

# router = APIRouter(prefix="/solve", tags=["Provide Gichul QnAs"])


# @router.get("/", response_model=SolveResponse)
# def get_one_inning(
#     examtype: Literal[ExamType.practice, ExamType.real],
#     year: Literal["2021", "2022", "2023"],
#     license: GichulSetType,
#     level: GichulSetGrade,
#     round: GichulSetInning,
#     db: Annotated[Session, Depends(get_db)],
#     current_user: Annotated[
#         Optional[User], Depends(get_optional_current_activate_user)
#     ],
# ):
#     return retrieve_one_inning(examtype, year, license, level, round, db, current_user)


# @router.get("/img/{endpath:path}", response_class=FileResponse)
# def get_one_image(endpath: str):
#     base_path = settings.BASE_PATH
#     path = (base_path / endpath).resolve()
#     if not str(path).startswith(str(base_path.resolve())):
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="접근이 허용되지 않은 경로입니다.",
#         )
#     if not path.is_file():
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail="이미지를 찾을 수 없습니다.",
#         )
#     return FileResponse(path=path, media_type="image/png")
