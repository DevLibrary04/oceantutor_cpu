from typing import Literal
from sqlmodel import Session
from fastapi import HTTPException, status
from ..models import (
    GichulSetType,
    GichulSetInning,
    GichulSetGrade,
)
from ..utils import solve_utils
from ..crud import solve_crud
from ..schemas import SolveResponse, QnaWithImgPaths


def retrieve_one_inning(
    year: Literal["2021", "2022", "2023"],
    license: GichulSetType,
    level: GichulSetGrade,
    round: GichulSetInning,
    db: Session,
) -> SolveResponse:
    # 해당 회차 폴더 정보
    directory = solve_utils.dir_maker(year, license, level, round)
    # 해당 회차 폴더 속 이미지 파일들 경로 -> {"@pic땡땡": "경로정보"}
    path_dict = solve_utils.path_getter(directory)
    gichulset = solve_crud.get_one_inning(year, license, level, round, db)
    if gichulset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="검색 실패: 기출 문제 없음"
        )
    new_qnas_list = solve_utils.add_imgPaths_to_questions_if_any(gichulset, path_dict)
    pdt_validated_list = [
        QnaWithImgPaths.model_validate(qna_dict) for qna_dict in new_qnas_list
    ]
    return SolveResponse(qnas=pdt_validated_list)
