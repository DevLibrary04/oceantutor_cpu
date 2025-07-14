from typing import Literal, List, Dict, Any
from sqlmodel import Session
from ..models import (
    GichulSet,
    GichulQna,
    GichulSetType,
    GichulSetInning,
    GichulSetGrade,
    GichulSubject,
)
from ..utils import solve_utils
from ..crud import solve_crud
from pathlib import Path
import re
from dotenv import load_dotenv


def retrieve_one_inning(
    year: Literal["2021", "2022", "2023"],
    license: GichulSetType,
    level: GichulSetGrade,
    round: GichulSetInning,
    db: Session,
) -> Dict[str, List[Dict[str, Any]]]:
    # 해당 회차 폴더 정보
    directory = solve_utils.dir_maker(year, license, level, round)
    # 해당 회차 폴더 속 이미지 파일들 경로 -> {"@pic땡땡": "경로정보"}
    path_dict = solve_utils.path_getter(directory)
    try:
        gichulset = solve_crud.get_one_inning(year, license, level, round, db)
    except:
        pass
    new_qnas_set = solve_utils.add_imgPaths_to_questions_if_any(gichulset, path_dict)
    return {"qnas": new_qnas_set}
