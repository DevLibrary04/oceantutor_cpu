from typing import Literal, Dict, Any, List
from sqlmodel import Session
from ..models import (
    GichulSet,
    GichulQna,
    GichulSetType,
    GichulSetInning,
    GichulSetGrade,
    GichulSubject,
)
from pathlib import Path
import os, re
from dotenv import load_dotenv

load_dotenv()

base_path_str = os.getenv("BASE_PATH")
if base_path_str is None:
    raise ValueError("BASE_PATH not set")
base_path = Path(base_path_str)


def dir_maker(
    year: str, license: GichulSetType, level: GichulSetGrade, round: GichulSetInning
) -> str:
    esd = ""
    if license == GichulSetType.gigwansa:
        esd = "E"
    elif license == GichulSetType.hanghaesa:
        esd = "D"
    elif license == GichulSetType.sohyeong:
        esd = "S"

    grd = level.value
    if level == GichulSetGrade.grade_none:
        grd = GichulSetGrade.grade_1.value

    inn = round.value

    dir_path = f"{license.value}/{esd}{grd}_{year}_0{inn}"
    return dir_path


def path_getter(directory: str) -> Dict[str, str]:
    path_to_search = base_path / directory
    png_files = list(path_to_search.glob("*.png"))
    path_dict = {}
    for p in png_files:
        file_stem = p.stem
        try:
            lookup_key = file_stem.split("-")[-1]
        except IndexError:
            print("split didn't work; additional work needed")
            continue  # '-'로 분리되지 않으면 건너뛰기
        rel_path = p.relative_to(base_path).as_posix()
        path_dict[lookup_key] = rel_path
    return path_dict


def add_imgPaths_to_questions_if_any(
    gichulset: GichulSet, path_dict: Dict[str, str]
) -> List[Dict[str, Any]]:
    # 경로 정보 추가하기 위해 dict로
    qnas_as_dicts = [qna.model_dump() for qna in gichulset.qnas]
    pic_marker_reg = re.compile(r"@(\w+)")  # 문항속 @pic 찾기위한 regex

    for qna_dict in qnas_as_dicts:
        full_text = " ".join(
            qna_dict.get(key, " ")
            for key in ["questionstr", "ex1str", "ex2str", "ex3str", "ex4str"]
        )
        # 문항 속 문제, 보기 다 합치고 @pic 찾기 -> ['@pic땡땡', ...]
        found_pics = pic_marker_reg.findall(full_text)
        if found_pics:
            print(found_pics)
            img_paths = [
                path_dict[pic_name.lower()]
                for pic_name in found_pics
                if pic_name.lower() in path_dict
            ]
            qna_dict["imgPaths"] = img_paths  # 해당 문항 속 이미지 경로 정보 추가
    return qnas_as_dicts
