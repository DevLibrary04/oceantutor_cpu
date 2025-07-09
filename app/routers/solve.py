from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import Session, select
from ..database import get_db
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

router = APIRouter(prefix="/solve", tags=["Transfer Gichul QnAs"])

base_path_str = os.getenv("BASE_PATH")
if base_path_str is None:
    raise ValueError("BASE_PATH not set")
base_path = Path(base_path_str)


def path_getter(directory: str):
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


def dir_maker(
    year: str, license: GichulSetType, level: GichulSetGrade, round: GichulSetInning
):
    esd = ""
    if license == GichulSetType.gigwansa:
        esd = "E"
    elif license == GichulSetType.hanghaesa:
        esd = "D"
    elif license == GichulSetType.sohyeong:
        esd = "S"

    grd = level.value

    inn = round.value

    dir_path = f"{license.value}/{esd}{grd}_{year}_0{inn}"
    return dir_path


@router.get("/")
def get_one_inning(
    year: str,
    license: GichulSetType,
    level: GichulSetGrade,
    round: GichulSetInning,
    db: Session = Depends(get_db),
):
    directory = dir_maker(year, license, level, round)
    path_dict = path_getter(directory)
    try:
        gichulset = db.exec(
            select(GichulSet).where(
                GichulSet.grade == level,
                GichulSet.year == int(year),
                GichulSet.type == license,
                GichulSet.inning == round,
            )
        ).one()
        qnas_as_dicts = [qna.model_dump() for qna in gichulset.qnas]
        pic_marker_reg = re.compile(r"@(\w+)")

        for qna_dict in qnas_as_dicts:
            full_text = " ".join(
                qna_dict.get(key, " ")
                for key in ["questionstr", "ex1str", "ex2str", "ex3str", "ex4str"]
            )
            found_pics = pic_marker_reg.findall(full_text)
            if found_pics:
                img_paths = [
                    path_dict[pic_name]
                    for pic_name in found_pics
                    if pic_name in path_dict
                ]
                qna_dict["imgPaths"] = img_paths
        return {"qnas": qnas_as_dicts}
    except Exception as e:
        return {"message": e}


@router.get("/img/{endpath:path}")
def get_one_image(endpath: str):
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
