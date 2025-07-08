from fastapi import APIRouter, Depends
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
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/solve", tags=["Transfer Gichul QnAs"])

base_path = Path(os.getenv("BASE_PATH"))


def path_getter(directory: str):
    path_to_search = base_path / directory
    png_files = list(path_to_search.glob("*.png"))
    rel_paths = [str(p.relative_to(base_path)) for p in png_files]
    return rel_paths


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
    paths = path_getter(directory)
    try:
        gichulset = db.exec(
            select(GichulSet).where(
                GichulSet.grade == level,
                GichulSet.year == int(year),
                GichulSet.type == license,
                GichulSet.inning == round,
            )
        ).one()
        return {"qnas": gichulset.qnas, "imgURLs": paths}
    except Exception as e:
        return {"message": e}


@router.get("/img/{endpath}")
def get_one_image(endpath: str):
    path = base_path / endpath
    return FileResponse(path=path, media_type="image/png")
