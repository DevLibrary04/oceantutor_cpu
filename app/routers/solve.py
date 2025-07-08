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

router = APIRouter(prefix="/solve", tags=["Transfer Gichul QnAs"])

def path_getter(directory: str):
    path_base = Path("C:/Users/user/Downloads/해기사기출DB(2021-2023)")
    path_to_search = path_base / directory
    png_files = list(path_to_search.glob("*.png"))
    return png_files

def dir_maker(year: str,
    license: GichulSetType,
    level: GichulSetGrade,
    round: GichulSetInning):


@router.get("/")
def get_one_inning(
    year: str,
    license: GichulSetType,
    level: GichulSetGrade,
    round: GichulSetInning,
    db: Session = Depends(get_db),
):
    directory = 
    paths = path_getter()
    try:
        gichulset = db.exec(
            select(GichulSet).where(
                GichulSet.grade == level,
                GichulSet.year == int(year),
                GichulSet.type == license,
                GichulSet.inning == round,
            )
        ).one()
        return {"qnas":gichulset.qnas, "imgPaths":paths}
    except Exception as e:
        return {"message": e}


@router.get("/img")
def get_one_image(path: Path):
    return FileResponse(path=path, media_type="image/png")

def main():
    path_getter("기관사/E1_2021_01")

if __name__ == "__main__":
    main()
