from fastapi import APIRouter, Depends
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

router = APIRouter(prefix="/solve", tags=["Transfer Gichul QnAs"])


@router.get("/")
def get_one_inning(
    year: str,
    license: GichulSetType,
    level: GichulSetGrade,
    round: GichulSetInning,
    db: Session = Depends(get_db),
):
    try:
        gichulset = db.exec(
            select(GichulSet).where(
                GichulSet.grade == level,
                GichulSet.year == int(year),
                GichulSet.type == license,
                GichulSet.inning == round,
            )
        ).one()
        return gichulset.qnas
    except Exception as e:
        return {"message": e}
