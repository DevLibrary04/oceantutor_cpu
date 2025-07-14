from typing import Literal
from sqlmodel import Session, select
from ..models import (
    GichulSet,
    GichulQna,
    GichulSetType,
    GichulSetInning,
    GichulSetGrade,
    GichulSubject,
)
from pathlib import Path
import re
from ..utils import solve_utils
from dotenv import load_dotenv


def get_one_inning(
    year: Literal["2021", "2022", "2023"],
    license: GichulSetType,
    level: GichulSetGrade,
    round: GichulSetInning,
    db: Session,
) -> GichulSet:
    gichulset = db.exec(
        select(GichulSet).where(
            GichulSet.grade == level,
            GichulSet.year == int(year),
            GichulSet.type == license,
            GichulSet.inning == round,
        )
    ).one()
    return gichulset
