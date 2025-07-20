from typing import List
from sqlmodel import Session, select
from ..models import GichulSet, GichulSetType, GichulSetGrade


def read_qna_sets(
    license: GichulSetType,
    level: GichulSetGrade,
    db: Session,
):
    return db.exec(
        select(GichulSet).where(GichulSet.type == license, GichulSet.grade == level)
    ).all()
