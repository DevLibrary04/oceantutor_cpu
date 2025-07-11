from ..app.models import (
    User,
    GichulSet,
    GichulQna,
    GichulSetType,
    GichulSetInning,
    GichulSetGrade,
    GichulSubject,
)
from ..app.database import get_db
from fastapi import FastAPI, HTTPException, Depends
from sqlmodel import Session, select, SQLModel
from typing import Annotated

app = FastAPI()


class Message(SQLModel):
    detail: str


@app.get(
    "/",
    response_model=User,
    responses={
        404: {"model": Message, "description": "해당 사용자를 찾을 수 없습니다."},
        503: {
            "model": Message,
            "description": "데이터베이스 연결을 사용할 수 없습니다.",
        },
    },
)
def return_one_user(db: Annotated[Session, Depends(get_db)]):
    user = db.exec(select(User)).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


Message.model_rebuild()


@app.get("/solve/")
def return_one_inning(
    year: str,
    license: str,
    level: str,
    round: str,
    db: Annotated[Session, Depends(get_db)],
):
    try:
        set = db.exec(
            select(GichulSet).where(
                GichulSet.grade == "1",
                GichulSet.year == 2022,
                GichulSet.type == "기관사",
                GichulSet.inning == "2",
            )
        ).one()
        return set.qnas
    except Exception as e:
        print(f"여러줄 있나봐!!! {e}")
        return {"message": e}
