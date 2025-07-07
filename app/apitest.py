from .models import (
    User,
    GichulSet,
    GichulQna,
    GichulSetType,
    GichulSetInning,
    GichulSetGrade,
    GichulSubject,
)
from .database import run_engine
from fastapi import FastAPI, HTTPException
from sqlmodel import Session, select, SQLModel

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
def return_one_user():
    engine = run_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="not available")
    with Session(engine) as session:
        user = session.exec(select(User)).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user


Message.model_rebuild()


@app.get("/solve/")
def return_one_inning(year: str, license: str, level: str, round: str):
    engine = run_engine()
    if engine:
        with Session(engine) as session:
            try:
                set = session.exec(
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
