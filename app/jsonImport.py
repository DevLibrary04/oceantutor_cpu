import enum
from datetime import datetime
import os
from typing import List, Optional, ClassVar
from dotenv import load_dotenv
from sqlmodel import SQLModel, Field, Column, Relationship, TIMESTAMP, create_engine
from sqlalchemy.sql import func
import pandas as pd

load_dotenv()


class GichulSetType(str, enum.Enum):
    gigwansa = "기관사"
    hanghaesa = "항해사"
    sohyeong = "소형선박조종사"


class GichulSetGrade(str, enum.Enum):
    grade_1 = "1"
    grade_2 = "2"
    grade_3 = "3"
    grade_4 = "4"
    grade_5 = "5"
    grade_6 = "6"


class GichulSetInning(str, enum.Enum):
    inning_1 = "1"
    inning_2 = "2"
    inning_3 = "3"
    inning_4 = "4"


class GichulSubject(str, enum.Enum):
    hanghae = "항해"
    unyong = "운용"
    beopgyu = "법규"
    english = "영어"
    sangseon = "상선전문"
    eoseon = "어선전문"
    gigwan1 = "기관1"
    gigwan2 = "기관2"
    gigwan3 = "기관3"
    gigwan = "기관"
    jikmu = "직무일반"


class GichulSet(SQLModel, table=True):
    """기출문제 세트 정보 테이블"""

    __tablename__: ClassVar[str] = "gichulset"

    id: Optional[int] = Field(default=None, primary_key=True)
    type: GichulSetType
    grade: GichulSetGrade
    year: int
    inning: GichulSetInning

    qnas: List["GichulQna"] = Relationship(back_populates="gichul_set")


class GichulQna(SQLModel, table=True):
    """개별 기출문제 정보 테이블"""

    __tablename__: ClassVar[str] = "gichulqna"

    id: Optional[int] = Field(default=None, primary_key=True)
    subject: GichulSubject
    qnum: Optional[int] = Field(default=None)
    questionstr: Optional[str] = Field(default=None, max_length=450)
    ex1str: Optional[str] = Field(default=None, max_length=45)
    ex2str: Optional[str] = Field(default=None, max_length=45)
    ex3str: Optional[str] = Field(default=None, max_length=45)
    ex4str: Optional[str] = Field(default=None, max_length=45)
    answer: Optional[str] = Field(default=None, max_length=45)
    explanation: Optional[str] = Field(default=None, max_length=450)
    gichulset_id: Optional[int] = Field(default=None, foreign_key="gichulset.id")

    gichul_set: Optional[GichulSet] = Relationship(back_populates="qnas")
    odaps: List["Odap"] = Relationship(back_populates="gichul_qna")


class Odap(SQLModel, table=True):
    """사용자의 오답 정보 테이블"""

    __tablename__: ClassVar[str] = "odap"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_answer: str = Field(max_length=45)
    created_date: Optional[datetime] = Field(
        default=None, sa_column=Column(TIMESTAMP, server_default=func.now())
    )

    user_id: int = Field(foreign_key="user.id")
    gichulqna_id: int = Field(foreign_key="gichulqna.id")

    user: Optional[User] = Relationship(back_populates="odaps")
    gichul_qna: Optional[GichulQna] = Relationship(back_populates="odaps")


# SQLAlchemy 엔진

mysql_url = os.getenv("DATABASE_URL")
if mysql_url is not None:
    engine = create_engine(mysql_url, echo=True)


def run_engine():
    if engine is not None:
        return engine
        # SQLModel.metadata.create_all(engine)
    else:
        print("engine not created")

def main():
    engine = run_engine()


if __name__ == "__main__":
    main()