import enum
from datetime import datetime
from typing import List, Optional, ClassVar

from sqlmodel import (
    SQLModel,
    Field,
    Column,
    Relationship,
    TIMESTAMP,
    Text,
)
from sqlalchemy.sql import func


# Enum 정의


class GichulSetType(str, enum.Enum):
    giguansa = "기관사"
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
    unyoung = "운용"
    beopgyu = "법규"
    english = "영어"
    sangseon = "상선전문"
    eoseon = "어선전문"
    gigwan1 = "기관1"
    gigwan2 = "기관2"
    gigwan3 = "기관3"
    gigwan = "기관"
    jikmu = "직무일반"


# 테이블 정의


class User(SQLModel, table=True):
    """사용자 정보 테이블"""

    __tablename__: ClassVar[str] = "user"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=45, unique=True, index=True)
    password: str = Field(max_length=255)

    chats: List["Chat"] = Relationship(back_populates="user")
    odaps: List["Odap"] = Relationship(back_populates="user")


class GichulSet(SQLModel, table=True):
    """기출문제 세트 정보 테이블"""

    __tablename__: ClassVar[str] = "gichulset"

    id: Optional[int] = Field(default=None, primary_key=True)
    type: GichulSetType
    grade: GichulSetGrade
    year: int
    inning: GichulSetInning

    qnas: List["GichulQna"] = Relationship(back_populates="gichul_set")
    odaps: List["Odap"] = Relationship(back_populates="gichul_set")


class Chat(SQLModel, table=True):
    """채팅 세션 정보 테이블"""

    __tablename__: ClassVar[str] = "chat"

    id: Optional[int] = Field(default=None, primary_key=True)
    create_date: Optional[datetime] = Field(
        default=None, sa_column=Column(TIMESTAMP, server_default=func.now())
    )

    user_id: Optional[int] = Field(default=None, foreign_key="user.id")

    user: Optional[User] = Relationship(back_populates="chats")
    chat_turns: List["ChatTurn"] = Relationship(back_populates="chat")


class ChatTurn(SQLModel, table=True):
    """개별 대화 턴 (질문-답변) 정보 테이블"""

    __tablename__: ClassVar[str] = "chatturns"

    id: Optional[int] = Field(default=None, primary_key=True)
    turn_num: Optional[int] = Field(default=None)
    prompt: Optional[str] = Field(default=None, sa_column=Column(Text))
    response: Optional[str] = Field(default=None, sa_column=Column(Text))

    chat_id: Optional[int] = Field(default=None, foreign_key="chat.id")
    chat: Optional[Chat] = Relationship(back_populates="chat_turns")


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


class Odap(SQLModel, table=True):
    """사용자의 오답 정보 테이블"""

    __tablename__: ClassVar[str] = "odap"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_answer: Optional[str] = Field(default=None, max_length=45)
    created_date: Optional[datetime] = Field(
        default=None, sa_column=Column(TIMESTAMP, server_default=func.now())
    )

    user_id: Optional[int] = Field(default=None, foreign_key="user.id")
    gichulset_id: Optional[int] = Field(default=None, foreign_key="gichulset.id")

    user: Optional[User] = Relationship(back_populates="odaps")
    gichul_set: Optional[GichulSet] = Relationship(back_populates="odaps")
