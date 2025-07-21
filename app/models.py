import enum
from datetime import datetime
from typing import List, Optional, ClassVar, Annotated
from sqlmodel import (
    SQLModel,
    Field,
    Column,
    Relationship,
    TIMESTAMP,
    Text,
)
from pydantic import EmailStr, ConfigDict
from sqlalchemy.sql import func
from sqlalchemy import Column, Enum as SQLAlchemyEnum, Text

# Enum 정의


class GichulSetType(str, enum.Enum):
    gigwansa = "기관사"
    hanghaesa = "항해사"
    sohyeong = "소형선박조종사"


class GichulSetGrade(str, enum.Enum):
    grade_none = "0"
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


class ExamType(str, enum.Enum):
    practice = "practice"
    real = "exam"
    cbt = "cbt"


class OdapChoice(str, enum.Enum):
    ex1 = "가"
    ex2 = "나"
    ex3 = "사"
    ex4 = "아"


# SQLModel 정의
class UserBase(SQLModel):
    username: EmailStr = Field(max_length=45, unique=True, index=True)
    indivname: str = Field(max_length=45)


class DBUser(UserBase):
    id: Optional[int] = Field(default=None, primary_key=True)
    hashed_password: str = Field(max_length=60)
    disabled: bool = Field(default=False)


class User(DBUser, table=True):
    """사용자 정보 테이블"""

    __tablename__: ClassVar[str] = "user"

    chats: List["Chat"] = Relationship(back_populates="user")
    odapsets: List["OdapSet"] = Relationship(back_populates="user")


class GichulSet(SQLModel, table=True):
    """기출문제 세트 정보 테이블"""

    __tablename__: ClassVar[str] = "gichulset"

    id: Optional[int] = Field(default=None, primary_key=True)
    type: GichulSetType = Field(
        sa_column=Column(
            SQLAlchemyEnum(
                GichulSetType, values_callable=lambda x: [e.value for e in x]
            )
        )
    )
    grade: GichulSetGrade = Field(
        sa_column=Column(
            SQLAlchemyEnum(
                GichulSetGrade, values_callable=lambda x: [e.value for e in x]
            )
        )
    )
    year: int
    inning: GichulSetInning = Field(
        sa_column=Column(
            SQLAlchemyEnum(
                GichulSetInning, values_callable=lambda x: [e.value for e in x]
            )
        )
    )

    qnas: List["GichulQna"] = Relationship(back_populates="gichulset")


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


class GichulQnaBase(SQLModel):
    id: Optional[int] = Field(default=None, primary_key=True)
    subject: GichulSubject = Field(
        sa_column=Column(
            SQLAlchemyEnum(
                GichulSubject, values_callable=lambda x: [e.value for e in x]
            )
        )
    )
    qnum: Optional[int] = Field(default=None)
    questionstr: Optional[str] = Field(default=None, sa_column=Column(Text))
    ex1str: Optional[str] = Field(default=None, sa_column=Column(Text))
    ex2str: Optional[str] = Field(default=None, sa_column=Column(Text))
    ex3str: Optional[str] = Field(default=None, sa_column=Column(Text))
    ex4str: Optional[str] = Field(default=None, sa_column=Column(Text))
    answer: Optional[str] = Field(default=None, max_length=45)
    explanation: Optional[str] = Field(default=None, max_length=450)
    gichulset_id: Optional[int] = Field(default=None, foreign_key="gichulset.id")


class GichulQna(GichulQnaBase, table=True):
    """개별 기출문제 정보 테이블"""

    __tablename__: ClassVar[str] = "gichulqna"

    gichulset: Optional[GichulSet] = Relationship(back_populates="qnas")
    odaps: List["Odap"] = Relationship(back_populates="gichul_qna")


class OdapSet(SQLModel, table=True):
    """오답을 생성한 페이지 정보"""

    __tablename__: ClassVar[str] = "odapset"

    id: Optional[int] = Field(default=None, primary_key=True)
    examtype: ExamType = Field(
        sa_column=Column(
            SQLAlchemyEnum(ExamType, values_callable=lambda x: [e.value for e in x])
        )
    )
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")
    created_date: Optional[datetime] = Field(
        default=None, sa_column=Column(TIMESTAMP, server_default=func.now())
    )

    user: Optional[User] = Relationship(back_populates="odapsets")
    odaps: List["Odap"] = Relationship(back_populates="odapset")


class Odap(SQLModel, table=True):
    """사용자의 오답 정보 테이블"""

    __tablename__: ClassVar[str] = "odap"

    id: Optional[int] = Field(default=None, primary_key=True)
    choice: OdapChoice = Field(
        sa_column=Column(
            SQLAlchemyEnum(OdapChoice, values_callable=lambda x: [e.value for e in x])
        )
    )
    gichulqna_id: int = Field(foreign_key="gichulqna.id")
    odapset_id: Optional[int] = Field(default=None, foreign_key="odapset.id")

    gichul_qna: Optional[GichulQna] = Relationship(back_populates="odaps")
    odapset: Optional[OdapSet] = Relationship(back_populates="odaps")
