from sqlmodel import Session
from typing import List, Optional, ClassVar

from sqlmodel import (
    SQLModel,
    Field,
    Column,
    Relationship,
    TIMESTAMP,
    Text,
    create_engine,
    select,
)
from dotenv import load_dotenv
import os

load_dotenv()


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


class User(SQLModel, table=True):
    """사용자 정보 테이블"""

    __tablename__: ClassVar[str] = "user"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=45, unique=True, index=True)
    password: str = Field(max_length=255)

    # chats: List["Chat"] = Relationship(back_populates="user")
    # odaps: List["Odap"] = Relationship(back_populates="user")


def create_users(engine):
    with Session(engine) as session:
        for n in range(10):
            user = User(name=f"member{n}", password=f"abcd{n}")
            session.add(user)
        session.commit()


def select_users(engine):
    with Session(engine) as session:
        st = select(User).where(User.name == "member9")
        results = session.exec(st)
        for user in results:
            print(user)


def update_users(engine):
    with Session(engine) as session:
        results = session.exec(select(User)).all()
        for user in results:
            user.name = f"user{user.id}"
            user.password = f"pass{user.id}"
            session.add(user)
        session.commit()


def delete_users(engine):
    with Session(engine) as session:
        results = session.exec(select(User)).all()
        for user in results:
            session.delete(user)
        session.commit()


def main():
    engine = run_engine()
    create_users(engine)
    # select_users(engine)
    update_users(engine)
    # delete_users(engine)


if __name__ == "__main__":
    main()
