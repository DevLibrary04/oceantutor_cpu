from typing import Literal, Optional
from sqlmodel import Session, select
from ..models import User


def read_one_user(
    username: str,
    db: Session,
) -> Optional[User]:
    user = db.exec(select(User).where(User.username == username)).one_or_none()
    return user


def create_one_user(user: User, db: Session):
    db.add(user)
    return user


def read_one_google_user(google_sub: str, db: Session):
    user = db.exec(select(User).where(User.google_sub == google_sub)).one_or_none()
    return user


def create_one_google_user(user: User, db: Session):
    db.add(user)
    return user
