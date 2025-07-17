import jwt
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional, Dict
from sqlmodel import Session
from fastapi import HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from ..models import User
from ..crud import user_crud
from ..schemas import CreateUser, CreateUserResponse
from ..core.security import pwd_context, SECRET_KEY, ALGORITHM


def verify_password(plain_password_from_front: str, hashed_password: str):
    return pwd_context.verify(plain_password_from_front, hashed_password)


def authenticate_user(form_data: OAuth2PasswordRequestForm, db: Session):
    db_user = user_crud.read_one_user(form_data.username, db)
    if not db_user:
        return False
    if not verify_password(form_data.password, db_user.hashed_password):
        return False
    return db_user


def create_access_token(valid_user: Dict, expires_delta: Optional[timedelta] = None):
    to_encode = valid_user.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
