from datetime import datetime, timedelta, timezone
from typing import Annotated
from sqlmodel import Session
from fastapi import HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from ..models import User
from ..crud import user_crud
from ..schemas import CreateUser, CreateUserResponse, Token
from ..core.security import pwd_context, ACCESS_TOKEN_EXPIRE_MINUTES
from ..utils import user_utils


def register_one_user(
    user_in: CreateUser,
    db: Session,
):
    user = user_crud.read_one_user(user_in.username, db)
    if user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="User already registered"
        )
    hashed_password = pwd_context.hash(user_in.password)
    user_in_dict = user_in.model_dump(exclude={"password"})
    user_in_dict.update({"hashed_password": hashed_password})
    regi_user = User(**user_in_dict)
    try:
        db_user = user_crud.create_one_user(regi_user, db)
        db.commit()
        db.refresh(db_user)
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register the user for an internal error",
        )
    return CreateUserResponse(email=db_user.username, name=db_user.indivname)


def sign_user_in(form_data: OAuth2PasswordRequestForm, db: Session):
    db_user = user_utils.authenticate_user(form_data, db)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = user_utils.create_access_token(
        {"sub": db_user.username}, access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")
