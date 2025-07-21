from datetime import timedelta
from typing import Union
from sqlmodel import Session
from fastapi import HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from ..models import User
from ..crud import user_crud
from ..schemas import CreateUser, CreateUserResponse, Token
from ..core.security import pwd_context, ACCESS_TOKEN_EXPIRE_MINUTES
from ..core.config import settings
from ..utils import user_utils


google_client_id = settings.GOOGLE_CLIENT_ID


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


def sign_google_user(id_token_jwt: Union[str, bytes], db: Session):
    try:
        idinfo = id_token.verify_oauth2_token(
            id_token_jwt,
            google_requests.Request(),
            google_client_id,
        )
        google_user = user_crud.read_one_google_user(idinfo["sub"], db)
        if google_user is None:
            new_user = User(
                indivname=idinfo["name"],
                username=idinfo["email"],
                google_sub=idinfo["sub"],
            )
            user_crud.create_one_google_user(new_user, db)
        else:
            pass
        return {}
    except ValueError:
        # ID 토큰이 유효하지 않은 경우
        raise HTTPException(status_code=401, detail="Invalid Google ID token")
