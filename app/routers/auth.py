from typing import Annotated
from fastapi import APIRouter, Depends, Request, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session
from google.oauth2 import id_token
from google.auth.transport import requests
from ..database import get_db
from ..models import UserBase
from ..services.user import register_one_user, sign_user_in
from ..dependencies import get_current_active_user
from ..schemas import CreateUser, CreateUserResponse, Token


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/signup", response_model=CreateUserResponse)
async def user_signup(
    user_in: CreateUser,
    db: Annotated[Session, Depends(get_db)],
):
    return register_one_user(user_in, db)


@router.post("/token", response_model=Token)
async def sign_user_in_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[Session, Depends(get_db)],
):
    return sign_user_in(form_data, db)


@router.get("/sign/me", response_model=UserBase)
async def get_user_info(
    current_user: Annotated[UserBase, Depends(get_current_active_user)],
):
    return current_user


@router.get("/login/google")
async def login_google():
    # 구글 로그인 URL 생성
    # scope는 구글로부터 어떤 정보를 받을지 결정 (openid, email, profile 등)
    return RedirectResponse(
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id=878361365167-2tk6hbj84tgs2sqfm675uqouqvre98tn.apps.googleusercontent.com"
        f"&redirect_uri=http://localhost:8000/api/auth/sign/google"
        f"&response_type=code"
        f"&scope=openid%20email%20profile"
    )


@router.post("/sign/google")
async def validate_google_token(token: str = Query()):
    try:
        idinfo = id_token.verify_oauth2_token(
            token,
            requests.Request(),
            "878361365167-2tk6hbj84tgs2sqfm675uqouqvre98tn.apps.googleusercontent.com",
        )
        userid = idinfo["sub"]
        return {"message": "hello! validated!"}
    except ValueError:
        pass
