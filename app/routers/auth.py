import requests
from typing import Annotated
from fastapi import APIRouter, Depends, Request, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from ..core.config import settings
from ..database import get_db
from ..models import UserBase, User
from ..services.user import register_one_user, sign_user_in
from ..dependencies import get_current_active_user
from ..schemas import CreateUser, CreateUserResponse, Token


router = APIRouter(prefix="/auth", tags=["Authentication"])


google_redirect_uri = settings.GOOGLE_REDIRECT_URI
google_client_id = settings.GOOGLE_CLIENT_ID
google_client_secret = settings.GOOGLE_CLIENT_SECRET


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
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    return UserBase(username=current_user.username, indivname=current_user.indivname)


@router.get("/login/google")
async def login_google():
    # 구글 로그인 URL 생성
    # scope는 구글로부터 어떤 정보를 받을지 결정 (openid, email, profile 등)

    return RedirectResponse(
        f"https://accounts.google.com/o/oauth2/v2/auth?client_id={google_client_id}&redirect_uri={google_redirect_uri}&response_type=code&scope=openid%20email%20profile"
    )


@router.get("/sign/google")
async def auth_google_callback(code: str):
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": google_client_id,
        "client_secret": google_client_secret,
        "redirect_uri": google_redirect_uri,
        "grant_type": "authorization_code",
    }
    response = requests.post(token_url, data=data)
    token_data = response.json()

    if "id_token" not in token_data:
        raise HTTPException(
            status_code=400, detail="ID token not found in response from Google"
        )

    id_token_jwt = token_data["id_token"]

    # 2. 발급받은 ID 토큰을 검증합니다.
    try:
        # 이제서야 id_token.verify_oauth2_token을 사용할 수 있습니다.
        idinfo = id_token.verify_oauth2_token(
            id_token_jwt,
            google_requests.Request(),
            google_client_id,
        )
        userid = idinfo["sub"]
        # 여기서 사용자 정보를 DB에 저장하거나 JWT를 생성하여 반환하는 등의 작업을 수행합니다.
        return {"message": f"hello {userid}! validated!", "user_info": idinfo}
    except ValueError:
        # ID 토큰이 유효하지 않은 경우
        raise HTTPException(status_code=401, detail="Invalid Google ID token")
