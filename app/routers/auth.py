from typing import Annotated
from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session
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
