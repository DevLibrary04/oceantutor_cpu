from typing import Annotated, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, status, logger
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlmodel import Session
from ..database import get_db
from ..crud import user_crud
from ..models import UserBase
from ..services.user import register_one_user, sign_user_in
from ..schemas import CreateUser, CreateUserResponse


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/signup", response_model=CreateUserResponse)
async def user_signup(
    user_in: CreateUser,
    db: Annotated[Session, Depends(get_db)],
):
    return register_one_user(user_in, db)


@router.post("/token")
async def sign_user_in_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[Session, Depends(get_db)],
):
    return sign_user_in(form_data, db)


@router.get("/me")
async def protected_endpoint_test():
    pass
