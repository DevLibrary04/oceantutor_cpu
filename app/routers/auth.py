from fastapi import APIRouter, Depends
from sqlmodel import Session, select
from ..database import get_db


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.get("/")
def auth_root():
    return {
        "message": "This is the GET method response from /auth. Development in progress…"
    }
