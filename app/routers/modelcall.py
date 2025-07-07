from fastapi import APIRouter, Depends
from sqlmodel import Session, select
from ..database import get_db


router = APIRouter(prefix="/modelcall", tags=["Call Local or External Models"])


@router.get("/")
def modelcall_root():
    return {
        "message": "This is the GET method response from /modelcall. Development in progress…"
    }
