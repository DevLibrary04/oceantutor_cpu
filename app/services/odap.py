from typing import Annotated, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, SQLModel, Field
from app.schemas import UserSolvedQna, ManyOdaps
from app.models import Odap, User
from app.crud.user_crud import read_one_user
from app.crud import odap_crud, odapset_crud


def save_user_solved_qna(odap_data: UserSolvedQna, current_user: User, db: Session):
    user = read_one_user(current_user.username, db)
    if user is None:
        raise HTTPException(
            status_code=401, detail="This feature is only for singned users"
        )
    new_odap = Odap(
        choice=odap_data.choice,
        gichulqna_id=odap_data.gichulqna_id,
        odapset_id=odap_data.odapset_id,
    )
    try:
        odap_crud.create_one_odap(new_odap, db)
        db.commit()
        db.refresh(new_odap)
    except:
        db.rollback()
        raise HTTPException(status_code=404)
    return new_odap


def save_user_solved_many_qnas(odaps: ManyOdaps, current_user: User, db: Session):
    try:
        user = read_one_user(current_user.username, db)
        if user is None:
            raise HTTPException(
                status_code=401, detail="This feature is only for singned users"
            )
        odaplist = [
            Odap(
                choice=odap.choice,
                gichulqna_id=odap.gichulqna_id,
                odapset_id=odaps.odapset_id,
            )
            for odap in odaps.odaps
        ]
        odap_crud.create_many_odaps(odaplist, db)
        db.commit()
    except:
        db.rollback()
        raise HTTPException(404)
    return odaps


def retrieve_user_saved_many_qnas(current_user: User, db: Session):
    assert current_user.id is not None
    return odapset_crud.read_many_odapsets(current_user.id, db)
