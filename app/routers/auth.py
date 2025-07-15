from typing import Annotated, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from ..schemas import UserBaseTest, UserInDB


router = APIRouter(prefix="/auth", tags=["Authentication"])


fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "fakehashedsecret",
        "disabled": False,
    },
    "alice": {
        "username": "alice",
        "full_name": "Alice Wonderson",
        "email": "alice@example.com",
        "hashed_password": "fakehashedsecret2",
        "disabled": True,
    },
}


def fake_hash_password(password: str):
    return "fakehashed" + password


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_user(db: Dict[str, UserInDB], username: str) -> Optional[UserInDB]:
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict.model_dump())


def fake_decode_token(token: str) -> UserBaseTest:
    return UserBaseTest(username=token + "fakedecoded", email="john@example.com")


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> UserBaseTest:
    user = fake_decode_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid auth credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_current_active_user(
    current_user: Annotated[UserBaseTest, Depends(get_current_user)],
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="inactive user")
    return current_user


# @router.post("/token")
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user_dict = fake_users_db.get(form_data.username)
    if not user_dict:
        raise HTTPException(status_code=400, detail="incorrect username or password")
    user = UserInDB(**user_dict)
    hashed_password = fake_hash_password(form_data.password)
    if not hashed_password == user.hashed_password:
        raise HTTPException(status_code=400, detail="incorrect username or password")
    return {"access_token": user.username, "token_type": "bearer"}


# @router.get("/users/me", response_model=UserBaseTest)
async def auth_root(
    current_user: Annotated[UserBaseTest, Depends(get_current_active_user)],
):
    return current_user
