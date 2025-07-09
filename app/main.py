from fastapi import FastAPI
from .routers import auth, solve, modelcall
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api")

app.include_router(solve.router, prefix="/api")

app.include_router(modelcall.router, prefix="/api")


@app.get("/")
def read_root():
    return {
        "message": "This is the GET method from the very root end.",
        "endpoints": "You can call /auth, /solve, /modelcall for practical features",
    }
