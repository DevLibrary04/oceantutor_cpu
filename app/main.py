from fastapi import FastAPI
from .routers import auth, solve, modelcall, cbt
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(root_path="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)

app.include_router(solve.router)

app.include_router(modelcall.router)

app.include_router(cbt.router)


@app.get("/")
def read_root():
    return {
        "message": "This is the GET method from the very root end.",
        "endpoints": "You can call /auth, /solve, /modelcall for practical features",
    }
