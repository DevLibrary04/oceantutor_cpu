from fastapi import FastAPI
from .routers import auth, solve, modelcall

app = FastAPI()

app.include_router(auth.router, prefix="/api")

app.include_router(solve.router, prefix="/api")

app.include_router(modelcall.router, prefix="/api")


@app.get("/")
def read_root():
    return {
        "message": "This is the GET method from the very root end.",
        "endpoints": "You can call /auth, /solve, /modelcall for practical features",
    }
