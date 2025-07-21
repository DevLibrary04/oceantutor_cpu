from fastapi import FastAPI
from .routers import auth, solve, modelcall, cbt, odap
from .schemas import RootResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from .routers import auth, solve, modelcall, cbt, odap
from .routers import rag as rag_router
from .services.rag_service import rag_service


app = FastAPI(root_path="/api")

@app.on_event("startup")
def on_startup():
    """
    애플리케이션이 시작될 때 RAG 서비스를 초기화합니다.
    모델 로딩 등 무거운 작업은 이 시점에 한 번만 수행됩니다.
    """
    rag_service.initialize()
    print("FastAPI application startup complete. All services are ready.")



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

app.include_router(odap.router)

app.include_router(rag_router.router)


@app.get("/", response_model=RootResponse)
def read_root():
    return {
        "message": "This is the GET method from the very root end.",
        "endpoints": "You can call /auth, /solve, /modelcall, /rag for practical features",
    }
