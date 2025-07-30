# app/main.py (수정된 최종본)

from fastapi import FastAPI
from app.routers import auth, solve, modelcall, cbt, odap, rag as rag_router
from app.schemas import RootResponse
from fastapi.middleware.cors import CORSMiddleware
from app.rag import config
import os

app = FastAPI(root_path="/api")

@app.on_event("startup")
def on_startup():
    """애플리케이션 시작 시 모든 서비스를 초기화합니다."""
    # 필요한 폴더 생성
    os.makedirs(config.TEMP_UPLOAD_DIR, exist_ok=True)

    # 기존 RAG 서비스 초기화
    from app.services.rag_service import get_rag_service
    rag_service = get_rag_service()
    rag_service.initialize()

    # [추가] 신규 이미지 매칭 서비스 초기화
    from app.services.image_matching_service import get_image_matching_service
    image_matching_service = get_image_matching_service()
    image_matching_service.initialize()
    
    print("FastAPI application startup complete. All services are ready.")

# ... (기존 미들웨어 및 라우터 포함 코드) ...
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.include_router(auth.router)
# app.include_router(solve.router)
# app.include_router(modelcall.router)
# app.include_router(cbt.router)
# app.include_router(odap.router)
app.include_router(rag_router.router)

@app.get("/", response_model=RootResponse)
def read_root():
    return {
        "message": "This is the GET method from the very root end.",
        "endpoints": "You can call /auth, /solve, /modelcall, /rag for practical features",
    }