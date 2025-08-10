from fastapi import FastAPI
from app.routers import rag as rag_router
from app.schemas import RootResponse
from fastapi.middleware.cors import CORSMiddleware
from app.rag import config
import os

app = FastAPI(root_path="/api")

@app.on_event("startup")
def on_startup():
    """애플리케이션 시작 시 모든 서비스를 초기화합니다."""
    os.makedirs(config.TEMP_UPLOAD_DIR, exist_ok=True)

    from app.services.rag_service import get_rag_service
    rag_service = get_rag_service()
    rag_service.initialize()
    
    from app.services.image_matching_service import get_image_matching_service
    image_matching_service = get_image_matching_service()
    image_matching_service.initialize()
    
    print("FastAPI application startup complete. All services are ready.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router.router)

@app.get("/", response_model=RootResponse)
def read_root():
    return {
        "message": "This is the GET method from the very root end.",
        "endpoints": "You can call /rag for practical features",
    }