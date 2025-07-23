# # app/routers/langserve_router.py

# from fastapi import APIRouter
# from langserve import add_routes
# import uvicorn
# from app.services.rag_service import get_rag_service
# from dotenv import load_dotenv

# load_dotenv()

# router = APIRouter()

# # 싱글톤 객체에서 LLM 체인을 불러옴
# rag_service = get_rag_service()
# rag_service.initialize()

# # LangGraph 체인을 LangServe 방식으로 노출
# add_routes(
#     router,
#     rag_service.rag_app,     # 이미 컴파일된 LangGraph 앱
#     path="/langserve",       # 호출 경로: /langserve/invoke
#     enable_feedback_endpoint=True,
#     enable_public_trace_link_endpoint=True,
#     playground_type="chat"
# )

