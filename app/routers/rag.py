from fastapi import APIRouter, UploadFile, File, Form, HTTPException
# from ..schemas.rag_schemas import RAGRequest, RAGResponse # 이 스키마는 이제 사용되지 않음
from ..services.rag_service import rag_service
import base64
from ..utils.schemas import RAGRequest, RAGResponse


router = APIRouter(
    prefix="/rag",
    tags=["RAG"]
)

@router.post("/query", response_model=RAGResponse)
async def query_rag_system(question: str = Form(...), image: UploadFile = File(None)):
    """
    멀티모달 RAG 시스템에 질문과 이미지를 전달하여 답변을 요청합니다.
    - **question**: 사용자의 질문 텍스트 (Form 데이터)
    - **image**: 이미지 파일 (Form 데이터)
    """
    image_b64 = None
    if image:
        # 파일 확장자 검사 (선택사항이지만 권장)
        allowed_extensions = {"png", "jpg", "jpeg"}
        if image.filename.split('.')[-1].lower() not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Image must be in PNG, JPG, or JPEG format.")
        
        image_bytes = await image.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        # [수정] 실제 rag_service 호출
        result_state = await rag_service.get_answer(
            question=question,
            image_b64=image_b64
        )
        
        final_answer = result_state.get('generate', {}).get('generation', '죄송합니다. 답변을 생성하지 못했습니다.')
        return RAGResponse(answer=final_answer)

    except Exception as e:
        print(f"Error during RAG query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while processing the RAG query.")