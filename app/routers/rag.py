# app/routers/rag.py (진짜 최종 완성본)

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.schemas import RAGResponse
from typing import Optional
import base64

# 모든 필요한 서비스와 함수를 import 합니다.
from app.services.rag_service import get_rag_service
from app.services.image_matching_service import get_image_matching_service
from app.ocr_service import get_ocr_reader # 범용 OCR 서비스

router = APIRouter(
    prefix="/rag",
    tags=["RAG"]
)

# OCR 처리를 별도 함수로 분리하여 재사용성을 높입니다.
async def perform_ocr(image_bytes: bytes) -> str:
    try:
        ocr_reader = get_ocr_reader()
        import numpy as np
        import cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        results = ocr_reader.readtext(img_np)
        extracted_text = " ".join([res[1] for res in results])
        return extracted_text
    except Exception as e:
        print(f"OCR 처리 중 오류 발생: {e}")
        return ""


@router.post("/query", response_model=RAGResponse)
async def query_rag_system(question: str = Form(...), image: Optional[UploadFile] = File(None)):
    
    image_b64 = None
    log_message = "\n--- [ROUTER] "
    
    # 1. 이미지가 있는지 확인하고, 있으면 Base64로 인코딩
    if image and image.filename:
        log_message += "이미지 포함 요청 수신 ---"
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="이미지 파일 내용이 비어있습니다.")
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    else:
        log_message += "텍스트 전용 요청 수신 ---"
        
    print(log_message)
    
    # 2. RAG 서비스에 질문과 이미지만 전달하면 끝!
    try:
        rag_service = get_rag_service()
        
        # ⭐⭐⭐ 모든 복잡한 로직은 rag_service가 알아서 처리합니다. ⭐⭐⭐
        result_state = await rag_service.get_answer(
            question=question, 
            image_b64=image_b64
        )
        
        final_answer = result_state.get('generation', '죄송합니다. 답변을 생성하지 못했습니다.')
        return RAGResponse(answer=final_answer)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"RAG 처리 중 오류 발생: {e}")