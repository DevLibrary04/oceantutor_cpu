# routers/rag.py

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import base64
from app.schemas import RAGRequest, RAGResponse
from typing import Optional, Union

import pytesseract
from PIL import Image
import io

import cv2
import numpy as np
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


router = APIRouter(
    prefix="/rag",
    tags=["RAG"]
)

async def improved_ocr_processing(image_bytes):
    """개선된 OCR 전처리 함수"""
    
    print("--- [ROUTER] 개선된 이미지 OCR 전처리 시작 ---")
    
    # NumPy 배열로 변환
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # 원본 이미지 저장 (디버깅용)
    cv2.imwrite("original_image.png", img_cv)
    
    # 여러 전처리 방법을 시도해보는 함수들
    def method_1_simple(img):
        """방법 1: 단순 그레이스케일 + 확대"""
        # 3배 확대 (더 큰 확대율)
        height, width = img.shape[:2]
        resized = cv2.resize(img, (width*3, height*3), interpolation=cv2.INTER_CUBIC)
        
        # 그레이스케일
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        return gray
    
    def method_2_adaptive_threshold(img):
        """방법 2: 적응적 임계값 처리"""
        height, width = img.shape[:2]
        resized = cv2.resize(img, (width*3, height*3), interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # 가우시안 블러 (노이즈 제거)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 적응적 임계값 (THRESH_BINARY 사용 - INV 제거)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def method_3_otsu(img):
        """방법 3: Otsu 이진화"""
        height, width = img.shape[:2]
        resized = cv2.resize(img, (width*3, height*3), interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # 약간의 블러
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Otsu 이진화
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def method_4_morphology(img):
        """방법 4: 모폴로지 연산 추가"""
        height, width = img.shape[:2]
        resized = cv2.resize(img, (width*3, height*3), interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Otsu 이진화
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 모폴로지 연산 (노이즈 제거)
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    # --- [새로운 방법 5] Contour Detection 추가 ---
    def method_5_contour_detection(img):
        """방법 5: 윤곽선 검출로 텍스트 영역만 추출"""
        height, width = img.shape[:2]
        resized = cv2.resize(img, (width*3, height*3), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Otsu 이진화로 글자와 배경 분리
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 윤곽선을 감싸는 최소 사각형(Bounding Box)들의 리스트 생성
        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # 너무 작거나 너무 큰 윤곽선은 노이즈일 가능성이 높으므로 제외
            # 이 값들은 이미지 특성에 맞게 튜닝이 필요할 수 있습니다.
            if 10 < w < 300 and 10 < h < 100:
                bounding_boxes.append((x, y, w, h))

        # 모든 텍스트 영역을 포함하는 새로운 캔버스(배경) 생성
        # 원본 이미지와 동일한 크기의 흰색 배경을 만듭니다.
        mask = np.full_like(gray, 255, dtype=np.uint8)
        
        # 디버깅용: 윤곽선을 원본 이미지에 그려서 확인
        img_with_contours = resized.copy()

        # 찾은 텍스트 영역(Bounding Box)만 검은색 글씨로 캔버스에 복사
        for x, y, w, h in bounding_boxes:
            # 텍스트 영역을 원본(그레이스케일)에서 잘라냅니다.
            roi = gray[y:y+h, x:x+w]
            # 해당 영역을 새로운 캔버스에 붙여넣습니다.
            mask[y:y+h, x:x+w] = roi
            # 디버깅용으로 사각형 그리기
            cv2.rectangle(img_with_contours, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 디버깅용 이미지 저장
        cv2.imwrite("debug_contours.png", img_with_contours)
        
        # 최종적으로 Otsu 이진화를 한번 더 적용하여 깔끔하게 만듭니다.
        _, final_thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return final_thresh
    
    # 각 방법별로 OCR 시도
    methods = [
        ("단순_확대", method_1_simple),
        ("적응적_임계값", method_2_adaptive_threshold),
        ("Otsu_이진화", method_3_otsu),
        ("모폴로지_정리", method_4_morphology),
        ("윤곽선_검출", method_5_contour_detection)
    ]
    
    best_result = ""
    best_confidence = 0
    
    for method_name, method_func in methods:
        try:
            print(f"--- 시도 중: {method_name} ---")
            
            processed_img = method_func(img_cv)
            
            # 전처리된 이미지 저장 (디버깅용)
            cv2.imwrite(f"preprocessed_{method_name}.png", processed_img)
            
            # 여러 PSM 모드 시도
            psm_configs = [
                '--oem 3 --psm 6',  # 균일한 텍스트 블록
                '--oem 3 --psm 11', # 흩어진 텍스트
                '--oem 3 --psm 13', # 한 줄 텍스트
                '--oem 3 --psm 8',  # 단어로 취급
            ]
            
            for config in psm_configs:
                try:
                    # OCR 실행
                    result = pytesseract.image_to_string(processed_img, lang='kor', config=config)
                    
                    # 신뢰도 정보도 가져오기
                    data = pytesseract.image_to_data(processed_img, lang='kor', config=config, output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    print(f"    PSM {config.split('psm')[1].split()[0]}: 신뢰도 {avg_confidence:.1f}% | 길이: {len(result.strip())}")
                    
                    if len(result.strip()) > len(best_result.strip()) and avg_confidence > 30:
                        best_result = result
                        best_confidence = avg_confidence
                        print(f"    → 현재 최적 결과로 업데이트!")
                
                except Exception as e:
                    print(f"    PSM 설정 오류: {e}")
                    continue
        
        except Exception as e:
            print(f"{method_name} 방법 실패: {e}")
            continue
    
    print(f"\n--- 최종 OCR 결과 (신뢰도: {best_confidence:.1f}%) ---")
    print(best_result)
    print("--- OCR 완료 ---")
    
    # return best_result


    print(f"\n--- OCR 원본 결과 (정제 전) ---")
    print(best_result)
    print("--------------------------------")

    # --- [새로운 단계] OCR 결과에서 유의미한 키워드만 추출 ---
    def refine_ocr_text(ocr_text: str) -> str:
        # 정규표현식을 사용하여 한글 단어만 추출합니다.
        # [가-힣]+ 는 연속된 한글 문자 1개 이상을 의미합니다.
        korean_words = re.findall(r'[가-힣]+', ocr_text)
        
        # 의미 있을 가능성이 높은, 2글자 이상의 단어만 필터링합니다.
        # 또는 모든 단어를 사용하려면 이 필터를 제거할 수 있습니다.
        meaningful_words = [word for word in korean_words if len(word) >= 1]
        
        # 중복을 제거하고 원래 순서를 유지합니다.
        unique_words = list(dict.fromkeys(meaningful_words))
        
        # 정제된 키워드들을 공백으로 구분하여 하나의 문자열로 합칩니다.
        refined_text = " ".join(unique_words)
        return refined_text

    final_text = refine_ocr_text(best_result)
    # --- 새로운 단계 끝 ---
    
    print(f"\n--- 최종 정제된 OCR 키워드 ---")
    print(final_text)
    print("---------------------------------")
    
    return final_text




@router.post("/query", response_model=RAGResponse)
async def query_rag_system(question: str = Form(...), image: Union[UploadFile, str] = File(None)):
    """
    멀티모달 RAG 시스템에 질문과 이미지를 전달하여 답변을 요청합니다.
    - **question**: 사용자의 질문 텍스트 (Form 데이터)
    - **image**: 이미지 파일 (Form 데이터)
    """
    from app.services.rag_service import get_rag_service

    rag_service = get_rag_service()

    print("라우터 /rag/query 요청 수신")
    print(f" 질문: {question}")
    print(f" 이미지 파일명: {image.filename if image else '없음'}")

    image_b64 = None
    extracted_text = None


    if image:

        print(f" -이미지 파일명: {image.filename}")
        image_bytes = await image.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        print(f"--- [ROUTER 진단] 이미지 인코딩 완료. Base64 길이: {len(image_b64)} ---")

        print("--- [ROUTER] 이미지에서 Pytesseract OCR 수행 중... ---")
        try:
            extracted_text = await improved_ocr_processing(image_bytes)
            
            print(f"--- [ROUTER] OCR 결과:\n{extracted_text}\n---")
          
        except pytesseract.TesseractNotFoundError:
            print("!!!!!! 에러: Tesseract를 찾을 수 없습니다. `tesseract_cmd` 경로가 올바른지 확인하세요. !!!!!!")
            raise HTTPException(status_code=500, detail="Tesseract-OCR 엔진을 찾을 수 없습니다.")
        except Exception as e:
            print(f"!!!!!! 에러: OCR 처리 중 오류 발생: {e} !!!!!!")
            # OCR 실패 시에도 에러를 내지 않고 계속 진행하도록 할 수 있습니다.
            extracted_text = "" # 또는 None

    else:
        print(f" -이미지 파일명: 없습니다 (수신된 값: '{image}')")


    try:
        # [수정] 실제 rag_service 호출
        print("라우터 rag_service.get_answer 호출 시작")

        result_state = await rag_service.get_answer(
            question=question,
            image_b64=image_b64,
            extracted_text=extracted_text
        )
        
        print("라우터 rag_service.get_answer 호출 완료")
        final_answer = result_state.get('generate', {}).get('generation', '죄송합니다. 답변을 생성하지 못했습니다.')
        return RAGResponse(answer=final_answer)

    except Exception as e:
        print(f"Error during RAG query: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error while processing the RAG query.")