# # # routers/rag.py

# app/routers/rag.py

import cv2
import numpy as np
import re
from PIL import Image
import io
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import base64
from app.schemas import RAGRequest, RAGResponse
from typing import Optional

# 새로 만든 OCR 서비스 함수를 import 합니다.
from app.ocr_service import get_ocr_reader

router = APIRouter(
    prefix="/rag",
    tags=["RAG"]
)

async def improved_ocr_processing(image_bytes):
    """EasyOCR을 사용한 개선된 OCR 전처리 함수"""
    
    print("--- [ROUTER] EasyOCR 이미지 처리 시작 ---")
    
    # NumPy 배열로 변환
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    cv2.imwrite("original_image.png", img_cv)
    
    def method_1_original(img):
        return img
    
    def method_2_enhanced_preprocessing(img):
        height, width = img.shape[:2]
        scale_factor = 2 if height < 1000 or width < 1000 else 1
        if scale_factor > 1:
            resized = cv2.resize(img, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)
        else:
            resized = img
        
        if len(resized.shape) == 3:
            lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(resized)
        return enhanced
    
    methods = [
        ("원본", method_1_original),
        ("향상된_전처리", method_2_enhanced_preprocessing),
    ]
    
    best_result_text = ""
    best_confidence = 0
    best_method = ""
    best_detection_count = 0
    
    for method_name, method_func in methods:
        try:
            print(f"--- OCR 시도 중: {method_name} ---")
            processed_img = method_func(img_cv)
            cv2.imwrite(f"preprocessed_{method_name}.png", processed_img)
            
            # ★ 수정: get_ocr_reader() 함수를 호출하여 싱글톤 인스턴스를 가져옵니다.
            ocr_reader = get_ocr_reader()
            result = ocr_reader.readtext(processed_img, detail=1, width_ths=0.7, height_ths=0.7)
            
            if not result:
                continue

            detected_texts = [item[1] for item in result if item[2] > 0.3]
            confidences = [item[2] for item in result if item[2] > 0.3]
            
            if not confidences:
                continue

            avg_confidence = (sum(confidences) / len(confidences)) * 100
            current_text = " ".join(detected_texts)
            korean_chars = len(re.findall(r'[가-힣]', current_text))
            
            print(f"    - 감지된 텍스트 블록: {len(detected_texts)}개, 평균 신뢰도: {avg_confidence:.1f}%, 한글 문자 수: {korean_chars}개")

            # 가장 긴 텍스트를 가진 결과를 최선으로 선택 (더 정교한 로직도 가능)
            if len(current_text) > len(best_result_text):
                best_result_text = current_text
                best_confidence = avg_confidence
                best_method = method_name
                best_detection_count = len(detected_texts)
                print(f"    → 현재 최적 결과로 업데이트!")

        except Exception as e:
            print(f"    - {method_name} 방법 실패: {e}")
            continue
    
    print(f"\n--- 최종 EasyOCR 결과 ---")
    print(f"최적 방법: {best_method}, 평균 신뢰도: {best_confidence:.1f}%, 감지 블록 수: {best_detection_count}개")
    
    # OCR 결과 후처리
    def refine_easyocr_text(ocr_text: str) -> str:
        text = ocr_text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    final_text = refine_easyocr_text(best_result_text)
    
    print(f"\n--- 최종 정제된 OCR 텍스트 ---")
    print(final_text)
    print("---------------------------------------")
    
    return final_text


@router.post("/query", response_model=RAGResponse)
async def query_rag_system(question: str = Form(...), image: Optional[UploadFile] = File(None)):
    from app.services.rag_service import get_rag_service
    
    print("\n\n--- [최신 버전 라우터 코드 실행 v5.0 - 덕 타이핑] ---\n\n")

    rag_service = get_rag_service()

    print("라우터 /rag/query 요청 수신")
    print(f" 질문: {question}")
    print(f" 수신된 image 파라미터 타입: {type(image)}")

    image_b64 = None
    extracted_text = None

    # --- [★ 이 부분을 덕 타이핑 방식으로 수정 ★] ---
    # isinstance 대신, 객체가 파일 객체처럼 필요한 속성을 가지고 있는지 확인합니다.
    # hasattr(object, 'attribute_name')은 객체에 해당 속성이 있는지 여부를 True/False로 반환합니다.
    is_valid_file = image is not None and hasattr(image, 'filename') and hasattr(image, 'read')

    if is_valid_file and image.filename:
        print(f" - 이미지 파일 수신됨 (덕 타이핑으로 확인): {image.filename}")
        
        image_bytes = await image.read()

        if not image_bytes:
            print("  - 경고: 이미지 파일 객체는 수신되었으나 실제 내용(bytes)이 비어있습니다.")
        else:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            print(f"--- [ROUTER 진단] 이미지 인코딩 완료. Base64 길이: {len(image_b64)} ---")
            
            try:
                extracted_text = await improved_ocr_processing(image_bytes)
            except Exception as e:
                print(f"!!!!!! 에러: OCR 처리 중 오류 발생: {e} !!!!!!")
                extracted_text = ""
    else:
        print(f" - 이미지 파일이 없습니다. (수신된 값: '{image}')")
    # --- [수정 끝] ---


    try:
        print("라우터 rag_service.get_answer 호출 시작")
        result_state = await rag_service.get_answer(
            question=question,
            image_b64=image_b64,
            extracted_text=extracted_text
        )
        print("라우터 rag_service.get_answer 호출 완료")

        generation_node_output = result_state.get('generate', {})
        final_answer = generation_node_output.get('generation', '죄송합니다. 답변을 생성하지 못했습니다.')
        
        print(f"--- [ROUTER] 최종 응답(문자열) 추출 완료: {final_answer[:100]}...")
        return RAGResponse(answer=final_answer)

    except Exception as e:
        print(f"Error during RAG query: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error while processing the RAG query.")





















































# import cv2
# import numpy as np
# import re
# from PIL import Image
# import io
# # from paddleocr import PaddleOCR
# from fastapi import APIRouter, UploadFile, File, Form, HTTPException
# import base64
# from app.schemas import RAGRequest, RAGResponse
# from typing import Optional, Union
# import easyocr
# # import pytesseract


# router = APIRouter(
#     prefix="/rag",
#     tags=["RAG"]
# )

# # EasyOCR 초기화 (한국어와 영어)
# ocr_reader = easyocr.Reader(['ko', 'en'], gpu=False)  # GPU 사용 원하면 gpu=True



# async def improved_ocr_processing(image_bytes):
#     """EasyOCR을 사용한 개선된 OCR 전처리 함수"""
    
#     print("--- [ROUTER] EasyOCR 이미지 처리 시작 ---")
    
#     # NumPy 배열로 변환
#     np_arr = np.frombuffer(image_bytes, np.uint8)
#     img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
#     # 원본 이미지 저장 (디버깅용)
#     cv2.imwrite("original_image.png", img_cv)
    
#     def method_1_original(img):
#         """방법 1: 원본 이미지 그대로"""
#         return img
    
#     def method_2_enhanced_preprocessing(img):
#         """방법 2: 향상된 전처리"""
#         # 이미지 크기 확인 및 적절한 크기로 조정
#         height, width = img.shape[:2]
        
#         # 너무 작으면 2배 확대
#         if height < 1000 or width < 1000:
#             scale_factor = 2
#             resized = cv2.resize(img, (width * scale_factor, height * scale_factor), 
#                                interpolation=cv2.INTER_CUBIC)
#         else:
#             resized = img
        
#         # 히스토그램 균등화로 대비 향상
#         if len(resized.shape) == 3:
#             # 컬러 이미지인 경우 LAB 색공간에서 L 채널만 균등화
#             lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
#             l, a, b = cv2.split(lab)
#             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#             l = clahe.apply(l)
#             enhanced = cv2.merge([l, a, b])
#             enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
#         else:
#             # 그레이스케일인 경우
#             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#             enhanced = clahe.apply(resized)
        
#         return enhanced
    
#     def method_3_denoising(img):
#         """방법 3: 노이즈 제거 특화"""
#         height, width = img.shape[:2]
        
#         if height < 1000 or width < 1000:
#             scale_factor = 2
#             resized = cv2.resize(img, (width * scale_factor, height * scale_factor), 
#                                interpolation=cv2.INTER_CUBIC)
#         else:
#             resized = img
        
#         # 양방향 필터로 노이즈 제거하면서 경계 보존
#         if len(resized.shape) == 3:
#             denoised = cv2.bilateralFilter(resized, 9, 75, 75)
#         else:
#             denoised = cv2.bilateralFilter(resized, 9, 75, 75)
        
#         return denoised
    
#     def method_4_sharpening(img):
#         """방법 4: 선명도 향상"""
#         height, width = img.shape[:2]
        
#         if height < 1000 or width < 1000:
#             scale_factor = 2
#             resized = cv2.resize(img, (width * scale_factor, height * scale_factor), 
#                                interpolation=cv2.INTER_LANCZOS4)
#         else:
#             resized = img
        
#         # 언샤프 마스킹으로 선명도 향상
#         if len(resized.shape) == 3:
#             # 컬러 이미지
#             gaussian = cv2.GaussianBlur(resized, (5, 5), 1.0)
#             sharpened = cv2.addWeighted(resized, 1.5, gaussian, -0.5, 0)
#         else:
#             # 그레이스케일
#             gaussian = cv2.GaussianBlur(resized, (5, 5), 1.0)
#             sharpened = cv2.addWeighted(resized, 1.5, gaussian, -0.5, 0)
        
#         return sharpened
    
#     def method_5_grayscale_optimized(img):
#         """방법 5: 그레이스케일 최적화"""
#         height, width = img.shape[:2]
        
#         if height < 1000 or width < 1000:
#             scale_factor = 2
#             resized = cv2.resize(img, (width * scale_factor, height * scale_factor), 
#                                interpolation=cv2.INTER_CUBIC)
#         else:
#             resized = img
        
#         # 그레이스케일 변환
#         if len(resized.shape) == 3:
#             gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = resized
        
#         # 히스토그램 균등화
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#         enhanced = clahe.apply(gray)
        
#         # 미디안 필터로 점 노이즈 제거
#         filtered = cv2.medianBlur(enhanced, 3)
        
#         # 다시 컬러로 변환 (EasyOCR도 컬러를 선호)
#         color_img = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        
#         return color_img
    
#     # 각 방법별로 OCR 시도
#     methods = [
#         ("원본", method_1_original),
#         ("향상된_전처리", method_2_enhanced_preprocessing),
#         ("노이즈_제거", method_3_denoising),
#         ("선명도_향상", method_4_sharpening),
#         ("그레이스케일_최적화", method_5_grayscale_optimized)
#     ]
    
#     best_result = ""
#     best_confidence = 0
#     best_method = ""
#     best_detection_count = 0
    
#     for method_name, method_func in methods:
#         try:
#             print(f"--- 시도 중: {method_name} ---")
            
#             processed_img = method_func(img_cv)
            
#             # 전처리된 이미지 저장 (디버깅용)
#             cv2.imwrite(f"preprocessed_{method_name}.png", processed_img)
            
#             # EasyOCR로 텍스트 인식
#             try:
#                 # EasyOCR 실행 (여러 파라미터 조합 시도)
#                 ocr_configs = [
#                     {'detail': 1, 'width_ths': 0.7, 'height_ths': 0.7},  # 기본 설정
#                     {'detail': 1, 'width_ths': 0.5, 'height_ths': 0.5},  # 더 민감하게
#                     {'detail': 1, 'width_ths': 0.9, 'height_ths': 0.9},  # 더 엄격하게
#                 ]
                
#                 best_config_result = ""
#                 best_config_confidence = 0
#                 best_config_count = 0
                
#                 for config in ocr_configs:
#                     try:
#                         result = ocr_reader.readtext(processed_img, **config)
                        
#                         if result:
#                             # 결과 파싱
#                             detected_texts = []
#                             total_confidence = 0
#                             detection_count = 0
                            
#                             for detection in result:
#                                 # EasyOCR 결과 형태: [bbox, text, confidence]
#                                 if len(detection) >= 3:
#                                     bbox, text, confidence = detection[0], detection[1], detection[2]
                                    
#                                     if confidence > 0.3:  # 신뢰도 30% 이상만 사용
#                                         detected_texts.append(text)
#                                         total_confidence += confidence
#                                         detection_count += 1
                            
#                             # 전체 텍스트 결합
#                             combined_text = " ".join(detected_texts)
#                             avg_confidence = (total_confidence / detection_count * 100) if detection_count > 0 else 0
                            
#                             # 현재 설정에서 가장 좋은 결과 선택
#                             if avg_confidence > best_config_confidence:
#                                 best_config_result = combined_text
#                                 best_config_confidence = avg_confidence
#                                 best_config_count = detection_count
                    
#                     except Exception as e:
#                         print(f"    설정 {config} 처리 오류: {e}")
#                         continue
                
#                 # 한글 문자 수 계산
#                 korean_chars = len(re.findall(r'[가-힣]', best_config_result))
                
#                 print(f"    감지된 텍스트 블록: {best_config_count}개")
#                 print(f"    평균 신뢰도: {best_config_confidence:.1f}%")
#                 print(f"    한글 문자 수: {korean_chars}개")
#                 print(f"    텍스트 샘플: {best_config_result[:100]}...")
                
#                 # 결과 품질 평가 (한글 문자 수 + 신뢰도 + 감지 블록 수)
#                 score = korean_chars * 3 + best_config_confidence + best_config_count * 2
#                 best_score = len(re.findall(r'[가-힣]', best_result)) * 3 + best_confidence + best_detection_count * 2
                
#                 if score > best_score and best_config_confidence > 20:
#                     best_result = best_config_result
#                     best_confidence = best_config_confidence
#                     best_method = method_name
#                     best_detection_count = best_config_count
#                     print(f"    → 현재 최적 결과로 업데이트! (점수: {score:.1f})")
                    
#             except Exception as e:
#                 print(f"    EasyOCR 처리 오류: {e}")
#                 continue
        
#         except Exception as e:
#             print(f"{method_name} 방법 실패: {e}")
#             continue
    
#     print(f"\n--- 최종 EasyOCR 결과 ---")
#     print(f"최적 방법: {best_method}")
#     print(f"평균 신뢰도: {best_confidence:.1f}%")
#     print(f"감지 블록 수: {best_detection_count}개")
#     print("--- OCR 완료 ---")
    
#     # OCR 결과 후처리
#     def refine_easyocr_text(ocr_text: str) -> str:
#         """EasyOCR 결과 정제 함수"""
        
#         print(f"\n--- EasyOCR 원본 결과 (정제 전) ---")
#         print(ocr_text)
#         print("---------------------------------------")
        
#         # 1. 기본 정리
#         text = ocr_text.strip()
        
#         # 2. 여러 공백을 하나로
#         text = re.sub(r' +', ' ', text)
        
#         # 3. 불필요한 기호 제거 (EasyOCR 특성에 맞게)
#         text = re.sub(r'[^\w\s가-힣0-9a-zA-Z.,!?()[\]-]', ' ', text)
        
#         # 4. 여러 공백 다시 정리
#         text = re.sub(r' +', ' ', text)
        
#         # 5. 의미있는 단어들 추출
#         # 한글 단어 (1글자 이상)
#         korean_words = re.findall(r'[가-힣]+', text)
#         korean_words = [w for w in korean_words if len(w) >= 1]
        
#         # 숫자 (1자리 이상)
#         numbers = re.findall(r'\d+', text)
        
#         # 영어 단어 (2글자 이상)
#         english_words = re.findall(r'[a-zA-Z]+', text)
#         english_words = [w for w in english_words if len(w) >= 2]
        
#         # 특수 기호나 수식 (수학 문제의 경우)
#         math_symbols = re.findall(r'[+\-*/=().,]', text)
        
#         # 6. 모든 의미있는 요소들 결합
#         all_elements = korean_words + numbers + english_words
        
#         # 7. 중복 제거하되 순서 유지
#         seen = set()
#         unique_elements = []
#         for element in all_elements:
#             if element not in seen:
#                 seen.add(element)
#                 unique_elements.append(element)
        
#         refined_text = " ".join(unique_elements)
        
#         # 8. 수학 기호가 많으면 추가 (수학 문제일 가능성)
#         if len(math_symbols) > 5:
#             unique_symbols = list(set(math_symbols))
#             refined_text += " " + " ".join(unique_symbols)
        
#         return refined_text
    
#     final_text = refine_easyocr_text(best_result)
    
#     print(f"\n--- 최종 정제된 OCR 키워드 ---")
#     print(final_text)
#     print("---------------------------------------")
    
#     return final_text

# @router.post("/query", response_model=RAGResponse)
# # ★ 2. 파라미터 타입을 Union으로 변경합니다.
# async def query_rag_system(question: str = Form(...), image: Optional[UploadFile] = File(None)):
#     """
#     멀티모달 RAG 시스템에 질문과 이미지를 전달하여 답변을 요청합니다.
#     - **question**: 사용자의 질문 텍스트 (Form 데이터)
#     - **image**: 이미지 파일 (Form 데이터, 선택 사항)
#     """
#     from app.services.rag_service import get_rag_service

#     rag_service = get_rag_service()

#     print("라우터 /rag/query 요청 수신")
#     print(f" 질문: {question}")
    
#     # 수신된 파라미터의 실제 타입을 확인하여 디버깅에 활용할 수 있습니다.
#     print(f" 수신된 image 파라미터 타입: {type(image)}")

#     image_b64 = None
#     extracted_text = None

#     # ★ 3. isinstance를 사용하여 실제 파일(UploadFile)인지 명확하게 확인합니다.
#     if isinstance(image, UploadFile) and image.filename: # 이미지가 UploadFile 객체이고, 파일명이 비어있지 않은 경우
#         print(f" -이미지 파일명: {image.filename}")
#         # filename = image.filename if image.filename else "이름없음"
        
#         image_bytes = await image.read()

#         if not image_bytes:
#             print("  - 경고: 이미지 파일은 수신되었으나 내용이 비어있습니다.")
#         else:
#             image_b64 = base64.b64encode(image_bytes).decode("utf-8")
#             print(f"--- [ROUTER 진단] 이미지 인코딩 완료. Base64 길이: {len(image_b64)} ---")

#             print("--- [ROUTER] 이미지에서 EasyOCR 수행 중... ---")
#             try:
#                 extracted_text = await improved_ocr_processing(image_bytes)
#             except Exception as e:
#                 print(f"!!!!!! 에러: OCR 처리 중 오류 발생: {e} !!!!!!")
#                 extracted_text = ""

#     else:
#         # 이미지가 없거나(None), 빈 문자열("")로 들어온 경우 모두 이쪽으로 처리됩니다.
#         print(f" -이미지 파일이 없습니다. (수신된 값: '{image}')")

#     try:
#         print("라우터 rag_service.get_answer 호출 시작")

#         result_state = await rag_service.get_answer(
#             question=question,
#             image_b64=image_b64,
#             extracted_text=extracted_text
#         )

#         print("라우터 rag_service.get_answer 호출 완료")
#         generation_node_output = result_state.get('generate', {})
#         final_answer = generation_node_output.get('generation', '죄송합니다. 답변을 생성하지 못했습니다.')

#         return RAGResponse(answer=final_answer)

#     except Exception as e:
#         print(f"Error during RAG query: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail="Internal server error while processing the RAG query.")

















































# from fastapi import APIRouter, UploadFile, File, Form, HTTPException
# import base64
# from app.schemas import RAGRequest, RAGResponse
# from typing import Optional, Union

# import pytesseract
# from PIL import Image
# import io

# import cv2
# import numpy as np
# import re

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# router = APIRouter(
#     prefix="/rag",
#     tags=["RAG"]

    
# )

# async def improved_ocr_processing(image_bytes):
#     """개선된 OCR 전처리 함수"""
    
#     print("--- [ROUTER] 개선된 이미지 OCR 전처리 시작 ---")
    
#     # NumPy 배열로 변환
#     np_arr = np.frombuffer(image_bytes, np.uint8)
#     img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
#     # 원본 이미지 저장 (디버깅용)
#     cv2.imwrite("original_image.png", img_cv)
    
#     # 여러 전처리 방법을 시도해보는 함수들
#     def method_1_simple(img):
#         """방법 1: 단순 그레이스케일 + 확대"""
#         # 3배 확대 (더 큰 확대율)
#         height, width = img.shape[:2]
#         resized = cv2.resize(img, (width*3, height*3), interpolation=cv2.INTER_CUBIC)
        
#         # 그레이스케일
#         gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
#         return gray
    
#     def method_2_adaptive_threshold(img):
#         """방법 2: 적응적 임계값 처리"""
#         height, width = img.shape[:2]
#         resized = cv2.resize(img, (width*3, height*3), interpolation=cv2.INTER_CUBIC)
        
#         gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
#         # 가우시안 블러 (노이즈 제거)
#         blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
#         # 적응적 임계값 (THRESH_BINARY 사용 - INV 제거)
#         thresh = cv2.adaptiveThreshold(
#             blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#             cv2.THRESH_BINARY, 11, 2
#         )
        
#         return thresh
    
#     def method_3_otsu(img):
#         """방법 3: Otsu 이진화"""
#         height, width = img.shape[:2]
#         resized = cv2.resize(img, (width*3, height*3), interpolation=cv2.INTER_CUBIC)
        
#         gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
#         # 약간의 블러
#         blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
#         # Otsu 이진화
#         _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
#         return thresh
    
#     def method_4_morphology(img):
#         """방법 4: 모폴로지 연산 추가"""
#         height, width = img.shape[:2]
#         resized = cv2.resize(img, (width*3, height*3), interpolation=cv2.INTER_CUBIC)
        
#         gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
#         # Otsu 이진화
#         _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
#         # 모폴로지 연산 (노이즈 제거)
#         kernel = np.ones((2, 2), np.uint8)
#         cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
#         return cleaned
    
#     # --- [새로운 방법 5] Contour Detection 추가 ---
#     def method_5_contour_detection(img):
#         """방법 5: 윤곽선 검출로 텍스트 영역만 추출"""
#         height, width = img.shape[:2]
#         resized = cv2.resize(img, (width*3, height*3), interpolation=cv2.INTER_CUBIC)
#         gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
#         # Otsu 이진화로 글자와 배경 분리
#         _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
#         # 윤곽선 찾기
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         # 윤곽선을 감싸는 최소 사각형(Bounding Box)들의 리스트 생성
#         bounding_boxes = []
#         for contour in contours:
#             x, y, w, h = cv2.boundingRect(contour)
#             # 너무 작거나 너무 큰 윤곽선은 노이즈일 가능성이 높으므로 제외
#             # 이 값들은 이미지 특성에 맞게 튜닝이 필요할 수 있습니다.
#             if 10 < w < 300 and 10 < h < 100:
#                 bounding_boxes.append((x, y, w, h))

#         # 모든 텍스트 영역을 포함하는 새로운 캔버스(배경) 생성
#         # 원본 이미지와 동일한 크기의 흰색 배경을 만듭니다.
#         mask = np.full_like(gray, 255, dtype=np.uint8)
        
#         # 디버깅용: 윤곽선을 원본 이미지에 그려서 확인
#         img_with_contours = resized.copy()

#         # 찾은 텍스트 영역(Bounding Box)만 검은색 글씨로 캔버스에 복사
#         for x, y, w, h in bounding_boxes:
#             # 텍스트 영역을 원본(그레이스케일)에서 잘라냅니다.
#             roi = gray[y:y+h, x:x+w]
#             # 해당 영역을 새로운 캔버스에 붙여넣습니다.
#             mask[y:y+h, x:x+w] = roi
#             # 디버깅용으로 사각형 그리기
#             cv2.rectangle(img_with_contours, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
#         # 디버깅용 이미지 저장
#         cv2.imwrite("debug_contours.png", img_with_contours)
        
#         # 최종적으로 Otsu 이진화를 한번 더 적용하여 깔끔하게 만듭니다.
#         _, final_thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#         return final_thresh
    
#     # 각 방법별로 OCR 시도
#     methods = [
#         ("단순_확대", method_1_simple),
#         ("적응적_임계값", method_2_adaptive_threshold),
#         ("Otsu_이진화", method_3_otsu),
#         ("모폴로지_정리", method_4_morphology),
#         ("윤곽선_검출", method_5_contour_detection)
#     ]
    
#     best_result = ""
#     best_confidence = 0
    
#     for method_name, method_func in methods:
#         try:
#             print(f"--- 시도 중: {method_name} ---")
            
#             processed_img = method_func(img_cv)
            
#             # 전처리된 이미지 저장 (디버깅용)
#             cv2.imwrite(f"preprocessed_{method_name}.png", processed_img)
            
#             # 여러 PSM 모드 시도
#             psm_configs = [
#                 '--oem 3 --psm 6',  # 균일한 텍스트 블록
#                 '--oem 3 --psm 11', # 흩어진 텍스트
#                 '--oem 3 --psm 13', # 한 줄 텍스트
#                 '--oem 3 --psm 8',  # 단어로 취급
#             ]
            
#             for config in psm_configs:
#                 try:
#                     # OCR 실행
#                     result = pytesseract.image_to_string(processed_img, lang='kor', config=config)
                    
#                     # 신뢰도 정보도 가져오기
#                     data = pytesseract.image_to_data(processed_img, lang='kor', config=config, output_type=pytesseract.Output.DICT)
#                     confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
#                     avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
#                     print(f"    PSM {config.split('psm')[1].split()[0]}: 신뢰도 {avg_confidence:.1f}% | 길이: {len(result.strip())}")
                    
#                     if len(result.strip()) > len(best_result.strip()) and avg_confidence > 30:
#                         best_result = result
#                         best_confidence = avg_confidence
#                         print(f"    → 현재 최적 결과로 업데이트!")
                
#                 except Exception as e:
#                     print(f"    PSM 설정 오류: {e}")
#                     continue
        
#         except Exception as e:
#             print(f"{method_name} 방법 실패: {e}")
#             continue
    
#     print(f"\n--- 최종 OCR 결과 (신뢰도: {best_confidence:.1f}%) ---")
#     print(best_result)
#     print("--- OCR 완료 ---")
    
#     # return best_result


#     print(f"\n--- OCR 원본 결과 (정제 전) ---")
#     print(best_result)
#     print("--------------------------------")

#     # --- [새로운 단계] OCR 결과에서 유의미한 키워드만 추출 ---
#     def refine_ocr_text(ocr_text: str) -> str:
#         # 정규표현식을 사용하여 한글 단어만 추출합니다.
#         # [가-힣]+ 는 연속된 한글 문자 1개 이상을 의미합니다.
#         korean_words = re.findall(r'[가-힣]+', ocr_text)
        
#         # 의미 있을 가능성이 높은, 2글자 이상의 단어만 필터링합니다.
#         # 또는 모든 단어를 사용하려면 이 필터를 제거할 수 있습니다.
#         meaningful_words = [word for word in korean_words if len(word) >= 1]
        
#         # 중복을 제거하고 원래 순서를 유지합니다.
#         unique_words = list(dict.fromkeys(meaningful_words))
        
#         # 정제된 키워드들을 공백으로 구분하여 하나의 문자열로 합칩니다.
#         refined_text = " ".join(unique_words)
#         return refined_text

#     final_text = refine_ocr_text(best_result)
#     # --- 새로운 단계 끝 ---
    
#     print(f"\n--- 최종 정제된 OCR 키워드 ---")
#     print(final_text)
#     print("---------------------------------")
    
#     return final_text




# @router.post("/query", response_model=RAGResponse)
# async def query_rag_system(question: str = Form(...), image: Union[UploadFile, str] = File(None)):
#     """
#     멀티모달 RAG 시스템에 질문과 이미지를 전달하여 답변을 요청합니다.
#     - **question**: 사용자의 질문 텍스트 (Form 데이터)
#     - **image**: 이미지 파일 (Form 데이터)
#     """
#     from app.services.rag_service import get_rag_service

#     rag_service = get_rag_service()

#     print("라우터 /rag/query 요청 수신")
#     print(f" 질문: {question}")
#     print(f" 이미지 파일명: {image.filename if image else '없음'}")

#     image_b64 = None
#     extracted_text = None


#     if image and isinstance(image, UploadFile):

#         print(f" -이미지 파일명: {image.filename}")
#         image_bytes = await image.read()
#         image_b64 = base64.b64encode(image_bytes).decode("utf-8")
#         print(f"--- [ROUTER 진단] 이미지 인코딩 완료. Base64 길이: {len(image_b64)} ---")

#         print("--- [ROUTER] 이미지에서 Pytesseract OCR 수행 중... ---")
#         try:
#             extracted_text = await improved_ocr_processing(image_bytes)
            
#             print(f"--- [ROUTER] OCR 결과:\n{extracted_text}\n---")
          
#         except pytesseract.TesseractNotFoundError:
#             print("!!!!!! 에러: Tesseract를 찾을 수 없습니다. `tesseract_cmd` 경로가 올바른지 확인하세요. !!!!!!")
#             raise HTTPException(status_code=500, detail="Tesseract-OCR 엔진을 찾을 수 없습니다.")
#         except Exception as e:
#             print(f"!!!!!! 에러: OCR 처리 중 오류 발생: {e} !!!!!!")
#             # OCR 실패 시에도 에러를 내지 않고 계속 진행하도록 할 수 있습니다.
#             extracted_text = "" # 또는 None

#     else:
#         print(f" -이미지 파일명: 없습니다 (수신된 값: '{image}')")


#     try:
#         # [수정] 실제 rag_service 호출
#         print("라우터 rag_service.get_answer 호출 시작")

#         result_state = await rag_service.get_answer(
#             question=question,
#             image_b64=image_b64,
#             extracted_text=extracted_text
#         )

        
#         print("라우터 rag_service.get_answer 호출 완료")
#         final_answer = result_state.get('generate', {}).get('generation', '죄송합니다. 답변을 생성하지 못했습니다.')
#         return RAGResponse(answer=final_answer)

#     except Exception as e:
#         print(f"Error during RAG query: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail="Internal server error while processing the RAG query.")