# app/ocr_service.py

import easyocr
import logging

logger = logging.getLogger(__name__)

# OCR Reader 인스턴스를 저장할 전역 변수 (싱글톤 패턴)
_ocr_reader_instance = None

def get_ocr_reader():
    """
    EasyOCR Reader의 싱글톤 인스턴스를 반환합니다.
    인스턴스가 없으면 새로 생성하고, 있으면 기존 것을 반환합니다.
    이 방식을 "게으른 초기화(Lazy Initialization)"라고 하며,
    서버 시작 시점이 아닌 실제 필요 시점에 리소스를 로딩하여 안정성을 높입니다.
    """
    global _ocr_reader_instance
    
    # 인스턴스가 아직 생성되지 않았을 때만 초기화 로직 실행
    if _ocr_reader_instance is None:
        logger.info("--- [OCR Service] EasyOCR Reader를 처음으로 초기화합니다 (시간이 걸릴 수 있습니다)... ---")
        try:
            # GPU를 사용하여 EasyOCR Reader 초기화 시도
            _ocr_reader_instance = easyocr.Reader(['ko', 'en'], gpu=True)
            logger.info("--- [OCR Service] GPU 모드로 EasyOCR Reader 초기화 완료. ---")
        except Exception as e:
            logger.error(f"GPU로 EasyOCR 초기화 실패: {e}")
            # GPU 사용 실패 시 CPU로 재시도
            logger.info("--- [OCR Service] CPU 모드로 재시도합니다... ---")
            _ocr_reader_instance = easyocr.Reader(['ko', 'en'], gpu=False)
            logger.info("--- [OCR Service] CPU 모드로 EasyOCR Reader 초기화 완료. ---")
            
    return _ocr_reader_instance