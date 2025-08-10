import easyocr
import logging

logger = logging.getLogger(__name__)

# OCR Reader 인스턴스를 저장할 전역 변수 (싱글톤 패턴)
_ocr_reader_instance = None

def get_ocr_reader():
    global _ocr_reader_instance
    
    if _ocr_reader_instance is None:
        logger.info("--- [OCR Service] EasyOCR Reader를 처음으로 초기화합니다 (시간이 걸릴 수 있습니다)... ---")
        try:
            _ocr_reader_instance = easyocr.Reader(['ko', 'en'], gpu=True)
            logger.info("--- [OCR Service] GPU 모드로 EasyOCR Reader 초기화 완료. ---")
        except Exception as e:
            logger.error(f"GPU로 EasyOCR 초기화 실패: {e}")
            # GPU 사용 실패 시 CPU로 재시도
            logger.info("--- [OCR Service] CPU 모드로 재시도합니다... ---")
            _ocr_reader_instance = easyocr.Reader(['ko', 'en'], gpu=False)
            logger.info("--- [OCR Service] CPU 모드로 EasyOCR Reader 초기화 완료. ---")
            
    return _ocr_reader_instance