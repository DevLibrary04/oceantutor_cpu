
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_STORAGE_PATH = os.path.join(PROJECT_ROOT, '.db_storage')
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache") # 임시 캐시 폴더
TEMP_UPLOAD_DIR = os.path.join(CACHE_DIR, "uploads") # 업로드 파일 임시 저장

DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data')
MARKDOWN_FILE_PATH = os.path.join(DATA_FOLDER, "final_final_report.md") # 이론 해설
# JSON_FILE_PATH = os.path.join()   # 문제 해설
TEXT_DB_PATH = os.path.join(DB_STORAGE_PATH, 'chroma_db_text_rag')

DATA_IMAGES_DIR = os.path.join(DATA_FOLDER, "images")
REFERENCE_IMAGES_DIR = os.path.join(DATA_IMAGES_DIR, "red_images") # 정답 교재 이미지 폴더
PROBLEM_IMAGES_DIR = os.path.join(DATA_IMAGES_DIR, "problem_images") # 마린소프트, 기출문제 이미지 폴더

YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolo_best.pt") # YOLO 모델 경로

TEXT_EMBEDDING_MODEL = "BAAI/bge-m3"
CLIP_MODEL = "openai/clip-vit-base-patch32"
RERANKER_MODEL = 'BAAI/bge-reranker-v2-m3'

LLM_MODEL = "gemini-2.5-flash" 

# CLIP_SIMILARITY_THRESHOLD = 0.8
PHASH_DISTANCE_THRESHOLD = 10       # 거리가 10 이하인 것 찾음

DEVICE = 'cpu'

RELEVANCE_THRESHOLD = 0.1
SIMILARITY_SEARCH_K = 10
RERANKER_TOP_K = 3
MAX_CONTEXT_DOCS = 3