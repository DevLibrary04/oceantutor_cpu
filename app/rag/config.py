# app/rag/config.py (수정된 최종본)

import os
from dotenv import load_dotenv

load_dotenv()

# --- 기본 경로 설정 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_STORAGE_PATH = os.path.join(PROJECT_ROOT, '.db_storage')
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache") # 임시 캐시 폴더
TEMP_UPLOAD_DIR = os.path.join(CACHE_DIR, "uploads") # 업로드 파일 임시 저장

# --- 텍스트 RAG 데이터 경로 ---
DATA_FOLDER = r'C:\Users\user\Desktop\tutor\bonproj_backend\data'
MARKDOWN_FILE_PATH = os.path.join(DATA_FOLDER, "final_final_report.md")
TEXT_DB_PATH = os.path.join(DB_STORAGE_PATH, 'chroma_db_text_rag_v2')

# --- 이미지 매칭 RAG (신규) 경로 설정 ---
# ❗️❗️ 아래 경로는 실제 파일 위치에 맞게 수정해야 합니다. ❗️❗️
REFERENCE_IMAGES_DIR = os.path.join(PROJECT_ROOT, "reference_images") # 정답 이미지 폴더
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolo_best.pt") # YOLO 모델 경로

# --- 텍스트 RAG 모델 설정 ---
TEXT_EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = 'BAAI/bge-reranker-v2-m3'
LLM_MODEL = "gemma3:12b" # Ollama에서 사용하는 모델 이름

# --- 공통 설정 ---
DEVICE = 'cuda'

# --- 텍스트 RAG 파이프라인 파라미터 ---
RELEVANCE_THRESHOLD = 0.1
SIMILARITY_SEARCH_K = 10
RERANKER_TOP_K = 3
MAX_CONTEXT_DOCS = 3