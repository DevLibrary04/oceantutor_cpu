import os
from dotenv import load_dotenv

load_dotenv()

# 프로젝트 루트를 기준으로 경로 설정
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_STORAGE_PATH = os.path.join(PROJECT_ROOT, '.db_storage')

# 이 부분은 실제 데이터 위치에 맞게 수정해야 합니다.
# 예: DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
DATA_FOLDER = r'C:\Users\user\Desktop\tutor\bonproj_backend\data'
MARKDOWN_FILE_PATH = os.path.join(DATA_FOLDER, "final_final_report.md")

# ChromaDB 저장 경로
TEXT_DB_PATH = os.path.join(DB_STORAGE_PATH, 'chroma_db_text_rag_v1')
IMAGE_DB_PATH = os.path.join(DB_STORAGE_PATH, 'chroma_db_image_rag_v1')

# --- 모델 설정 ---
DEVICE = 'cuda'  # 'cuda' 또는 'cpu'
TEXT_EMBEDDING_MODEL = "BAAI/bge-m3"
IMAGE_EMBEDDING_MODEL = 'sentence-transformers/clip-ViT-B-32'
RERANKER_MODEL = 'BAAI/bge-reranker-v2-m3'
LLM_MODEL = "gemma3:12b" # Ollama에서 사용하는 모델 이름

# --- RAG 파이프라인 파라미터 ---
RELEVANCE_THRESHOLD = 0.2
SIMILARITY_SEARCH_K = 10
RERANKER_TOP_K = 3
MAX_CONTEXT_DOCS = 3