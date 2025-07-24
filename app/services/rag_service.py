# app/services/rag_service.py

import os
import shutil
import logging
import time
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama
from langchain_teddynote.tools.tavily import TavilySearch

from fastapi.concurrency import run_in_threadpool
from typing import Dict, Any, Optional, List

# 절대 경로 임포트로 변경하여 안정성 확보
from app.rag import config
from app.rag.loader import load_markdown_documents
from app.rag.rag_pipeline import build_rag_app

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 싱글톤 인스턴스를 저장할 변수
_rag_service_instance = None

class RAGService:
    _initialized = False

    def __init__(self):
        self.rag_app = None
        self.text_embedding = None
        self.image_embedding = None
        logger.info("RAGService 인스턴스 생성됨 (아직 초기화 전)")

    def _load_embedding_model_safe(self, model_name: str, device: str, model_type: str = "text") -> Optional[HuggingFaceEmbeddings]:
        """안전한 임베딩 모델 로딩"""
        try:
            start_time = time.time()
            logger.info(f"{model_type} 임베딩 모델 로딩 시작: {model_name}")
            
            # HuggingFace 캐시 디렉토리 설정
            os.environ['HF_HOME'] = './hf_cache'
            os.environ['TRANSFORMERS_CACHE'] = './transformers_cache'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            
            model_kwargs = {
                'device': device,
                'trust_remote_code': True
            }
            
            encode_kwargs = {'normalize_embeddings': True} if model_type == "text" else {}
            
            embedding = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            # 간단한 테스트로 모델이 제대로 로딩되었는지 확인
            test_result = embedding.embed_query("test")
            if not test_result or len(test_result) == 0:
                raise ValueError("임베딩 결과가 비어있습니다.")
            
            load_time = time.time() - start_time
            logger.info(f"{model_type} 임베딩 모델 로딩 완료 ({load_time:.2f}초)")
            logger.info(f"임베딩 차원: {len(test_result)}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"{model_type} 임베딩 모델 로딩 실패: {str(e)}")
            logger.error(f"모델명: {model_name}, 디바이스: {device}")
            return None

    def initialize(self):
        if self._initialized:
            return
        
        logger.info("--- RAG Service 초기화 시작 (모델 및 데이터 로딩 중...) ---")
        load_dotenv()
        
        try:
            # --- [수정 1] 모델 로딩을 먼저 수행합니다 ---
            # DB 생성에 필요한 임베딩 모델을 먼저 준비합니다.
            logger.info(f"텍스트 임베딩 모델 로딩 시작: {config.TEXT_EMBEDDING_MODEL}")
            self.text_embedding = self._load_embedding_model_safe(
                config.TEXT_EMBEDDING_MODEL,
                config.DEVICE,
                "text"
            )
            if not self.text_embedding:
                raise RuntimeError("텍스트 임베딩 모델 로딩에 실패하였습니다.")

            # --- [수정 2] DB 존재 여부를 확인하고, 없으면 생성, 있으면 로드합니다 ---
            text_vectorstore = None
            if not os.path.exists(config.TEXT_DB_PATH):
                logger.info("기존 텍스트 벡터스토어가 존재하지 않습니다. 새로 생성합니다...")
                
                # 1. DB가 없을 때만 디렉토리 생성 및 문서 로딩 수행
                os.makedirs(config.DB_STORAGE_PATH, exist_ok=True)
                
                start_time = time.time()
                logger.info("문서 로딩 시작...")
                text_docs, _ = load_markdown_documents(config.MARKDOWN_FILE_PATH)
                logger.info(f"문서 로딩 완료 ({time.time() - start_time:.2f}초)")
                logger.info(f"텍스트 문서: {len(text_docs)}개")

                # 2. 벡터스토어 생성 (이 부분은 처음 한 번만 실행됩니다!)
                start_time = time.time()
                logger.info("텍스트 벡터스토어 생성 및 저장 중... (시간이 오래 걸릴 수 있습니다)")
                text_vectorstore = Chroma.from_documents(
                    documents=text_docs, 
                    embedding=self.text_embedding,
                    persist_directory=config.TEXT_DB_PATH, 
                    ids=[d.metadata['id'] for d in text_docs]
                )
                logger.info(f"텍스트 벡터스토어 생성 및 저장 완료 ({time.time() - start_time:.2f}초)")
                
            else:
                # 3. DB가 이미 존재하면, 그냥 불러오기만 합니다. (매우 빠름!)
                start_time = time.time()
                logger.info(f"기존 텍스트 벡터스토어를 로드합니다: {config.TEXT_DB_PATH}")
                text_vectorstore = Chroma(
                    persist_directory=config.TEXT_DB_PATH,
                    embedding_function=self.text_embedding
                )
                logger.info(f"텍스트 벡터스토어 로드 완료 ({time.time() - start_time:.2f}초)")
            
            # 이미지 벡터스토어는 이제 사용하지 않으므로 None으로 고정
            image_vectorstore = None



            #     shutil.rmtree(config.TEXT_DB_PATH)
            # if os.path.exists(config.IMAGE_DB_PATH): 
            #     shutil.rmtree(config.IMAGE_DB_PATH)
            # logger.info(f"디렉토리 설정 완료 ({time.time() - start_time:.2f}초)")

            # 2. 문서 로딩
            # start_time = time.time()
            # logger.info("문서 로딩 시작...")
            # text_docs, image_docs = load_markdown_documents(config.MARKDOWN_FILE_PATH)
            # logger.info(f"문서 로딩 완료 ({time.time() - start_time:.2f}초)")
            
            # 3. 텍스트 임베딩 모델 로딩
            # start_time = time.time()
            # logger.info(f"텍스트 임베딩 모델 로딩 시작: {config.TEXT_EMBEDDING_MODEL}")
        
            # self.text_embedding = self._load_embedding_model_safe(
            #         config.TEXT_EMBEDDING_MODEL,
            #         config.DEVICE,
            #         "text"
            # )
            # if not self.text_embedding:
            #     raise RuntimeError("텍스트 임베딩 모델 로딩에 실패하였습니다.")
                # text_embedding = HuggingFaceEmbeddings(
                #     model_name=config.TEXT_EMBEDDING_MODEL, 
                #     model_kwargs={
                #         'device': config.DEVICE,
                #         'trust_remote_code': True  # 필요한 경우
                #     },
                #     encode_kwargs={'normalize_embeddings': True}
                # )
                # logger.info(f"텍스트 임베딩 모델 로딩 완료 ({time.time() - start_time:.2f}초)")


            # 4. 이미지 임베딩 모델 로딩 (필요한 경우)
            # image_embedding = None
            # if hasattr(config, 'IMAGE_EMBEDDING_MODEL') and config.IMAGE_EMBEDDING_MODEL:
            #     self.image_embedding = self._load_embedding_model_safe(
            #         config.IMAGE_EMBEDDING_MODEL,
            #         config.DEVICE,
            #         "image"
            #     )
            #     if not self.image_embedding:
            #         raise RuntimeError("이미지 임베딩 모델 로딩에 실패하였습니다. 텍스트만으로 진행합니다.")
            
            # 5. Reranker 모델 로딩
            start_time = time.time()
            logger.info(f"Reranker 모델 로딩 시작: {config.RERANKER_MODEL}")
            try:
                reranker = CrossEncoder(config.RERANKER_MODEL, max_length=512, device=config.DEVICE)
                logger.info(f"Reranker 모델 로딩 완료 ({time.time() - start_time:.2f}초)")
            except Exception as e:
                logger.error(f"Reranker 모델 로딩 실패: {str(e)}")
                raise
            
            # 6. LLM 및 웹 검색 도구 초기화
            start_time = time.time()
            logger.info("LLM 및 웹 검색 도구 초기화 중...")
            llm = ChatOllama(model=config.LLM_MODEL, temperature=0)
            web_search_tool = TavilySearch(max_results=3)
            logger.info(f"LLM 및 웹 검색 도구 초기화 완료 ({time.time() - start_time:.2f}초)")
            
            # 7. 벡터스토어 생성
            # start_time = time.time()
            # logger.info("텍스트 벡터스토어 생성 중...")
            # text_vectorstore = Chroma.from_documents(
            #     documents=text_docs, 
            #     embedding=self.text_embedding,
            #     persist_directory=config.TEXT_DB_PATH, 
            #     ids=[d.metadata['id'] for d in text_docs]
            # )
            # logger.info(f"텍스트 벡터스토어 생성 완료 ({time.time() - start_time:.2f}초)")
            
            # image_vectorstore = None
            # if image_docs and self.image_embedding:
            #     start_time = time.time()
            #     logger.info("이미지 벡터스토어 생성 중...")
            #     image_vectorstore = Chroma.from_documents(
            #         documents=image_docs, 
            #         embedding=self.image_embedding,
            #         persist_directory=config.IMAGE_DB_PATH, 
            #         ids=[d.metadata['id'] for d in image_docs]
            #     )
            #     logger.info(f"이미지 벡터스토어 생성 완료 ({time.time() - start_time:.2f}초)")

            # 8. RAG 앱 빌드
            start_time = time.time()
            logger.info("RAG 파이프라인 빌드 중...")
            rag_config_params = {
                "RELEVANCE_THRESHOLD": config.RELEVANCE_THRESHOLD,
                "SIMILARITY_SEARCH_K": config.SIMILARITY_SEARCH_K,
                "RERANKER_TOP_K": config.RERANKER_TOP_K,
                "MAX_CONTEXT_DOCS": config.MAX_CONTEXT_DOCS
            }
            
            self.rag_app = build_rag_app(
                llm, reranker, text_vectorstore, image_vectorstore, 
                web_search_tool, rag_config_params
            )
            logger.info(f"RAG 파이프라인 빌드 완료 ({time.time() - start_time:.2f}초)")
            
            self._initialized = True
            logger.info("--- RAG Service 초기화 완료 ---")
            
        except Exception as e:
            logger.error(f"RAG Service 초기화 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            self._initialized = False
            raise

    async def get_answer(self, question: str, image_b64: Optional[str], extracted_text: Optional[str]) -> Dict[str, Any]:
        if not self._initialized or self.rag_app is None:
            raise RuntimeError("RAG Service is not initialized. Check server startup logs.")
        
        print(f"--- [SERVICE 진단] get_answer 호출됨. image_b64 길이: {len(image_b64) if image_b64 else 0} ---")

        def run_sync_pipeline() -> Dict[str, Any]:
            inputs = {
                "question": question, 
                "uploaded_image_b64": image_b64,
                "extracted_text": extracted_text
            }
            image_data = inputs.get('uploaded_image_b64')
            image_length = len(image_data) if image_data is not None else 0
            print(f"--- [SERVICE 진단] 파이프라인으로 전달될 초기 상태(inputs)의 이미지 길이: {image_length} ---")
            
            final_state = {}
            for output in self.rag_app.stream(inputs): 
                for key, value in output.items():
                    logger.info(f"--- [RAG Service] Node '{key}' finished. ---")
                final_state = output
            return final_state

        logger.info("--- [RAG Service] Starting RAG pipeline in a separate thread... ---")
        result = await run_in_threadpool(run_sync_pipeline)
        logger.info("--- [RAG Service] RAG pipeline finished. ---")
        return result

def get_rag_service() -> RAGService:
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance