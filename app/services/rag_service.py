import os
import shutil
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_community.chat_models import ChatOllama
from langchain_teddynote.tools.tavily import TavilySearch

# 상대 경로 임포트
from app.rag import config
from app.rag.loader import load_markdown_documents
from app.rag.rag_pipeline import build_rag_app

class RAGService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("--- RAG Service 초기화 시작 ---")
            cls._instance = super(RAGService, cls).__new__(cls)
            cls._instance._initialize_system()
            print("--- RAG Service 초기화 완료 ---")
        return cls._instance

    def _initialize_system(self):
        """실제 초기화 로직. 모델 로딩, DB 생성 등."""
        load_dotenv()
        
        os.makedirs(config.DB_STORAGE_PATH, exist_ok=True)
        if os.path.exists(config.TEXT_DB_PATH): shutil.rmtree(config.TEXT_DB_PATH)
        if os.path.exists(config.IMAGE_DB_PATH): shutil.rmtree(config.IMAGE_DB_PATH)

        text_docs, image_docs = load_markdown_documents(config.MARKDOWN_FILE_PATH)

        text_embedding = HuggingFaceEmbeddings(
            model_name=config.TEXT_EMBEDDING_MODEL, model_kwargs={'device': config.DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
        image_embedding = HuggingFaceEmbeddings(
            model_name=config.IMAGE_EMBEDDING_MODEL, model_kwargs={'device': config.DEVICE}
        )

        reranker = CrossEncoder(config.RERANKER_MODEL, max_length=512, device=config.DEVICE)
        llm = ChatOllama(model=config.LLM_MODEL, temperature=0)
        web_search_tool = TavilySearch(max_results=3)

        text_vectorstore = Chroma.from_documents(
            documents=text_docs, embedding=text_embedding,
            persist_directory=config.TEXT_DB_PATH, ids=[d.metadata['id'] for d in text_docs]
        )
        image_vectorstore = None
        if image_docs:
            image_vectorstore = Chroma.from_documents(
                documents=image_docs, embedding=image_embedding,
                persist_directory=config.IMAGE_DB_PATH, ids=[d.metadata['id'] for d in image_docs]
            )

        rag_config_params = {
            "RELEVANCE_THRESHOLD": config.RELEVANCE_THRESHOLD,
            "SIMILARITY_SEARCH_K": config.SIMILARITY_SEARCH_K,
            "RERANKER_TOP_K": config.RERANKER_TOP_K,
            "MAX_CONTEXT_DOCS": config.MAX_CONTEXT_DOCS
        }
        
        self.rag_app = build_rag_app(llm, reranker, text_vectorstore, image_vectorstore, web_search_tool, rag_config_params)

    async def get_answer(self, question: str, image_b64: str | None) -> dict:
        """RAG 파이프라인을 비동기적으로 실행하고 최종 결과를 반환합니다."""
        inputs = {"question": question, "uploaded_image_b64": image_b64}
        final_state = {}
        async for output in self.rag_app.astream(inputs):
            for key, value in output.items():
                print(f"Node '{key}' finished.")
            final_state = output
        return final_state

rag_service = RAGService()