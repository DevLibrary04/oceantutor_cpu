# app/rag/rag_pipeline.py (통합 검색어 생성 최종 아키텍처 버전)
import json
import os
import base64
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
import re
from langgraph.graph import StateGraph, END
from langchain.schema import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

# --- 서비스 및 템플릿 import ---
from app.services.image_matching_service import get_image_matching_service
from app.ocr_service import get_ocr_reader
from .prompt_templates import GradeDocuments, grade_prompt, rewrite_prompt, create_generate_prompt, create_final_query_prompt

# --- 1. State 정의: 데이터 가방 ---
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    web_search_needed: bool
    generation: str
    error: Optional[str]
    uploaded_image_b64: Optional[str]
    extracted_text: Optional[str]
    vqa_keyword: Optional[str]
    final_query: Optional[str] # ⭐ '최종 수사 질의어'를 담을 공간

# --- 2. Helper 함수 ---
def safe_parse_json(text: str) -> GradeDocuments:
    # (기존과 동일)
    try:
        return GradeDocuments(**json.loads(text))
    except (json.JSONDecodeError, TypeError):
        text_lower = text.lower()
        if 'yes' in text_lower:
            return GradeDocuments(binary_score='yes')
        return GradeDocuments(binary_score='no')

# --- 3. Graph(파이프라인) 빌드 함수 ---
def build_rag_app(llm, reranker, text_vectorstore, image_vectorstore, web_search_tool, cfg):
    # --- 3-1. 노드들의 개인 도구 준비 ---
    retrieval_grader = grade_prompt | llm | (lambda ai_msg: safe_parse_json(ai_msg.content))
    question_rewriter = rewrite_prompt | llm | StrOutputParser()
    final_query_generator = create_final_query_prompt | llm | StrOutputParser()

    # --- 3-2. 각 노드(직원)의 업무 내용 정의 ---

    # 직원 1: OCR 전문가
    def perform_ocr(state):
        print("\n--- [노드] OCR 수행 시작 ---")
        image_b64 = state.get("uploaded_image_b64")
        if not image_b64: return {"extracted_text": None}
        try:
            ocr_reader = get_ocr_reader()
            image_bytes = base64.b64decode(image_b64)
            import numpy as np
            import cv2
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            results = ocr_reader.readtext(img_np)
            text = " ".join([res[1] for res in results])
            print(f"  -> OCR 텍스트 추출 성공: {text[:100]}...")
            return {"extracted_text": text}
        except Exception as e:
            print(f"  -> OCR 처리 중 오류: {e}")
            return {"extracted_text": None}

    # 직원 2: 이미지 분석가 (Visual Q&A)
    def perform_visual_qa(state):
        print("\n--- [노드] Visual Q&A (YOLO, CLIP 등) 시작 ---")
        image_b64 = state.get("uploaded_image_b64")
        if not image_b64: return {"vqa_keyword": None}
        try:
            service = get_image_matching_service()
            image_bytes = base64.b64decode(image_b64)
            keyword = service.find_keyword_from_image(image_bytes)
            if "실패" in keyword:
                print(f"  -> VQA 실패: {keyword}")
                return {"vqa_keyword": None}
            else:
                print(f"  -> VQA 성공! 추출된 키워드: '{keyword}'")
                return {"vqa_keyword": keyword}
        except Exception as e:
            print(f"  -> VQA 처리 중 오류: {e}")
            return {"vqa_keyword": None}
            
    # ⭐⭐⭐ 새로운 핵심 직원: 수사반장 (최종 질의어 생성) ⭐⭐⭐
    def create_final_query(state):
        print("\n--- [노드] 최종 검색어 생성 시작 ---")
        # 모든 증거를 수집
        question = state["question"]
        ocr_text = state.get("extracted_text")
        vqa_keyword = state.get("vqa_keyword")
        
        # LLM을 사용하여 최적의 검색어를 생성
        final_query = final_query_generator.invoke({
            "user_question": question,
            "ocr_text": ocr_text,
            "vqa_keyword": vqa_keyword
        })
        print(f"  -> 생성된 최종 검색어: '{final_query}'")
        return {"final_query": final_query}

    # 직원 4: 텍스트 자료조사원 (이제 'final_query'만 사용)
    def retrieve_from_text(state):
        print("\n--- [노드] 텍스트 DB 검색 시작 ---")
        search_query = state.get("final_query") or state["question"] # final_query가 없으면 원래 질문 사용
        documents = text_vectorstore.similarity_search(search_query, k=cfg['SIMILARITY_SEARCH_K'])
        print(f"  -> '{search_query[:50]}...'(으)로 {len(documents)}개 문서 찾음.")
        return {"documents": documents}

    # (이후 grade_documents, web_search, generate 노드는 기존과 동일하므로 생략 가능)
    # ... grade_documents, web_search, generate 함수 정의 ...
    def grade_documents(state):
        print("\n--- [노드] 문서 품질 검수 시작 ---")
        question = state.get("final_query") or state["question"]
        documents = state.get("documents", [])
        
        if not documents:
            print("  -> 검수할 문서가 없어 웹 검색 필요.")
            return {"web_search_needed": True, "documents": []}
            
        print(f"  -> '{question[:50]}...'와 관련하여 {len(documents)}개 문서 검수 중...")

        # 1. Reranker로 점수 계산 및 정렬 (이 부분은 매우 잘 작동합니다)
        pairs = [(question, doc.page_content) for doc in documents]
        scores = reranker.predict(pairs)
        docs_with_scores = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        
        # 2. Reranker 점수가 기준치 이상인 문서만 필터링
        filtered_docs = []
        print("  -> Reranker 점수 기반 필터링:")
        for score, doc in docs_with_scores[:cfg["RERANKER_TOP_K"]]:
            print(f"    - 문서: '{doc.page_content[:30]}...', 점수: {score:.4f}")
            if score >= cfg["RELEVANCE_THRESHOLD"]:
                filtered_docs.append(doc)
                print("      -> (통과)")
            else:
                print("      -> (탈락)")
                
        # ⭐⭐⭐ LLM의 'yes/no' 평가는 일단 건너뜁니다! ⭐⭐⭐
        # grade = retrieval_grader.invoke(...)
        # if grade.binary_score == 'yes':
        #     ...
        
        print(f"  -> 최종 통과된 문서 (Reranker 기준): {len(filtered_docs)}개")
        return {"documents": filtered_docs, "web_search_needed": not bool(filtered_docs)}
    
    def refine_context_by_keyword(state):
            """
            [업그레이드된 정제 전문가 v2.0]
            VQA 키워드가 포함된 '전체 단락'을 추출하여 풍부한 문맥을 제공합니다.
            """
            print("\n--- [노드] VQA 키워드로 핵심 문맥 정제 시작 ---")
            vqa_keyword = state.get("vqa_keyword")
            documents = state.get("documents", [])
            
            if not vqa_keyword or not documents:
                print("  -> VQA 키워드/문서가 없어 정제를 건너뜁니다.")
                return {"documents": documents}
                
            refined_docs = []
            for doc in documents:
                # 문서 내용을 빈 줄(\n\n)을 기준으로 단락으로 나눕니다.
                # Markdown 테이블 같은 경우도 하나의 단락으로 처리됩니다.
                paragraphs = re.split(r'\n\s*\n', doc.page_content)
                
                relevant_paragraphs = []
                for p in paragraphs:
                    # 현재 단락에 VQA 키워드가 포함되어 있는지 확인
                    if vqa_keyword in p:
                        relevant_paragraphs.append(p)
                
                if relevant_paragraphs:
                    # 키워드가 포함된 모든 단락을 합쳐서 새로운 문서를 만듭니다.
                    new_content = "\n\n".join(relevant_paragraphs)
                    refined_docs.append(Document(page_content=new_content, metadata=doc.metadata))
                    print(f"  -> 문서 정제 완료. 키워드 '{vqa_keyword}'가 포함된 단락을 추출했습니다.")
                    print(f"     - 정제 후 내용(일부): {new_content[:150]}...")
                else:
                    # 만약 키워드가 포함된 단락이 하나도 없다면, 원본 문서를 그대로 사용합니다 (안전장치).
                    print(f"  -> 경고: 문서에서 '{vqa_keyword}' 키워드가 포함된 단락을 찾지 못했습니다. 원본 문서를 사용합니다.")
                    refined_docs.append(doc)

            return {"documents": refined_docs}

    def web_search(state):
        print("\n--- [노드] 웹 검색 시작 ---")
        question = state["final_query"] or state["question"]
        rewritten_question = question_rewriter.invoke({"question": question})
        web_results = web_search_tool.invoke({"query": rewritten_question})
        web_docs = [Document(page_content=d["content"], metadata={"source": d["url"]}) for d in web_results]
        return {"documents": web_docs}

    def generate(state):
        print("\n--- [노드] 최종 답변 생성 시작 ---")
        question = state["question"] # 답변 생성 시에는 사용자의 원본 질문을 사용
        documents = state.get("documents", [])
        uploaded_image_b64 = state.get("uploaded_image_b64")
        extracted_text = state.get("extracted_text")
        prompt_text = create_generate_prompt(question, documents, bool(uploaded_image_b64), extracted_text)
        message_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        if uploaded_image_b64:
            message_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{uploaded_image_b64}"}})
        response = llm.invoke([HumanMessage(content=message_content)])
        print("--- LLM 최종 답변 : ---" + "\n" + response.content)
        return {"generation": response.content}

    # --- 3-3. 교통정리 담당(라우터)의 판단 로직 정의 ---
    def route_after_grading(state):
        print("\n--- [라우터] DB 검색 결과에 따른 경로 결정 ---")
        if state.get("web_search_needed"):
            return "needs_web_search"
        else:
            return "has_documents"

    # --- 3-4. 워크플로우(회사 조직도) 구성 ---
    workflow = StateGraph(GraphState)
    
    # 1. 직원(노드)들을 회사에 등록
    workflow.add_node("perform_ocr", perform_ocr)
    workflow.add_node("perform_visual_qa", perform_visual_qa)
    workflow.add_node("create_final_query", create_final_query)
    workflow.add_node("retrieve_from_text", retrieve_from_text)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("refine_context_by_keyword", refine_context_by_keyword)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    
    # 2. 업무 시작점은 텍스트냐 이미지냐에 따라 분기
    def route_initial_request(state):
        if state.get("uploaded_image_b64"):
            return "perform_ocr"
        else:
            return "create_final_query"

    workflow.set_conditional_entry_point(route_initial_request, {
        "perform_ocr": "perform_ocr",
        "create_final_query": "create_final_query"
    })
    
    # 3. 업무 흐름(배선) 연결
    workflow.add_edge("perform_ocr", "perform_visual_qa")
    workflow.add_edge("perform_visual_qa", "create_final_query")
    workflow.add_edge("create_final_query", "retrieve_from_text")
    workflow.add_edge("retrieve_from_text", "grade_documents")
    
    workflow.add_conditional_edges("grade_documents", route_after_grading, {
        "needs_web_search": "web_search",
        "has_documents": "refine_context_by_keyword"
    })
    workflow.add_edge("refine_context_by_keyword", "generate")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    # 4. 최종 조직도(앱) 완성!
    return workflow.compile()