# # app/rag/rag_pipeline.py (통합 검색어 생성 최종 아키텍처 버전)
# import json
# import os
# import base64
# from io import BytesIO
# from PIL import Image
# from typing import List, Dict, Any, Optional
# from typing_extensions import TypedDict
# import re
# from langgraph.graph import StateGraph, END
# from langchain.schema import Document
# from langchain_core.messages import HumanMessage
# from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

# from app.services.image_matching_service import get_image_matching_service
# from app.ocr_service import get_ocr_reader
# from .prompt_templates import (
#     GradeDocuments, grade_prompt, rewrite_prompt, create_final_query_prompt, 
#     create_multimodal_prompt
# )

# class GraphState(TypedDict):
#     question: str
#     documents: List[Document]
#     web_search_needed: bool
#     generation: str
#     uploaded_image_b64: Optional[str]
#     matched_reference_image_b64: Optional[str]
#     vqa_keyword: Optional[str]
#     final_query: Optional[str] 

# # Helper 함수
# def safe_parse_json(text: str) -> GradeDocuments:
#     try:
#         return GradeDocuments(**json.loads(text))
#     except (json.JSONDecodeError, TypeError):
#         text_lower = text.lower()
#         if 'yes' in text_lower:
#             return GradeDocuments(binary_score='yes')
#         return GradeDocuments(binary_score='no')

# # Graph(파이프라인) 빌드 함수
# def build_rag_app(llm, reranker, text_vectorstore, image_vectorstore, web_search_tool, cfg):
#     # --- 3-1. 노드들의 개인 도구 준비 ---
#     retrieval_grader = grade_prompt | llm | (lambda ai_msg: safe_parse_json(ai_msg.content))
#     question_rewriter = rewrite_prompt | llm | StrOutputParser()
#     final_query_generator = create_final_query_prompt | llm | StrOutputParser()

#     # 직원 1: OCR 전문가
#     def perform_ocr(state):
#         print("\n--- [노드] OCR 수행 시작 ---")
#         image_b64 = state.get("uploaded_image_b64")
#         if not image_b64: return {"extracted_text": None}
#         try:
#             ocr_reader = get_ocr_reader()
#             image_bytes = base64.b64decode(image_b64)
#             import numpy as np
#             import cv2
#             nparr = np.frombuffer(image_bytes, np.uint8)
#             img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#             results = ocr_reader.readtext(img_np)
#             text = " ".join([res[1] for res in results])
#             print(f"  -> OCR 텍스트 추출 성공: {text[:100]}...")
#             return {"extracted_text": text}
#         except Exception as e:
#             print(f"  -> OCR 처리 중 오류: {e}")
#             return {"extracted_text": None}

#     # 직원 2: 이미지 분석가 (Visual Q&A)
#     def perform_visual_qa(state):
#         print("\n--- [노드] 멀티모달 VQA 시작 ---")
#         user_image_b64 = state.get("uploaded_image_b64")
#         if not user_image_b64: 
#             return {"vqa_keyword": None, "matched_reference_image_b64": None}
        
#         try:
#             service = get_image_matching_service()
#             image_bytes = base64.b64decode(user_image_b64)
#             ref_image_b64 = service.match_reference_image(image_bytes)
            
#             if not ref_image_b64:
#                 print("VQA 실패 : 매칭되는 정답 이미지를 찾지 못했습니다.")
#                 return {"vqa_keyword": None, "matched_reference_image_b64": None}
            
#             print("정답이미지 매칭 성공! 이제 Gemini에게 키워드 추출을 요청합니다.")
#             user_img = Image.open(BytesIO(image_bytes))
#             ref_img = Image.open(BytesIO(base64.b64decode(ref_image_b64)))
            
#             match = re.search
            
            
            
            
            
            
#             # result_dict = service.find_keyword_and_reference_image(image_bytes)
            
#             # keyword = result_dict.get("keyword")
#             # ref_image_b64 = result_dict.get("reference_image_b64")
             
#             # if "실패" in (keyword or ""):
#             #     print(f"  -> VQA 실패: {keyword}")
#             #     return {"vqa_keyword": None, "matched_reference_image_b64": None}
#             # else:
#             #     print(f"  -> VQA 성공! 추출된 키워드: '{keyword}'")
#             #     if ref_image_b64:
#             #         print("참조 이미지도 성공적으로 가져왔습니다. . .")
#             #     return {"vqa_keyword": keyword, "matched_reference_iamge_b64": ref_image_b64}
            
#         except Exception as e:
#             print(f"  -> VQA 처리 중 오류: {e}")
#             return {"vqa_keyword": None, "matched_reference_image_b64": None}
        
#     # 최종 질의어 생성        
#     def create_final_query(state):
#         print("\n--- [노드] 최종 검색어 생성 시작 ---")

#         question = state["question"]
#         ocr_text = state.get("extracted_text")
#         vqa_keyword = state.get("vqa_keyword")
        
#         # LLM을 사용하여 최적의 검색어를 생성
#         final_query = final_query_generator.invoke({
#             "user_question": question,
#             "ocr_text": ocr_text,
#             "vqa_keyword": vqa_keyword
#         })
        
#         print(f"  -> 생성된 최종 검색어: '{final_query}'")
#         return {"final_query": final_query}

#     # 직원 4: 텍스트 자료조사원 (이제 'final_query'만 사용)
#     def retrieve_from_text(state):
#         print("\n--- [노드] 텍스트 DB 검색 시작 ---")
#         search_query = state.get("final_query") or state["question"] # final_query가 없으면 원래 질문 사용
#         documents = text_vectorstore.similarity_search(search_query, k=cfg['SIMILARITY_SEARCH_K'])
#         print(f"  -> '{search_query[:50]}...'(으)로 {len(documents)}개 문서 찾음.")
#         return {"documents": documents}

#     # Retreival 평가
#     def grade_documents(state):
#         print("\n--- [노드] 문서 품질 검수 시작 ---")
#         question = state.get("final_query") or state["question"]
#         documents = state.get("documents", [])
        
#         if not documents:
#             print("  -> 검수할 문서가 없어 웹 검색 필요.")
#             return {"web_search_needed": True, "documents": []}
            
#         print(f"  -> '{question[:50]}...'와 관련하여 {len(documents)}개 문서 검수 중...")

#         # 1. Reranker로 점수 계산 및 정렬
#         pairs = [(question, doc.page_content) for doc in documents]
#         scores = reranker.predict(pairs)
#         docs_with_scores = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        
#         # 2. Reranker 점수가 기준치 이상인 문서만 필터링
#         filtered_docs = []
#         print("  -> Reranker 점수 기반 필터링:")
#         for score, doc in docs_with_scores[:cfg["RERANKER_TOP_K"]]:
#             print(f"    - 문서: '{doc.page_content[:30]}...', 점수: {score:.4f}")
#             if score >= cfg["RELEVANCE_THRESHOLD"]:
#                 filtered_docs.append(doc)
#                 print("      -> (통과)")
#             else:
#                 print("      -> (탈락)")
        
#         print(f"  -> 최종 통과된 문서 (Reranker 기준): {len(filtered_docs)}개")
#         return {"documents": filtered_docs, "web_search_needed": not bool(filtered_docs)}
    
#     def refine_context_by_keyword(state):
#             """
#             VQA 키워드가 포함된 '전체 단락'을 추출하여 풍부한 문맥을 제공합니다.
#             """
#             print("\n--- [노드] VQA 키워드로 핵심 문맥 정제 시작 ---")
#             vqa_keyword = state.get("vqa_keyword")
#             documents = state.get("documents", [])
            
#             if not vqa_keyword or not documents:
#                 print("  -> VQA 키워드/문서가 없어 정제를 건너뜁니다.")
#                 return {"documents": documents}
                
#             refined_docs = []
#             for doc in documents:
#                 # 문서 내용을 빈 줄(\n\n)을 기준으로 단락으로 나눕니다.
#                 # Markdown 테이블 같은 경우도 하나의 단락으로 처리됩니다.
#                 paragraphs = re.split(r'\n\s*\n', doc.page_content)
                
#                 relevant_paragraphs = []
#                 for p in paragraphs:
#                     # 현재 단락에 VQA 키워드가 포함되어 있는지 확인
#                     if vqa_keyword in p:
#                         relevant_paragraphs.append(p)
                
#                 if relevant_paragraphs:
#                     # 키워드가 포함된 모든 단락을 합쳐서 새로운 문서를 만듭니다.
#                     new_content = "\n\n".join(relevant_paragraphs)
#                     refined_docs.append(Document(page_content=new_content, metadata=doc.metadata))
#                     print(f"  -> 문서 정제 완료. 키워드 '{vqa_keyword}'가 포함된 단락을 추출했습니다.")
#                     print(f"     - 정제 후 내용(일부): {new_content[:150]}...")
#                 else:
#                     # 만약 키워드가 포함된 단락이 하나도 없다면, 원본 문서를 그대로 사용합니다 (안전장치).
#                     print(f"  -> 경고: 문서에서 '{vqa_keyword}' 키워드가 포함된 단락을 찾지 못했습니다. 원본 문서를 사용합니다.")
#                     refined_docs.append(doc)

#             return {"documents": refined_docs}

#     def web_search(state):
#         print("\n--- [노드] 웹 검색 시작 ---")
#         question = state["final_query"] or state["question"]
#         rewritten_question = question_rewriter.invoke({"question": question})
#         web_results = web_search_tool.invoke({"query": rewritten_question})
#         web_docs = [Document(page_content=d["content"], metadata={"source": d["url"]}) for d in web_results]
#         return {"documents": web_docs}

#     def generate(state):
#         print("\n--- [노드] 최종 답변 생성 시작 ---")
#         question = state["question"]
#         documents = state.get("documents", [])
#         user_image_b64 = state.get("uploaded_image_b64")
#         ref_image_b64 = state.get("matched_reference_image_b64")
#         ocr_text = state.get("extracted_text")

#         message_content = create_multimodal_prompt(
#             question=question, 
#             documents=documents, 
#             user_image_b64=user_image_b64,
#             ref_image_b64=ref_image_b64, 
#             ocr_text=ocr_text
#         )

#         response = llm.invoke([HumanMessage(content=message_content)])
        
#         print("--- LLM 최종 답변 : ---" + "\n" + response.content)
#         return {"generation": response.content}

#     # --- 3-3. 교통정리 담당(라우터)의 판단 로직 정의 ---
#     def route_after_grading(state):
#         print("\n--- [라우터] DB 검색 결과에 따른 경로 결정 ---")
#         if state.get("web_search_needed"):
#             return "needs_web_search"
#         else:
#             return "has_documents"

#     # --- 3-4. 워크플로우(회사 조직도) 구성 ---
#     workflow = StateGraph(GraphState)
    
#     # 1. 직원(노드)들을 회사에 등록
#     workflow.add_node("perform_ocr", perform_ocr)
#     workflow.add_node("perform_visual_qa", perform_visual_qa)
#     workflow.add_node("create_final_query", create_final_query)
#     workflow.add_node("retrieve_from_text", retrieve_from_text)
#     workflow.add_node("grade_documents", grade_documents)
#     workflow.add_node("refine_context_by_keyword", refine_context_by_keyword)
#     workflow.add_node("web_search", web_search)
#     workflow.add_node("generate", generate)
    
#     # 2. 업무 시작점은 텍스트냐 이미지냐에 따라 분기
#     def route_initial_request(state):
#         if state.get("uploaded_image_b64"):
#             return "perform_ocr"
#         else:
#             return "create_final_query"

#     workflow.set_conditional_entry_point(route_initial_request, {
#         "perform_ocr": "perform_ocr",
#         "create_final_query": "create_final_query"
#     })
    
#     # 3. 업무 흐름(배선) 연결
#     workflow.add_edge("perform_ocr", "perform_visual_qa")
#     workflow.add_edge("perform_visual_qa", "create_final_query")
#     workflow.add_edge("create_final_query", "retrieve_from_text")
#     workflow.add_edge("retrieve_from_text", "grade_documents")
    
#     workflow.add_conditional_edges("grade_documents", route_after_grading, {
#         "needs_web_search": "web_search",
#         "has_documents": "refine_context_by_keyword"
#     })
#     workflow.add_edge("refine_context_by_keyword", "generate")
#     workflow.add_edge("web_search", "generate")
#     workflow.add_edge("generate", END)

#     # 4. 최종 조직도(앱) 완성!
#     return workflow.compile()




# app/rag/rag_pipeline.py

import base64
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
import re
from langgraph.graph import StateGraph, END

# LangChain 및 관련 라이브러리 임포트
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage 

# 내부 모듈 및 서비스 임포트
from app.services.image_matching_service import get_image_matching_service
from .prompt_templates import (
    create_final_query_prompt, 
    create_multimodal_prompt
)


# --- 1. Graph State 정의 ---
class GraphState(TypedDict):
    """
    그래프의 각 노드 간에 전달되는 상태 객체입니다.
    딕셔너리처럼 작동하며, 각 키는 특정 유형의 데이터를 가집니다.
    """
    question: str
    documents: List[Document]
    uploaded_image_b64: Optional[str]
    matched_reference_image_b64: Optional[str]
    vqa_keyword: Optional[str]
    final_query: Optional[str]
    generation: str

# --- 2. Graph(파이프라인) 빌드 함수 ---
def build_rag_app(llm, reranker, text_vectorstore, web_search_tool, cfg):

    # --- 2-1. 각 노드(직원)가 사용할 도구 준비 ---
    final_query_generator = create_final_query_prompt | llm | StrOutputParser()

    # --- 2-2. 각 노드(직원)의 업무 내용 정의 ---

    # 직원 1: 이미지 분석가 (Gemini VQA)
    def perform_visual_qa(state: GraphState) -> Dict:
        print("\n--- [노드] Gemini를 이용한 VQA 키워드 추출 시작 ---")
        user_image_b64 = state.get("uploaded_image_b64")
        question = state.get("question", "")

        if not user_image_b64: 
            return {"vqa_keyword": None, "matched_reference_image_b64": None}
        
        try:
            # 1. ImageMatchingService로 '정답 이미지'를 찾아오는 시도를 한다.
            service = get_image_matching_service()
            image_bytes = base64.b64decode(user_image_b64)
            ref_image_b64 = service.match_reference_image(image_bytes)

            user_img = Image.open(BytesIO(image_bytes))
            
            # 2. 상황에 맞는 프롬프트와 이미지를 준비한다.
            vqa_prompt_text = f"이 이미지(들)와 학생의 질문 '{question}'을 보고, 이 질문에 답하기 위해 교재에서 검색해야 할 가장 핵심적인 기술 용어나 키워드를 하나만 추출해줘. 명사 형태로 간결하게."
            
            vqa_content_parts = [{"type": "text", "text": vqa_prompt_text}]
            vqa_content_parts.append({"type": "image_url", "image_url": f"data:image/png;base64,{user_image_b64}"})
            
            if ref_image_b64:
                print("  -> 정답 이미지를 찾았으므로 함께 분석을 요청합니다.")
                vqa_content_parts.append({"type": "image_url", "image_url": f"data:image/png;base64,{ref_image_b64}"})

            else:
                print("  -> 정답 이미지를 찾지 못했습니다. 문제 이미지만으로 분석합니다.")

            # HumanMessage로 감싸서 invoke 호출
            response = llm.invoke([HumanMessage(content=vqa_content_parts)])
            keyword = response.content.strip().replace('"', '').replace("'", "").replace(".", "")
                        
                        
            if keyword:
                print(f"  -> Gemini VQA 성공! 추출된 키워드: '{keyword}'")
                return {"vqa_keyword": keyword, "matched_reference_image_b64": ref_image_b64}
            else:
                print("  -> Gemini가 유효한 키워드를 추출하지 못했습니다.")
                return {"vqa_keyword": None, "matched_reference_image_b64": ref_image_b64}            
        
        except Exception as e:
            print(f"  -> VQA 처리 중 오류: {e}")
        return {"vqa_keyword": None, "matched_reference_image_b64": None}                
                    
        
    # 직원 2: 최종 검색어 생성가
    def create_final_query(state: GraphState) -> Dict:
        print("\n--- [노드] 최종 검색어 생성 시작 ---")
        question = state["question"]
        vqa_keyword = state.get("vqa_keyword")
        
        # VQA 키워드가 있으면 이를 기반으로 검색어 생성, 없으면 원본 질문 사용
        if vqa_keyword:
            final_query = final_query_generator.invoke({
                "user_question": question,
                "vqa_keyword": vqa_keyword
            })
        else:
            final_query = question
        
        print(f"  -> 생성된 최종 검색어: '{final_query}'")
        return {"final_query": final_query}

    # 직원 3: 텍스트 자료조사원
    def retrieve_from_text(state: GraphState) -> Dict:
        print("\n--- [노드] 텍스트 DB 검색 시작 ---")
        search_query = state.get("final_query", "")
        documents = text_vectorstore.similarity_search(search_query, k=cfg['SIMILARITY_SEARCH_K'])
        print(f"  -> '{search_query[:50]}...' (으)로 {len(documents)}개 문서 찾음.")
        return {"documents": documents}

    # 직원 4: 자료 검수원 (Reranker)
    def grade_documents(state: GraphState) -> Dict:
        print("\n--- [노드] 문서 품질 검수 시작 ---")
        question = state.get("final_query", "")
        documents = state.get("documents", [])
        
        if not documents:
            return {"documents": []}
            
        pairs = [(question, doc.page_content) for doc in documents]
        scores = reranker.predict(pairs)
        docs_with_scores = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        
        filtered_docs = [doc for score, doc in docs_with_scores if score >= cfg["RELEVANCE_THRESHOLD"]]
        print(f"  -> Reranker 필터링 후 {len(filtered_docs)}개 문서 통과.")
        return {"documents": filtered_docs[:cfg["MAX_CONTEXT_DOCS"]]}
    
    # 직원 5: 웹 서퍼
    def web_search(state: GraphState) -> Dict:
        print("\n--- [노드] 웹 검색 시작 ---")
        question = state["final_query"] or state["question"]
        web_results = web_search_tool.invoke({"query": question})
        web_docs = [Document(page_content=d["content"], metadata={"source": d["url"]}) for d in web_results]
        print(f"  -> 웹에서 {len(web_docs)}개 정보 찾음.")
        # 웹 검색 결과는 기존 문서에 추가
        existing_docs = state.get("documents", [])
        return {"documents": existing_docs + web_docs}

    # 직원 6: 최종 답변 작성가 (Gemini)
    def generate(state: GraphState) -> Dict:
        print("\n--- [노드] Gemini를 이용한 최종 답변 생성 시작 ---")
        
        message_content = create_multimodal_prompt(state)
        
        response = llm.invoke([HumanMessage(content=message_content)])
        
        print("--- Gemini 최종 답변 : ---\n" + response.content)
        return {"generation": response.content}

    # --- 2-3. 워크플로우(회사 조직도) 구성 ---
    workflow = StateGraph(GraphState)
    
    workflow.add_node("perform_visual_qa", perform_visual_qa)
    workflow.add_node("create_final_query", create_final_query)
    workflow.add_node("retrieve_from_text", retrieve_from_text)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    
    # --- 2-4. 업무 흐름(배선) 및 조건부 분기(라우터) 정의 ---

    def route_initial_request(state: GraphState) -> str:
        """사용자 요청에 이미지가 포함되어 있는지에 따라 첫 노드를 결정"""
        if state.get("uploaded_image_b64"):
            return "perform_visual_qa"
        else:
            return "create_final_query"

    def route_after_vqa(state: GraphState) -> str:
        """VQA로 키워드를 얻었는지에 따라 경로 결정"""
        print("\n--- [라우터] VQA 결과에 따른 경로 결정 ---")
        if state.get("vqa_keyword"):
            print("  -> VQA 키워드 추출 성공. 검색어 생성으로 이동.")
            return "create_final_query"
        else:
            print("  -> VQA 키워드 추출 실패. RAG 없이 바로 답변 생성으로 이동.")
            return "generate_no_rag"

    def route_after_grading(state: GraphState) -> str:
        """DB 검색 결과가 유의미한지에 따라 경로 결정"""
        print("\n--- [라우터] DB 검색 결과에 따른 경로 결정 ---")
        if state.get("documents"):
             print("  -> 관련 문서 찾음. 답변 생성으로 이동.")
             return "generate_with_rag"
        else:
             print("  -> 관련 문서 없음. 웹 검색으로 이동.")
             return "web_search"

    workflow.set_conditional_entry_point(
        route_initial_request,
        {
            "perform_visual_qa": "perform_visual_qa",
            "create_final_query": "create_final_query",
        }
    )
    
    workflow.add_conditional_edges(
        "perform_visual_qa",
        route_after_vqa,
        {
            "create_final_query": "create_final_query",
            "generate_no_rag": "generate" # 실패 시 바로 generate로 점프
        }
    )

    workflow.add_edge("create_final_query", "retrieve_from_text")
    workflow.add_edge("retrieve_from_text", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate_with_rag": "generate",
            "web_search": "web_search"
        }
    )
    
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()