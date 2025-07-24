# # rag_pipeline.py
# import json
# import os
# import base64
# from io import BytesIO
# from PIL import Image
# from typing import List, Dict, Any, Optional
# from typing_extensions import TypedDict

# from langgraph.graph import StateGraph, END
# from langchain.schema import Document
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableLambda


# # GradeDocuments는 기존 프롬프트에서 가져온다고 가정합니다.
# from .prompt_templates import GradeDocuments, grade_prompt, rewrite_prompt, create_generate_prompt

# # --- 1. GraphState 확장 --- (<--- MODIFIED)
# class GraphState(TypedDict):
#     question: str
#     documents: List[Document]
#     web_search_needed: bool
#     generation: str
#     error: Optional[str]
#     uploaded_image_b64: Optional[str]
#     extracted_text: Optional[str]
#     image_intent: Optional[str]        # <--- NEW: 이미지 질문의 의도 (rag vs direct_generate)
#     generation_feedback: Optional[str] # <--- NEW: 답변 재생성을 위한 피드백

# # --- 기존 유틸리티 함수 ---
# def safe_parse_json(text: str) -> GradeDocuments:
#     try:
#         return GradeDocuments(**json.loads(text))
#     except (json.JSONDecodeError, TypeError):
#         text_lower = text.lower()
#         if 'yes' in text_lower:
#             return GradeDocuments(binary_score='yes')
#         return GradeDocuments(binary_score='no')

# # --- 개선된 파이프라인 빌드 함수 ---
# def build_rag_app(llm, reranker, text_vectorstore, image_vectorstore, web_search_tool, cfg):
#     # --- 기존 체인들 ---
#     # 1. 주력으로 사용할 Pydantic 파서를 정의합니다. (기존 코드)
#     grade_parser = PydanticOutputParser(pydantic_object=GradeDocuments)

#     # 2. Fallback(대체 작동)으로 사용할 파서를 정의합니다. 
#     #    기존 safe_parse_json 로직을 RunnableLambda로 감싸줍니다.
#     #    LLM의 전체 출력(ai_msg)을 받아 content를 추출하는 것까지 포함합니다.
#     fallback_parser = RunnableLambda(lambda ai_msg: safe_parse_json(ai_msg.content))

#     # 3. 주력 파서에 fallback을 연결하여 더욱 강력한 파서를 만듭니다.
#     #    grade_parser가 실패하면, fallback_parser가 실행됩니다.
#     robust_grader_parser = grade_parser.with_fallbacks(
#         fallbacks=[fallback_parser],
#     )

#     # 4. 최종적으로 retrieval_grader 체인에 새로운 파서를 적용합니다.
#     retrieval_grader = grade_prompt | llm | robust_grader_parser
    
    
#     retrieval_grader = grade_prompt | llm | (lambda ai_msg: safe_parse_json(ai_msg.content))
#     question_rewriter = rewrite_prompt | llm | StrOutputParser()

#     # --- 2. 새로운 노드를 위한 프롬프트 및 파서 정의 --- (<--- NEW)
    
#     # 2-1. 이미지 질문 의도 분류 프롬프트
#     intent_classifier_prompt = ChatPromptTemplate.from_messages([
#         SystemMessage(content="""You are an expert at classifying user intent for a multimodal RAG system.
# The user has provided an image and a question. Your task is to determine the user's primary goal.
# Choose one of the following two intents:
# 1. `information_retrieval`: The user wants to find information RELATED TO the content of the image. This usually involves recognizing text, objects, or concepts in the image and then searching a knowledge base. Examples: "What theory does this diagram explain?", "Summarize the document in this picture."
# 2. `direct_description`: The user wants a direct description, analysis, or creation based on the image itself, without needing an external knowledge base. Examples: "Describe this image.", "What is funny about this picture?", "Translate the text in this image."

# Respond with ONLY `information_retrieval` or `direct_description`."""),
#         HumanMessage(content=[
#             {"type": "text", "text": "{question}"},
#             {"type": "image_url", "image_url": {"url": "data:image/png;base64,{image_b64}"}}
#         ])
#     ])
#     intent_classifier_chain = intent_classifier_prompt | llm | StrOutputParser()
    
    

#     # 2-2. 생성된 답변 품질 평가 프롬프트
#     class GradeGeneration(TypedDict):
#         score: str # 'good' or 'bad'
#         feedback: str # Reason for the score, used for regeneration.

#     generation_grader_parser = JsonOutputParser(pydantic_object=GradeGeneration)
#     generation_grader_prompt = ChatPromptTemplate.from_messages([
#         SystemMessage(content="""You are a quality assurance expert for an AI tutor. Your task is to evaluate a generated answer based on the user's question and the provided context (documents).
# Provide a score of 'good' if the answer is faithful to the context, directly addresses the user's question, and is clear.
# Provide a score of 'bad' if the answer contains hallucinations, is irrelevant, or is poorly written.
# When the score is 'bad', provide concise feedback on what to fix.
# Respond with a JSON object containing 'score' and 'feedback' keys."""),
#         HumanMessage(content="""
# [USER QUESTION]: {question}
# -------------------
# [CONTEXT DOCUMENTS]:
# {documents}
# -------------------
# [GENERATED ANSWER]:
# {generation}
# -------------------
# JSON Response:
# """)
#     ])
#     generation_grader_chain = generation_grader_prompt | llm | generation_grader_parser


#     # --- 3. 노드 함수 정의 ---

#     # 3-1. 새로운 노드 함수들 (<--- NEW)
#     def classify_image_intent(state: GraphState) -> Dict[str, Any]:
#         """이미지 기반 질문의 의도를 분류하여 state에 저장합니다."""
#         print("노드: classify_image_intent")
#         question = state["question"]
#         image_b64 = state["uploaded_image_b64"]
        
#         if not image_b64:
#              return {"image_intent": "no_image"}

#         intent = intent_classifier_chain.invoke({"question": question, "image_b64": image_b64})
        
#         # LLM 응답 정리
#         intent_clean = "direct_description"
#         if "information_retrieval" in intent.lower():
#             intent_clean = "information_retrieval"
        
#         print(f" -> 분류된 의도: {intent_clean}")
#         return {"image_intent": intent_clean}

#     def grade_generation_node(state: GraphState) -> Dict[str, Any]:
#         """생성된 답변의 품질을 평가하고, 피드백을 state에 저장합니다."""
#         print("노드: grade_generation")
#         question = state["question"]
#         documents = state["documents"]
#         generation = state["generation"]

#         doc_content = "\n\n".join([doc.page_content for doc in documents])
        
#         try:
#             grade = generation_grader_chain.invoke({
#                 "question": question, 
#                 "documents": doc_content, 
#                 "generation": generation
#             })
#             print(f" -> 답변 평가 결과: {grade['score']}. 피드백: {grade['feedback']}")
#             if grade['score'] == 'bad':
#                 return {"generation_feedback": grade['feedback']}
#             else:
#                 return {"generation_feedback": None} # 품질이 좋으면 피드백 없음
#         except Exception as e:
#             print(f" -> 답변 평가 중 에러 발생: {e}. 품질을 'good'으로 간주합니다.")
#             return {"generation_feedback": None}

#     # 3-2. 기존 노드 함수들 (일부 수정)
#     def decide_image_or_text(state: GraphState) -> str:
#         print("판단: 이미지 또는 텍스트 RAG?")
#         if state.get("uploaded_image_b64"):
#             print(" -> 경로: 이미지 있음. 'classify_image_intent'로 이동")
#             return "classify_image_intent"
#         else:
#             print(" -> 경로: 이미지 없음. 'retrieve'로 이동")
#             return "retrieve"

#     def retrieve(state: GraphState):
#         print("노드: retrieve")
#         question = state["question"]
#         documents = text_vectorstore.similarity_search(question, k=cfg['SIMILARITY_SEARCH_K'])
#         return {"documents": documents, "web_search_needed": False} # 검색 후 초기화

#     def retrieve_from_ocr(state: GraphState):
#         print("노드: retrieve_from_ocr")
#         extracted_text = state.get("extracted_text")
#         if not extracted_text:
#             print(" -> OCR 텍스트가 없어 문서를 찾을 수 없습니다.")
#             return {"documents": [], "web_search_needed": True} # 문서 없으면 웹검색 필요
        
#         print(f" -> OCR 텍스트로 문서를 검색합니다: {extracted_text[:100]}...")
#         documents = text_vectorstore.similarity_search(extracted_text, k=cfg['SIMILARITY_SEARCH_K'])
#         return {"documents": documents, "web_search_needed": not bool(documents)}

#     def grade_documents_node(state: GraphState):
#         print("노드: grade_documents")
#         question = state["question"]
#         documents = state["documents"]
        
#         if not documents:
#             return {"documents": [], "web_search_needed": True}
        
#         question_for_grading = state.get("extracted_text") or question
#         print(f" -> 문서 관련성 평가 기준 질문: '{question_for_grading[:100]}...'")

#         pairs = [(question_for_grading, doc.page_content) for doc in documents]
#         scores = reranker.predict(pairs)
#         docs_with_scores = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        
#         filtered_docs = []
#         for score, doc in docs_with_scores[:cfg["RERANKER_TOP_K"]]:
#             if score < cfg["RELEVANCE_THRESHOLD"]: continue
#             grade = retrieval_grader.invoke({"question": question_for_grading, "document": doc.page_content})
#             if grade.binary_score == 'yes':
#                 filtered_docs.append(doc)
        
#         web_search_needed = not bool(filtered_docs)
#         # 이미지 RAG 경로에서 문서를 못 찾았을 경우 웹 검색을 허용하되,
#         # 단순 이미지 설명 요청 시에는 웹 검색을 막는 로직이 필요. (여기서는 분기 로직으로 제어)
#         if state.get("image_intent") == "direct_description":
#              web_search_needed = False

#         return {"documents": filtered_docs, "web_search_needed": web_search_needed}

#     def decide_to_generate(state: GraphState) -> str:
#         print("판단: Generate OR Web Search?")
#         # 오타 수정: web_search_neded -> web_search_needed (<--- MODIFIED)
#         if state.get("web_search_needed"):
#             print(" -> 경로: 문서 품질 낮음. 'rewrite_query'로 이동하여 웹 검색")
#             return "rewrite_query"
#         else:
#             print(" -> 경로: 문서 품질 좋음. 'generate'로 이동")
#             return "generate"
    
#     def rewrite_query(state: GraphState):
#         print("노드: rewrite_query")
#         question = state["question"]
#         rewritten_question = question_rewriter.invoke({"question": question})
#         return {"question": rewritten_question}

#     def web_search(state: GraphState):
#         print("노드: web_search")
#         question = state["question"]
#         web_results = web_search_tool.invoke({"query": question})
#         web_docs = [Document(page_content=d["content"], metadata={"source": d["url"]}) for d in web_results]
#         return {"documents": web_docs}

#     def generate(state: GraphState):
#         print("--- 노드: generate ---")
#         feedback = state.get("generation_feedback")
#         if feedback:
#             print(f" -> 답변 재생성 요청. 피드백: {feedback}")
        
#         question = state["question"]
#         documents = state.get("documents", [])
#         uploaded_image_b64 = state.get("uploaded_image_b64")
#         extracted_text = state.get("extracted_text")

#         # create_generate_prompt가 피드백을 인자로 받도록 수정되었다고 가정
#         # 예: def create_generate_prompt(..., feedback: Optional[str] = None)
#         prompt_text = create_generate_prompt(
#             question, documents, bool(uploaded_image_b64), extracted_text, feedback
#         )
        
#         print(" -> 생성 프롬프트 준비 완료")
#         message_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]

#         if uploaded_image_b64:
#             message_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{uploaded_image_b64}"}})
        
#         if documents:
#             for doc in documents:
#                 img_path = doc.metadata.get("image_path")
#                 if img_path and os.path.exists(img_path):
#                     try:
#                         with Image.open(img_path) as img:
#                             buffer = BytesIO()
#                             img.convert("RGB").save(buffer, format="PNG")
#                             img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
#                             message_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}})
#                     except Exception as e:
#                         print(f"  -> 경고: 이미지 파일 로딩 실패. {img_path}, 오류: {e}")
        
#         response = llm.invoke([HumanMessage(content=message_content)])
#         print(" -> LLM 답변 생성 완료")
#         return {"generation": response.content, "generation_feedback": None} # 생성 후 피드백 초기화

#     # --- 4. 그래프 구성 및 연결 --- (<--- MODIFIED)
#     workflow = StateGraph(GraphState)

#     # 노드 추가
#     workflow.add_node("retrieve", retrieve)
#     workflow.add_node("classify_image_intent", classify_image_intent) # <--- NEW
#     workflow.add_node("retrieve_from_ocr", retrieve_from_ocr)
#     workflow.add_node("grade_documents", grade_documents_node)
#     workflow.add_node("rewrite_query", rewrite_query)
#     workflow.add_node("web_search", web_search)
#     workflow.add_node("generate", generate)
#     workflow.add_node("grade_generation", grade_generation_node) # <--- NEW

#     # 진입점 설정
#     workflow.set_conditional_entry_point(
#         decide_image_or_text,
#         {
#             "classify_image_intent": "classify_image_intent",
#             "retrieve": "retrieve",
#         }
#     )

#     # 이미지 의도에 따른 분기
#     workflow.add_conditional_edges(
#         "classify_image_intent",
#         lambda state: state.get("image_intent"),
#         {
#             "information_retrieval": "retrieve_from_ocr",
#             "direct_description": "generate", # 단순 설명은 RAG 없이 바로 생성으로
#         }
#     )
    
#     # RAG 경로 설정
#     workflow.add_edge("retrieve", "grade_documents")
#     workflow.add_edge("retrieve_from_ocr", "grade_documents")
    
#     # 문서 품질 평가 후 분기
#     workflow.add_conditional_edges(
#         "grade_documents",
#         decide_to_generate,
#         {"rewrite_query": "rewrite_query", "generate": "generate"}
#     )

#     # 웹 검색 경로
#     workflow.add_edge("rewrite_query", "web_search")
#     workflow.add_edge("web_search", "grade_documents") # 웹 검색 후 다시 문서 평가

#     # 답변 생성 후 자체 평가
#     workflow.add_edge("generate", "grade_generation")
#     workflow.add_conditional_edges(
#         "grade_generation",
#         lambda state: "regenerate" if state.get("generation_feedback") else "end",
#         {
#             "regenerate": "generate", # 품질 나쁘면 재생성
#             "end": END,               # 품질 좋으면 종료
#         }
#     )

#     return workflow.compile()





































































import json
import os
import base64
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langchain.schema import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

from .prompt_templates import GradeDocuments, grade_prompt, rewrite_prompt, create_generate_prompt

class GraphState(TypedDict):
    question: str              # 사용자 질문
    documents: List[Document]  # 찾은 관련 문서들
    web_search_needed: bool    # 웹 검색이 필요한지
    generation: str           # AI가 생성한 답변
    error: Optional[str]      # 에러 발생시 메시지
    uploaded_image_b64: Optional[str]  # 업로드된 이미지
    extracted_text: Optional[str]   # OCR 텍스트

def safe_parse_json(text: str) -> GradeDocuments:
    try:
        return GradeDocuments(**json.loads(text))
    except (json.JSONDecodeError, TypeError):
        text_lower = text.lower()
        if 'yes' in text_lower:
            return GradeDocuments(binary_score='yes')
        return GradeDocuments(binary_score='no')

# def route_query(state):
#     """
#     쿼리 종류에 따라 경로를 결정하는 라우터 노드.
#     이미지가 있으면 바로 generate로, 없으면 retrieve로 보낸다.
#     """
#     print("판단: 쿼리 라우팅")
#     if state.get("uploaded_image_b64"):
#         print(" -> 경로: 이미지 설명 요청. 'generate'로 직접 이동")
#         state['documents'] = []
#         return "generate"
#     else:
#         print(" -> 경로: 텍스트 RAG 요청. 'retrieve'로 이동")
#         return "retrieve"
    
def decide_image_or_text(state):
    """
    상태를 보고 'generate'로 갈지 'retrieve'로 갈지 '결정'만 하는 함수.
    단순히 문자열을 반환합니다.
    """
    print("판단: 이미지 또는 텍스트 RAG?")
    # 'uploaded_image_b64' 키가 존재하고, 그 값이 비어있지 않은지(Truthy) 확인
    print("\n--- [PIPELINE 진단] 경로 결정 노드 시작 ---")
    image_data = state.get("uploaded_image_b64")
    print(f"--- [PIPELINE 진단] state에서 확인한 이미지 데이터 길이: {len(image_data) if image_data and isinstance(image_data, str) else 0} ---")
    
    if image_data:
        print(" -> 경로: 이미지 있음. 'retrieve_from_ocr'로 이동")
        return "retrieve_from_ocr"
    else:
        print(" -> 경로: 이미지 없음. 'retrieve'로 이동")
        return "retrieve"



    # if state.get("uploaded_image_b64"):
    #     print(" -> 경로: 이미지 있음. 'generate'로 이동")
    #     return "generate"
    # else:
    #     print(" -> 경로: 이미지 없음. 'retrieve'로 이동")
    #     return "retrieve"

def build_rag_app(llm, reranker, text_vectorstore, image_vectorstore, web_search_tool, cfg):
    grade_parser = PydanticOutputParser(pydantic_object=GradeDocuments)
    retrieval_grader = grade_prompt | llm | (lambda ai_msg: safe_parse_json(ai_msg.content))
    question_rewriter = rewrite_prompt | llm | StrOutputParser()

    def retrieve(state):
        print("노드: retrieve")
        question = state["question"]
        documents = text_vectorstore.similarity_search(question, k=cfg['SIMILARITY_SEARCH_K'])
        return {"documents": documents}

    def grade_documents_node(state):
        print("노드: grade_documents")
        question = state["question"]
        documents = state["documents"]
        extracted_text = state.get("extracted_text") # OCR 텍스트를 가져옵니다.
        
        if not documents:
            return {"documents": [], "web_search_needed": True}
        
        question_for_grading = extracted_text if extracted_text else question
        
        if not question_for_grading:
            print(f"  -> 문서 관련성 평가 기준 질문이 없어 웹검색을 수행합니다")
            return {"documents": [], "web_search_needed": True}
        
        if question_for_grading:
            print(f" -> 문서 관련성 평가 기준 질문: '{question_for_grading[:100]}...'")
        else:
            print(f" -> 문서 관련성 평가 기준 질문이 없습니다.")
            return {"documents": [], "web_search_needed": True}

        pairs = [(question_for_grading, doc.page_content) for doc in documents]
        scores = reranker.predict(pairs)
        docs_with_scores = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        
        filtered_docs = []
        for score, doc in docs_with_scores[:cfg["RERANKER_TOP_K"]]:
            if score < cfg["RELEVANCE_THRESHOLD"]:
                continue

            grade = retrieval_grader.invoke({"question": question_for_grading, "document": doc.page_content})
            if grade.binary_score == 'yes':
                filtered_docs.append(doc)
        
        # 만약 모든 문서가 걸러졌다면, 웹 검색이 필요하다고 판단할 수 있습니다.
        # (단, 이미지 경로에서는 웹 검색을 막는 로직이 필요할 수 있습니다.)
        web_search_needed = not bool(filtered_docs)
        if extracted_text:
            web_search_needed = False # 이미지 설명 요청 시에는 웹 검색을 하지 않도록 강제

        return {"documents": filtered_docs, "web_search_needed": web_search_needed}

    def decide_to_generate(state):
        print("판단: Generate OR Web Search?")
        if state.get("uploaded_image_b64"): 
            print(" 경로: 이미지 있음, 'generate'로 이동")
            return "generate"
        
        if state.get("documents"):
            print(" 경로: 텍스트 있음, 'retrieve'로 이동")
            return "web_search" if state.get("web_search_needed") == True else "generate"

        return "generate"

    def rewrite_query(state):
        print("노드: rewrite_query")
        question = state["question"]
        rewritten_question = question_rewriter.invoke({"question": question})
        return {"question": rewritten_question}

    def web_search(state):
        print("노드: web_search")
        question = state["question"]
        web_results = web_search_tool.invoke({"query": question})
        web_docs = [Document(page_content=d["content"], metadata={"source": d["url"]}) for d in web_results]
        return {"documents": web_docs}

    def retrieve_from_ocr(state):
        print("노드: retrieve_from_ocr")
        extracted_text = state.get("extracted_text")
        if not extracted_text:
            print(" -> OCR 텍스트가 없어 문서를 찾을 수 없습니다.")
            return {"documents": []}
        
        print(f" -> OCR 텍스트로 문서를 검색합니다: {extracted_text[:100]}...")
        # OCR 텍스트를 검색어로 사용하여 텍스트 DB를 검색합니다.
        documents = []
        if extracted_text:
            documents = text_vectorstore.similarity_search(extracted_text, k=cfg['SIMILARITY_SEARCH_K'])
        return {"documents": documents}
    
    # 이 함수를 build_rag_app 함수 내부에 추가하거나, 외부에 정의하고 전달할 수 있습니다.
    def route_after_ocr_retrieval(state):
        """
        OCR 기반 문서 검색 후, 결과에 따라 경로를 결정합니다.
        문서를 찾았으면 평가(grade)하고, 못 찾았으면 바로 생성(generate)합니다.
        """
        print("판단: OCR 검색 후 경로 결정")
        documents = state.get("documents", [])
        
        if documents:
            print(f" -> 경로: OCR로 문서 {len(documents)}개 찾음. 'grade_documents'로 이동하여 평가 계속")
            return "grade_documents"
        else:
            print(" -> 경로: OCR로 문서 못 찾음. 'generate'로 직접 이동하여 이미지 설명 요청")
            # 이 경우, state의 documents는 이미 비어있으므로 generate 노드가 올바르게 동작합니다.
            return "generate"

    def generate(state):
        print("--- 노드: generate 시작합니다 ---")
        try:
            state_for_log = state.copy() # 1. state를 복사합니다.
            if state_for_log.get("uploaded_image_b64"):
            # 2. 이미지 데이터가 있다면, 긴 문자열 대신 짧은 메시지로 바꿉니다.
                img_len = len(state_for_log["uploaded_image_b64"]) if state_for_log["uploaded_image_b64"] else 0
                state_for_log["uploaded_image_b64"] = f"<Image Base64 Data, Length: {img_len}>"
        
            # 3. 깔끔하게 정리된 복사본을 출력합니다.
            print(f"[진단] 현재 state (이미지 제외): {state_for_log}")
            # print(f"[진단] 현재 state: {state}")

            question = state["question"]
            documents = state.get("documents", [])  # rag로 찾은 텍스트 문서들 
            uploaded_image_b64 = state.get("uploaded_image_b64")
            extracted_text = state.get("extracted_text")
            
            print(f"[진단] 질문: {question}")
            print(f"[진단] 검색된 문서 개수: {len(documents)}")
            print(f"[진단] 이미지 업로드 여부: {bool(uploaded_image_b64)}")

            print("[진단] create_generate_prompt 함수 호출 직전...")
            prompt_text = create_generate_prompt(
                question, 
                documents, 
                bool(uploaded_image_b64), 
                extracted_text
            )

            print("[진단] 프롬프트 생성 완료. 내용은 다음과 같음:")
            print("-------------------- 프롬프트 시작 --------------------")
            print(prompt_text)
            print("-------------------- 프롬프트 끝 ----------------------")


            message_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]

            # 1. 사용자가 직접 업로드한 이미지가 있담녀 먼저 추가\
            if uploaded_image_b64:
                print("[진단] 업로드된 이미지를 메시지에 추가합니다.")
                message_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{uploaded_image_b64}"}})

            # 2. rag로 찾은 텍스트문서들을 확인하며 꼬리표(metadata)에 이미지 경로가 있는지 확인
            print(f"RAG로 찾은 문서 {len(documents)}개를 확인하여 관련 이미지를 찾습니다.")
            if documents:
                print(f"[진단] 문서 {len(documents)}개에서 연결된 이미지를 찾습니다.")
                for doc in documents:
                # 문서의 메타데이터에서 'image_path' 키를 찾습니다.
                    img_path = doc.metadata.get("image_path")
                
                # 이미지 경로가 있고, 그 파일이 실제로 존재한다면
                if img_path and os.path.exists(img_path):
                    print(f"  -> 발견! 문서와 연결된 이미지: {img_path}")
                    try:
                        # 해당 이미지를 열어서 base64로 인코딩합니다.
                        with Image.open(img_path) as img:
                            buffer = BytesIO()
                            # PNG 포맷으로 저장하여 호환성을 높입니다.
                            img.convert("RGB").save(buffer, format="PNG")
                            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                            
                            # LLM에게 보낼 메시지에 이미지를 추가합니다.
                            message_content.append({
                                "type": "image_url", 
                                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                            })
                    except Exception as e:
                        print(f"  -> 경고: 이미지 파일을 여는 데 실패했습니다. {img_path}, 오류: {e}")

            # 4. LLM 호출 직전에 최종 메시지 내용을 확인합니다.
            print(f"[진단] LLM에 전달할 최종 메시지 개수: {len(message_content)}")
            print("[진단] llm.invoke 호출 직전...")
            print(f"LLM에게 총 {len(message_content)}개의 콘텐츠(텍스트 1개, 이미지 {len(message_content)-1}개)를 전달합니다.")
            response = llm.invoke([HumanMessage(content=message_content)])
            print("[진단] llm.invoke 호출 완료.")
            print("-" * 50, f"\nLLM 응답: {response.content}\n", "-" * 50)
            print("--- [진단] generate 노드 정상 종료 ---\n")
            return {"generation": response.content}

        except Exception as e:
            # 5. 에러가 발생하면, 정확한 에러 메시지와 위치를 출력합니다.
            import traceback
            print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!! [진단] generate 노드에서 에러 발생 !!!!!!")
            print(f"!!!!!! 에러 타입: {type(e)}")
            print(f"!!!!!! 에러 메시지: {e}")
            print("!!!!!! Traceback (에러 발생 위치):")
            traceback.print_exc()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
            # 에러를 다시 발생시켜서 파이프라인이 멈추도록 합니다.
            raise e



    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("retrieve_from_ocr", retrieve_from_ocr)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    
    # 시작점을 설정합
    workflow.set_conditional_entry_point(
        decide_image_or_text,
        {
            "retrieve_from_ocr": "retrieve_from_ocr", # 이미지가 있으면, ocr 검색 노듬ㅀ
            "retrieve": "retrieve", # 없으면, 기존 텍스트 검색 노드록 감. retrieve로 시작
        }
    )

    # 텍스트 rag
    # [경로 1: 일반 텍스트 RAG]
    workflow.add_edge("retrieve", "grade_documents")
    
    # [경로 2: 이미지 기반 RAG] - 검색 후 바로 평가 단계로 합류합니다.
    # workflow.add_edge("retrieve_from_ocr", "grade_documents")

    workflow.add_conditional_edges(
        "retrieve_from_ocr",
        route_after_ocr_retrieval,
        {
            "grade_documents": "grade_documents",
            "generate": "generate",
        }
    )

    # [공통 경로: 평가 이후]
    workflow.add_conditional_edges(
        "grade_documents", 
        decide_to_generate, 
        {"rewrite_query": "rewrite_query", "generate": "generate"}
    )
    workflow.add_edge("rewrite_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()