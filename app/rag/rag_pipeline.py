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
    print(f"--- [PIPELINE 진단] state에서 확인한 이미지 데이터 길이: {len(image_data) if image_data else 0} ---")
    
    if state.get("uploaded_image_b64"):
        print(" -> 경로: 이미지 있음. 'generate'로 이동")
        return "generate"
    else:
        print(" -> 경로: 이미지 없음. 'retrieve'로 이동")
        return "retrieve"

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
        print(f" -> 문서 관련성 평가 기준 질문: '{question_for_grading[:100]}...'")


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
        print("판단: Generate OR Web Search OR Retrieve?")
        if state.get("uploaded_image_b64"): 
            print(" 경로: 이미지 있음, 'generate'로 이동")
            return "generate"
        
        if state.get("documents"):
            print(" 경로: 텍스트 있음, 'retrieve'로 이동")
            return "web_search" if state.get("web_search_neded") else "generate"

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
        documents = text_vectorstore.similarity_search(extracted_text, k=cfg['SIMILARITY_SEARCH_K'])
        return {"documents": documents}

    def generate(state):
        print("--- 노드: generate 시작합니다 ---")
        try:
            state_for_log = state.copy() # 1. state를 복사합니다.
            if state_for_log.get("uploaded_image_b64"):
            # 2. 이미지 데이터가 있다면, 긴 문자열 대신 짧은 메시지로 바꿉니다.
                img_len = len(state_for_log["uploaded_image_b64"])
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
            "generate": "retrieve_from_ocr", # 이미지가 있으면, ocr 검색 노듬ㅀ
            "retrieve": "retrieve", # 없으면, 기존 텍스트 검색 노드록 감. retrieve로 시작
        }
    )

    # 텍스트 rag
    # [경로 1: 일반 텍스트 RAG]
    workflow.add_edge("retrieve", "grade_documents")
    
    # [경로 2: 이미지 기반 RAG] - 검색 후 바로 평가 단계로 합류합니다.
    workflow.add_edge("retrieve_from_ocr", "grade_documents")

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