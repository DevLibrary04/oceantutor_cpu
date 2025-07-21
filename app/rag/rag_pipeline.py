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
    question: str
    documents: List[Document]
    web_search_needed: bool
    generation: str
    error: Optional[str]
    uploaded_image_b64: Optional[str]

def safe_parse_json(text: str) -> GradeDocuments:
    try:
        return GradeDocuments(**json.loads(text))
    except (json.JSONDecodeError, TypeError):
        text_lower = text.lower()
        if 'yes' in text_lower:
            return GradeDocuments(binary_score='yes')
        return GradeDocuments(binary_score='no')

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
        
        if not documents:
            return {"documents": [], "web_search_needed": True}
        
        pairs = [(question, doc.page_content) for doc in documents]
        scores = reranker.predict(pairs)
        docs_with_scores = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        
        filtered_docs = []
        for score, doc in docs_with_scores[:cfg["RERANKER_TOP_K"]]:
            if score < cfg["RELEVANCE_THRESHOLD"]:
                continue
            grade = retrieval_grader.invoke({"question": question, "document": doc.page_content})
            if grade.binary_score == 'yes':
                filtered_docs.append(doc)
        
        return {"documents": filtered_docs, "web_search_needed": not bool(filtered_docs)}

    def decide_to_generate(state):
        print("판단: Generate OR Web Search?")
        if state.get("uploaded_image_b64"): return "generate"
        if state.get("web_search_needed"): return "rewrite_query"
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

    def generate(state):
        print("노드: generate")
        question = state["question"]
        documents = state.get("documents", [])
        uploaded_image_b64 = state.get("uploaded_image_b64")
        
        prompt_text = create_generate_prompt(question, documents, bool(uploaded_image_b64))
        message_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]

        if uploaded_image_b64:
            message_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{uploaded_image_b64}"}})

        if image_vectorstore:
            image_docs = image_vectorstore.similarity_search(question, k=2)
            for doc in image_docs:
                img_path = doc.metadata.get("image_path")
                if img_path and os.path.exists(img_path):
                    with Image.open(img_path) as img:
                        buffer = BytesIO()
                        img.convert("RGB").save(buffer, format="PNG")
                        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        message_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}})
        
        response = llm.invoke([HumanMessage(content=message_content)])
        print("-" * 50, f"\nLLM 응답: {response.content}\n", "-" * 50)
        return {"generation": response.content}

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate, {"rewrite_query": "rewrite_query", "generate": "generate"})
    workflow.add_edge("rewrite_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()