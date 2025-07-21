from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain.schema import Document
from . import config

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="문서가 질문과 관련이 있으면 'yes', 없으면 'no'")

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing relevance of a retrieved document to a user question. "
               "Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question. "
               "Respond ONLY with a JSON object in this exact format: {{\"binary_score\": \"yes\"}} or {{\"binary_score\": \"no\"}}. "
               "Do not include any other text or explanations."),
    ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
])

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a question re-writer. Your task is to convert an input question into a concise, "
               "and effective version that is optimized for search. The output MUST be in the same language "
               "as the input question. Do NOT provide explanations, options, or any surrounding text. "
               "Respond ONLY with the rewritten question."),
    ("human", "Here is the initial question:\n\n{question}\n\nRewritten question:"),
])

def create_generate_prompt(question: str, documents: List[Document], user_has_uploaded_image: bool) -> str:
    prompt_parts = [
        "You are a helpful and brilliant AI assistant. Your main goal is to accurately answer the user's question. Follow these rules as your best:",
        "1. 이미지 파일이 제공된다면 사용자가 업로드한 이미지를 가장 먼저 우선시해주세요. 만약 사용자가 그림 또는 사진에 대해 설명을 요구한다면 예: '이 그림에 대해 설명해줘' or '첨부한 사진에 대해 설명해줘'.",
        "2. 사용자가 업로드한 이미지가 있다면, 가장 먼저 retrieved context를 사용해서(text와 images from the database) 질문에 대해 답해주세요",
        "3. 만약 retrieved context가 question과 관련성이 낮다면, retrieved context를 무시하시고 사용자의 question에 자연스럽게 일상적인 대답을 해주세요.",
        "4. 제공된 정보들 (text, user's image, retrieved text, retrieved images)를 모두 사용하여 교육 정보를 구조적으로 작성한 뒤, 최종적으로 사람이 받아들일 수 있는 답변을 해주세요",
        "5. 마지막으로, 사용자가 따로 명시하지 않는다면, 항상 한국어로 답변을 제공해주세요.\n"
    ]
    
    if user_has_uploaded_image:
        prompt_parts.append("--- User's Uploaded Image ---\n(An image has been provided by the user. It is the FIRST image. Please refer to it for your answer.)\n")

    if documents:
        prompt_parts.append("--- Retrieved Context from Database (Use only if relevant) ---")
        text_context = "\n".join([f"Doc {i+1}: {doc.page_content}" for i, doc in enumerate(documents[:config.MAX_CONTEXT_DOCS])])
        prompt_parts.append(text_context)
        prompt_parts.append("----------------------------------------------------------------\n")
    
    prompt_parts.append(f"=== User's Question ===\n{question}\n")
    prompt_parts.append("=== Final Answer (in Korean) ===")

    return "\n".join(prompt_parts)