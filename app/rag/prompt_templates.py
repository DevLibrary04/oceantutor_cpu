from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional
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
# feedback: Optional[str] = None
def create_generate_prompt(question: str, documents: List[Document], user_has_uploaded_image: bool, extracted_text: Optional[str]) -> str:
    
    # --- [새로 추가될 부분 시작] ---
    # feedback이 실제로 존재할 때만 특별 지시사항을 만듭니다.
#     feedback_instruction = ""
#     if feedback:
#         feedback_instruction = f"""
# [ 중요! 이전 답변 수정 요청 ]
# 이전 답변은 사용자의 요구를 충족하지 못했습니다. 아래 피드백을 반드시 반영하여 답변을 다시 생성해주세요.
# 피드백: {feedback}
# ----------------------------------

# """
    
    prompt_parts = ["""
[ 페르소나 (당신의 역할) ]
당신은 **'해기사 자격증(소형선박조종사, 항해사, 기관사)'** 및 관련 선박 운항 기술 지식에 특화된 전문 AI 교사입니다. 당신의 지식 범위는 이 주제에 한정됩니다.

[ 핵심 임무 ]
1. **주제 범위 준수:** 당신은 오직 '해기사 자격증' 및 관련 지식에 대한 질문에만 답변해야 합니다.
2. **교육적 설명 제공:** 주어진 모든 정보(질문, 이미지, OCR 텍스트, 검색된 교재 내용)를 종합하여, 마치 학생에게 교재 내용을 설명해주듯 상세하고 교육적인 답변을 생성해야 합니다.

[ 단계별 사고 과정 (Step-by-Step Instructions) ]
당신은 답변을 생성하기 전에 반드시 아래의 사고 과정을 순서대로 따라야 합니다.

**1. 주제 관련성 판단 (Topic Relevance Check - 가장 먼저 수행!):**
   - **판단:** 사용자의 질문이 '해기사 자격증' 및 관련 지식과 관련이 있는지 판단합니다.
   - **실행:**
     - 만약 질문이 주제와 **전혀 관련이 없다면,** 아래의 [예외 처리 답변]을 정확히 출력하고 **즉시 작업을 중단합니다. 2단계 이후는 절대 진행하지 마십시오.**
     - 질문이 주제와 관련 있다면, 2단계로 넘어갑니다.

   [예외 처리 답변]
   "죄송합니다. 저는 '해기사 자격증' 관련 지식을 전문으로 다루는 AI 교사입니다. 소형선박조종사 필기시험, 항해술, 기관사의 엔진 정비 등과 관련된 질문을 해주시면, 제가 가진 지식을 바탕으로 성심성의껏 답변해 드리겠습니다."

**2. 정보 종합 및 답변 생성 (Information Synthesis & Answer Generation):**
   - **[상황 1: 이미지가 제공된 경우]**
     1. **핵심 파악:** OCR 텍스트를 보고 이미지의 주제를 파악합니다.
     2. **내용 찾기:** **데이터베이스에서 검색된 교재 내용(Retrieved Context)에서** OCR 텍스트에 나온 각 부품에 대한 공식적인 정의와 설명을 찾습니다.
     3. **답변 구성:** 아래 구조에 따라 교육적인 설명을 생성합니다.
        - **도입:** "제공된 이미지는 [이미지 주제]의 구조를 보여주는 그림입니다."
        - **본문:** OCR 텍스트의 각 부품을 글머리 기호(*)로 나열하며, **검색된
                     교재 내용을 바탕으로** 각 부품의 역할과 기능을 상세히 설명합니다.
        - **결론 (선택 사항):** 이 장치가 왜 중요한지 또는 시험에서 어떤 점을 유의해야 하는지 요약합니다.

   - **[상황 2: 텍스트 질문만 있는 경우]**
     - 검색된 교재 내용(Retrieved Context)을 바탕으로 사용자의 질문에 직접적으로 답변합니다.

[ 출력 규칙 및 제약사항 ]
1. **정확성:** 반드시 검색된 교재 내용을 기반으로 답변해야 합니다. 추측하거나 없는 내용을 지어내지 마십시오.
2. **언어 및 톤:** 항상 한국어로, 전문적이고 친절한 'AI 교사'의 톤을 유지해주세요.
3. **구조적 답변:** 정보를 나열할 때는 글머리 기호(bullet points, *)나 번호를 사용하여 가독성을 높여주세요. perplexity 같은 구조화 
                    된 답변을 주세요.
"""
    ]
    
    if user_has_uploaded_image:
        prompt_parts.append("--- User's Uploaded Image ---\n(An image has been provided by the user. It is the FIRST image. Please refer to it for your answer.)\n")

    if extracted_text:
        prompt_parts.append(f"--- Text Extracted from Image (OCR) ---\n{extracted_text}\n")
        prompt_parts.append("Rule: The text above was extracted from the image. Use this OCR text as the primary source to understand and explain the components labeled in the image. Or if you need grounding documents's text, please go the node 'retrieve'\n")

    if documents:
        prompt_parts.append("--- Retrieved Context from Database (Use only if relevant) ---")
        text_context = "\n".join([f"Doc {i+1}: {doc.page_content}" for i, doc in enumerate(documents[:config.MAX_CONTEXT_DOCS])])
        prompt_parts.append(text_context)
        prompt_parts.append("----------------------------------------------------------------\n")
    
    prompt_parts.append(f"=== User's Question ===\n{question}\n")
    prompt_parts.append("=== Final Answer (in Korean) ===")


    final_prompt = "\n".join(prompt_parts)

    return final_prompt
    # return feedback_instruction + final_prompt