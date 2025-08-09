from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from pydantic.v1 import BaseModel, Field
from io import BytesIO
from PIL import Image
import base64
from . import config

# --- 1. GradeDocuments 클래스 정의 ---
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="문서가 질문과 관련이 있으면 'yes', 없으면 'no'")

# --- 2. grade_prompt (문서 평가용 프롬프트) ---
grade_prompt_str = """당신은 특정 주제에 대한 전문가입니다. 사용자의 질문과 관련된 문서 뭉치를 선별하는 임무를 받았습니다.
문서가 사용자의 질문에 직접적으로 답변하는 데 도움이 될 수 있는지 여부를 고려하여 'yes' 또는 'no'로 점수를 매겨주세요.
질문에 대한 답변이 문서에 포함되어 있나요?

[사용자 질문]:
{question}

[문서 내용]:
{document}
"""
grade_prompt = ChatPromptTemplate.from_template(grade_prompt_str)

# --- 3. rewrite_prompt (웹 검색어 재작성용 프롬프트) ---
rewrite_prompt_str = """당신은 질문 재작성 전문가입니다. 사용자의 질문을 검색 엔진에 더 적합한, 명확하고 간결한 형태로 변환해주세요.
질문의 핵심 의도는 유지하면서, 검색에 용이한 키워드 중심으로 재구성해야 합니다.

[원본 질문]:
{question}
"""
rewrite_prompt = ChatPromptTemplate.from_template(rewrite_prompt_str)

# 3. 최종 검색어
create_final_query_prompt_str = """당신은 사용자의 의도를 파악하여 최적의 검색어를 생성하는 AI입니다.
사용자의 원본 질문과, 이미지 분석을 통해 얻은 추가적인 키워드를 조합하여,
우리의 내부 교재 데이터베이스를 검색하기 위한 가장 효과적인 단일 검색어를 생성해주세요.
검색어는 자연스러운 질문 형태가 좋습니다.

[사용자 원본 질문]:
{user_question}

[이미지 분석으로 추출된 핵심 키워드]:
{vqa_keyword}

[생성할 검색어 예시]:
- 만약 키워드가 '러더 암'이라면 -> "선박의 러더 암의 구조와 역할"
- 만약 키워드가 '횡요(Rolling)'라면 -> "선박의 횡요 현상 원인과 복원력"

위 예시를 참고하여, 주어진 정보에 가장 적합한 검색어를 생성하세요.
"""
create_final_query_prompt = ChatPromptTemplate.from_template(create_final_query_prompt_str)

# 최종 답변 생성
def create_multimodal_prompt(state: Dict) -> List[Any]:
    """GraphState를 입력받아서, 리스트 [text_prompt, image1, image2]를 반환한다"""
    question = state.get("question", "") 
    documents = state.get("documents", [])
    user_image_b64 = state.get("uploaded_image_b64")
    ref_image_b64 = state.get("matched_reference_image_b64")
    final_query = state.get("final_query", "") 
    
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc in documents])
    
    base_prompt = f"""
    당신은 해기사(소형선박조종사, 항해사, 기관사) 자격증 시험을 전문으로 가르치는 AI 교사입니다. 당신의 임무는 주어진 모든 정보를 종합하여, 학생의 질문에 대해 명확하고, 친절하며, 교육적인 답변을 생성하는 것입니다.
    답변은 반드시 아래에 제공되는 **[참고 정보]**를 최우선으로 활용하여 작성해야 합니다.

    **[학생의 질문]**
    {question}
    """
    
    info_parts = ["\n---", "**[참고 정보]**"]
    
    # Case 1: 이미지가 있는 경우
    if user_image_b64 and ref_image_b64:
        info_parts.append("""
      '문제 이미지'와 '정답 이미지'를 시각적으로 비교 분석하십시오. 학생의 질문은 이 이미지들과 관련이 있습니다. 
      
      [ 페르소나 (당신의 역할) ]
      당신은 '해기사 자격증' 관련 이미지 비교 분석 전문가입니다.
      
      [ 주어진 정보 ]
      - 첫 번째 이미지: 사용자가 질문과 함께 업로드한 '문제 이미지'입니다.
      - 두 번째 이미지: 우리가 데이터베이스에서 찾아낸, 문제와 가장 유사한 '정답 이미지(원본)'입니다.
      - 검색된 교재 내용: 이미지와 관련된 추가적인 텍스트 설명입니다.
      
      [ 핵심 임무 ]
      1. '문제 이미지'와 '정답 이미지'를 시각적으로 비교하여 차이점과 공통점을 파악하세요.
      2. 사용자의 질문('{question}')에 답하기 위해, **두 이미지를 비교한 내용**을 핵심 근거로 사용하세요.
      3. 답변 시작 시, "제공된 이미지는 **[정답 이미지의 주제]** 에 대한 것으로 보입니다." 와 같이 언급하여, 비교 분석할 것임을 명확히 하시고 검색된 내용을 통해 추론하세요.
      4. '검색된 교재 내용'을 활용하여 각 부분의 명칭이나 기능에 대한 설명을 더욱 상세하고 전문적으로 만드세요.
      5. 최종 답변은 한국어로, 친절하고 교육적인 어조로 작성해주세요.
      """)
    # 정답이미지가 없는 경우
    elif user_image_b64:
        info_parts.append("1. '문제 이미지'는 소형선박조종사 및 해양 항해사와 관련된 이미지입니다. 이 주제에 맞게 시각적 내용을 분석하십시오. 학생의 질문은 이 이미지와 관련이 있습니다.")
    
    
    # Case 2: RAG 또는 웹 검색 결과가 있는 경우
    if documents:
        info_parts.append(f"2. 아래 '검색된 교재 내용'을 참고하십시오.\n(검색어: '{final_query}')\n\n**[검색된 교재 내용]**\n{context_text}")
    else:
        # 이미지가 있는데 RAG 결과가 없는 경우 (VQA 실패 등)
        if user_image_b64:
            info_parts.append("2. 참고할 만한 교재 내용이 검색되지 않았습니다. 이미지와 질문 내용을 바탕으로 최선을 다해 답변하십시오. 단, 소형선박조종사 및 항해사와 관련되어 답변을 하십시오.")
        # 텍스트 질문인데 RAG/웹 결과가 모두 없는 경우
        else:
             info_parts.append("2. 내부 교재와 웹 검색에서 관련 정보를 찾을 수 없었습니다. 당신의 기본 지식을 바탕으로 답변하되, 정보의 출처가 불분명함을 명시해주십시오.")

    final_prompt_text = base_prompt + "\n".join(info_parts) + "\n---\n\n**[최종 지시사항]**\n위 모든 정보를 종합하여 학생의 질문에 대한 최종 답변을 생성하십시오."
    
    
    
    # 4. 최종 메시지 콘텐츠 리스트 생성
    message_content = [{"type":"text", "text": final_prompt_text}]
    
    if user_image_b64:
        message_content.append({
            "type": "image_url",
            "image_url": f"data:image/png;base64,{user_image_b64}"
        })

    if ref_image_b64:
        message_content.append({
            "type": "image_url",
            "image_url": f"data:image/png;base64,{ref_image_b64}"
        })
            
    return message_content