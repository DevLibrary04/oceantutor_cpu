# from fastapi import APIRouter
# from fastapi.responses import StreamingResponse
# # from sqlmodel import Session, select  <- 이 코드는 현재 파일에서 사용되지 않으므로 주석 처리하거나 삭제해도 됩니다.
# # from app.database import get_db       <- 이 코드는 현재 파일에서 사용되지 않으므로 주석 처리하거나 삭제해도 됩니다.
# from dotenv import load_dotenv
# import google.generativeai as genai
# import os

# load_dotenv()

# router = APIRouter(prefix="/modelcall", tags=["Call Local or External Models"])

# # 1. API 키 설정
# gemApikey = os.getenv("GEMINI_APIKEY")
# genai.configure(api_key=gemApikey)

# # 2. 모델 설정 (시스템 프롬프트, 온도 등 모든 설정을 여기서 한 번에 정의)
# # 참고: gemini-2.5-flash 모델은 아직 없으므로, 현재 사용 가능한 gemini-1.5-flash로 수정했습니다.
# model = genai.GenerativeModel(
#     model_name="gemini-1.5-flash",
#     system_instruction="You are an assistant who helps the user get ready for the navigator('해기사') exam. Always answer in Korean. Keep the response below 700 characters long.",
#     generation_config={"temperature": 0.3}
# )

# async def geminiChat(user_prompt: str):
#     # 3. API 호출 방식 변경: model.generate_content 사용 및 stream=True 옵션 추가
#     modelResponse = model.generate_content(
#         user_prompt,
#         stream=True
#     )
#     for chunk in modelResponse:
#         # chunk.text가 비어있는 경우를 대비한 예외 처리
#         if chunk.text:
#             yield f"data: {chunk.text}\n\n"

# @router.get("/gemini")
# async def modelcall_root(user_prompt: str):
#     return StreamingResponse(geminiChat(user_prompt), media_type="text/event-stream")