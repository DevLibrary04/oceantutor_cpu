# test_langchain_ollama.py
print("--- 테스트 시작 ---")
print("1. langchain_ollama에서 ChatOllama 임포트 시도...")
from langchain_ollama import ChatOllama
print("   -> 임포트 성공!")

try:
    print("2. ChatOllama(model='gemma3:4b') 초기화 시도...")
    llm = ChatOllama(model="gemma3:4b", temperature=0)
    print("   -> 초기화 성공!")

    print("3. llm.invoke() 호출 시도...")
    response = llm.invoke("Why is the sky blue?")
    print("   -> invoke 성공!")
    
    print("\n--- 최종 결과 ---")
    print(response)
    print("\n--- 테스트 완료 ---")

except Exception as e:
    print(f"\n!!! 에러 발생: {e}")