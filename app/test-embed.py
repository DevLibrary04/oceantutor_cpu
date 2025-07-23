# test_embedding.py
# 이 스크립트로 임베딩 모델이 제대로 로딩되는지 테스트하세요

import os
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# 환경변수 설정
os.environ['HF_HOME'] = './hf_cache'
os.environ['TRANSFORMERS_CACHE'] = './transformers_cache'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

load_dotenv()

# config에서 모델명 가져오기 (실제 config 파일 경로에 맞게 수정)
try:
    from app.rag import config
    model_name = config.TEXT_EMBEDDING_MODEL
    device = config.DEVICE
except:
    # config를 불러올 수 없는 경우 기본값 사용
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 가벼운 테스트 모델
    device = "cpu"

print(f"테스트할 모델: {model_name}")
print(f"사용할 디바이스: {device}")

try:
    start_time = time.time()
    print("임베딩 모델 로딩 시작...")
    
    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    load_time = time.time() - start_time
    print(f"모델 로딩 완료! ({load_time:.2f}초)")
    
    # 간단한 테스트
    test_text = "This is a test sentence."
    print(f"테스트 텍스트: {test_text}")
    
    start_time = time.time()
    result = embedding.embed_query(test_text)
    embed_time = time.time() - start_time
    
    print(f"임베딩 생성 완료! ({embed_time:.2f}초)")
    print(f"임베딩 차원: {len(result)}")
    print(f"임베딩 첫 5개 값: {result[:5]}")
    
    print("✅ 임베딩 모델 테스트 성공!")
    
except Exception as e:
    print(f"❌ 오류 발생: {str(e)}")
    print(f"오류 타입: {type(e).__name__}")
    
    # 더 자세한 오류 정보
    import traceback
    traceback.print_exc()