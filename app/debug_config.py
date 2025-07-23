# debug_config.py
# config 파일의 설정값들을 확인하는 스크립트

import sys
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

print("=== 환경 정보 ===")
print(f"Python 버전: {sys.version}")
print(f"현재 작업 디렉토리: {os.getcwd()}")
print()

try:
    from app.rag import config
    
    print("=== Config 파일 정보 ===")
    
    # 모든 config 속성 출력
    config_attrs = [attr for attr in dir(config) if not attr.startswith('_')]
    for attr in config_attrs:
        try:
            value = getattr(config, attr)
            print(f"{attr}: {value}")
        except Exception as e:
            print(f"{attr}: 오류 - {str(e)}")
    
    print()
    print("=== 중요 설정 확인 ===")
    
    # 필수 설정들 확인
    required_configs = [
        'TEXT_EMBEDDING_MODEL',
        'DEVICE', 
        'RERANKER_MODEL',
        'LLM_MODEL',
        'MARKDOWN_FILE_PATH',
        'DB_STORAGE_PATH'
    ]
    
    for config_name in required_configs:
        try:
            value = getattr(config, config_name)
            print(f"✅ {config_name}: {value}")
        except AttributeError:
            print(f"❌ {config_name}: 설정되지 않음")
    
    print()
    print("=== 파일/디렉토리 존재 확인 ===")
    
    # 마크다운 파일 확인
    try:
        markdown_path = getattr(config, 'MARKDOWN_FILE_PATH')
        if os.path.exists(markdown_path):
            print(f"✅ 마크다운 파일 존재: {markdown_path}")
            file_size = os.path.getsize(markdown_path) / (1024*1024)  # MB
            print(f"   파일 크기: {file_size:.2f} MB")
        else:
            print(f"❌ 마크다운 파일 없음: {markdown_path}")
    except:
        print("❌ MARKDOWN_FILE_PATH 설정 확인 불가")
    
    # 저장 디렉토리 확인
    try:
        storage_path = getattr(config, 'DB_STORAGE_PATH')
        print(f"📁 DB 저장 경로: {storage_path}")
        if not os.path.exists(storage_path):
            print(f"   (디렉토리가 존재하지 않음 - 자동 생성됨)")
    except:
        print("❌ DB_STORAGE_PATH 설정 확인 불가")
    
except ImportError as e:
    print(f"❌ config 모듈을 불러올 수 없습니다: {str(e)}")
    print("프로젝트 구조를 확인해주세요.")
    
except Exception as e:
    print(f"❌ 예상치 못한 오류: {str(e)}")
    import traceback
    traceback.print_exc()

print()
print("=== 환경변수 확인 ===")
env_vars = ['HF_HOME', 'TRANSFORMERS_CACHE', 'TOKENIZERS_PARALLELISM']
for var in env_vars:
    value = os.environ.get(var)
    if value:
        print(f"✅ {var}: {value}")
    else:
        print(f"❌ {var}: 설정되지 않음")

print()
print("=== 메모리 및 디스크 공간 확인 ===")
import psutil

# 메모리 확인
memory = psutil.virtual_memory()
print(f"전체 메모리: {memory.total / (1024**3):.1f} GB")
print(f"사용 가능 메모리: {memory.available / (1024**3):.1f} GB")
print(f"메모리 사용률: {memory.percent}%")

# 디스크 공간 확인
disk = psutil.disk_usage('.')
print(f"전체 디스크 공간: {disk.total / (1024**3):.1f} GB")
print(f"사용 가능 디스크 공간: {disk.free / (1024**3):.1f} GB")