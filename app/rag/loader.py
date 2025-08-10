import os
import re
import uuid
from typing import List, Tuple

from langchain.schema import Document

def parse_markdown_file(file_path: str) -> Tuple[List[Document], List[Document]]:
    """
    Markdown 파일을 하나 받아서,
    텍스트와 이미지를 '연결'하여
    텍스트 문서 리스트를 생성함.

    이미지 자체를 위한 문서는 만들지 않음.
    """
    print(f"  -> Markdown 파일 파싱 중: {os.path.basename(file_path)}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 이미지를 기준으로 텍스트를 나눈다.
    # 이렇게 하면 각 텍스트 덩어리는 최대 하나의 이미지만을 참조하게 된다.
    chunks = re.split(r'(\!\[.*?\]\(.*?\))', content)


    text_docs: List[Document] = []
    # 이전 텍스트를 임시 저장
    previous_text = ""
    
    for chunk in chunks:
        image_match = re.search(r'\!\[(.*?)\]\((.*?)\)', chunk)
        
        if image_match:
            # 이미지 태그를 만났을떄
            image_path = image_match.group(2).strip()

            # 바로 직전에 있던 텍스트 덩어리에 이미지 경로 정보를 메타데이터로 추가합니다.
            if previous_text:
                text_metadata = {
                    "id": str(uuid.uuid4()),
                    "source_file": file_path,
                    "image_path": os.path.join(os.path.dirname(file_path), '..', image_path) # 상대 경로를 절대 경로로 계산
                }
                text_docs.append(Document(page_content=previous_text, metadata=text_metadata))
                previous_text = "" # 사용했으니 초기화 
            
        else:
            # 텍스트 덩어리를 만났을 떄
            text_chunk = chunk.strip()
            if text_chunk:
                previous_text += "\n" + text_chunk
                
    # 마지막에 남은 텍스트 덩어리가 있다면 추가
    if previous_text:
        text_metadata = {
            "id": str(uuid.uuid4()),
            "source_file": file_path,
        }
        text_docs.append(Document(page_content=previous_text, metadata=text_metadata))
    
    # 이제 텍스트 문서만 반환하고, 이미지 문서는 빈 리스트를 반환합니다.
    return text_docs, []

def load_markdown_documents(file_path: str) -> Tuple[List[Document], List[Document]]:
    """지정된 경로의 Markdown 파일을 로드하고 파싱합니다."""
    print("\n--- Markdown 파일 처리 시작 ---")
    if not os.path.exists(file_path):
        print(f"경고: {file_path} 에서 .md 파일을 찾을 수 없습니다!")
        return [], []
    
    text_documents, image_documents = parse_markdown_file(file_path)

    print("\n--- 최종 데이터 처리 결과 ---")
    print(f"총 텍스트 문서: {len(text_documents)}개")
    print(f"총 이미지 문서: {len(image_documents)}개")
    
    return text_documents, image_documents