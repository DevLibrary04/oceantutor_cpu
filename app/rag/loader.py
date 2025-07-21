import os
import re
import uuid
from typing import List, Tuple

from langchain.schema import Document

def parse_markdown_file(file_path: str) -> Tuple[List[Document], List[Document]]:
    """
    Markdown 파일을 하나 받아서, 텍스트 문서(Document) 리스트와 
    이미지 문서(Document) 리스트로 분리합니다.
    """
    print(f"  -> Markdown 파일 파싱 중: {os.path.basename(file_path)}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    text_docs: List[Document] = []
    image_docs: List[Document] = []

    chunks = re.split(r'(\!\[.*?\]\(.*?\))', content)
    
    for i, chunk in enumerate(chunks):
        image_match = re.search(r'\!\[(.*?)\]\((.*?)\)', chunk)
        
        if image_match:
            image_path = image_match.group(2)
            context = ""
            if i > 0:
                context = chunks[i-1].strip().split('\n')[-1]

            image_metadata = {
                "id": str(uuid.uuid4()),
                "image_path": image_path,
                "related_text": context,
                "source_file": file_path,
            }
            image_docs.append(Document(page_content=image_path, metadata=image_metadata))
            
        else:
            text_chunk = chunk.strip()
            if text_chunk:
                text_metadata = {
                    "id": str(uuid.uuid4()),
                    "source_file": file_path,
                }
                text_docs.append(Document(page_content=text_chunk, metadata=text_metadata))
                
    return text_docs, image_docs

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