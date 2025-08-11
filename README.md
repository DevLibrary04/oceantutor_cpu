
# 🛥️ OceanTutor: AI 기반 해기사 자격증 튜터

KDT01. 마린소프트 해기사 튜터 프로젝트  

AI와 함께 항해의 꿈을 현실로!  
해기사 수험생을 위한 차세대 멀티모달 학습 보조 시스템

---

## 📝 목차
- 프로젝트 개요
- 핵심 기능
- 시스템 아키텍처
- 기술 스택
- 주요 도전 과제 및 해결 과정
- 설치 및 실행 방법
- 디렉토리 구조
- 향후 개선 과제

---

## 📌 프로젝트 개요
OceanTutor는 해기사 자격증 시험을 준비하는 수험생들을 위한 AI 기반 질의응답 웹 서비스입니다.  
텍스트 기반의 이론 질문뿐만 아니라, 실제 기출문제에 등장하는 흐릿하거나 일부 정보가 가려진 이미지에 대해서도 깊이 있는 해설을 제공하는 것을 목표로 합니다.

본 프로젝트는 복잡하고 다양한 형태의 질문(텍스트, 이미지, 복합)을 효과적으로 처리하기 위해, 최신 멀티모달 LLM(Google Gemini 2.5 Flash)을 중심으로 강건한 카테고리 분류 시스템과 지능형 RAG(검색 증강 생성) 파이프라인을 LangGraph를 통해 유기적으로 결합하여 구축되었습니다.

---

## ✨ 핵심 기능
- **텍스트 기반 질의응답**: 해사법규, 항해술, 기관학 등 이론에 대한 서술형 질문에 RAG를 통해 정확한 답변 제공  
- **이미지 문제 해설**: 기출문제 이미지 업로드 시, 이미지 종류 자동 분류 후 정답 다이어그램과 비교 분석하여 상세 해설 생성  
- **강건한 이미지 인식**: 흐릿·절단·색상 변화가 있는 이미지도 pHash 클러스터링 기반 카테고리 분류로 안정적 식별  
- **지능형 RAG 파이프라인**:
  - 멀티모달 VQA를 통해 이미지에서 검색 키워드 추출  
  - 내부 DB 검색 실패 시 Tavily 웹 검색으로 자동 전환  
  - BGE-M3 Reranker로 검색 정확도 향상  
- **동적·유연한 응답**: 정보 부족 시 RAG 단계를 건너뛰어 LLM이 현재 정보만으로 최선의 답변 생성  

---

## 🏗️ 시스템 아키텍처
```markdown
graph TD
    A[사용자 요청<br>(질문 텍스트 + 이미지?)] --> B{이미지 포함 여부};

    B -- No --> F[2. 검색어 생성];
    B -- Yes --> C[1A. 이미지 카테고리 분류<br>(pHash Clustering)];
    C --> D[1B. 멀티모달 VQA<br>(Gemini)];
    D --> E{VQA 키워드 추출 성공?};

    E -- Yes --> F;
    F --> G[3. RAG: 텍스트 DB 검색<br>(ChromaDB)];
    G --> H[4. 문서 품질 검수<br>(BGE Reranker)];
    H --> I{관련 문서 찾음?};

    I -- Yes --> K[6. 최종 답변 생성<br>(Gemini)];
    I -- No --> J[5. Fallback: 웹 검색<br>(Tavily)];
    J --> K;
    E -- No --> K;

    subgraph "Image Processing"
        C
        D
    end

    subgraph "RAG Pipeline"
        F
        G
        H
        J
    end

    K --> L[최종 답변];
````

---

## 🛠️ 기술 스택

| 구분                   | 기술                      | 설명                         |
| -------------------- | ----------------------- | -------------------------- |
| **Backend**          | FastAPI, Uvicorn        | 비동기 웹 프레임워크 및 ASGI 서버      |
| **AI Orchestration** | LangGraph               | 복잡한 AI 에이전트 및 파이프라인 구축     |
| **LLM & Vision**     | Google Gemini 2.5 Flash | 핵심 추론 및 멀티모달 VQA 엔진        |
| **Embedding**        | BAAI/bge-m3             | 텍스트 임베딩 모델                 |
| **Reranker**         | BAAI/bge-reranker-v2-m3 | 검색 결과 재순위화 모델              |
| **Vector DB**        | ChromaDB                | 텍스트 임베딩 저장 및 검색            |
| **Web Search**       | Tavily                  | RAG 실패 시 Fallback 웹 검색 API |
| **Image Matching**   | ImageHash (pHash)       | 시각적 유사도 기반 이미지 분류          |
| **Computer Vision**  | OpenCV, EasyOCR, YOLOv8 | 이미지 전처리, OCR, 객체 탐지        |
| **Environment**      | Conda                   | Python 가상환경 관리             |

---

## 💡 주요 도전 과제 및 해결 과정

**도전 1: 흐릿한 이미지 카테고리 식별 문제**

* CLIP 기반 유사도 모델이 색상·해상도 차이로 실패 → pHash 기반 지문 생성 + 사전 매핑으로 해결

**도전 2: 이미지 내 포인터 추출 불안정성**

* YOLO/OCR 방식 실패율 높음 → Gemini VQA로 두 이미지 비교 후 키워드 직접 추출

**도전 3: RAG 실패 시 답변 품질 저하**

* 조건부 분기로 VQA 실패 시 RAG/Web 검색 건너뛰고 LLM 직행 → 불확실성 인지한 솔직한 답변 제공

---

## 🚀 설치 및 실행 방법

```bash
# 1. Git 리포지토리 클론
git clone https://github.com/your-username/oceantutor-test.git
cd oceantutor-test

# 2. Conda 가상환경 생성 및 활성화
conda env create -f environment.yml
conda activate oceantutor

# 3. 환경 변수 설정 (.env)
GOOGLE_API_KEY="your_google_api_key"
TAVILY_API_KEY="your_tavily_api_key"

# 4. 데이터/모델 준비
# data/images/red_images/에 정답 이미지
# data/images/problem_images/에 문제 이미지
# models/yolo_best.pt 모델 파일 배치
# data/final_final_report.md 텍스트 데이터 배치

# 5. (최초 실행 시) ChromaDB 벡터스토어 생성

# 6. 애플리케이션 실행
uvicorn app.main:app --reload
```

API 문서: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📁 디렉토리 구조

```plaintext
oceantutor-test/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── ocr_service.py
│   ├── routers/
│   │   └── rag.py
│   ├── services/
│   │   ├── image_matching_service.py
│   │   └── rag_service.py
│   └── rag/
│       ├── config.py
│       ├── prompt_templates.py
│       └── rag_pipeline.py
├── data/
│   ├── final_final_report.md
│   └── images/
│       ├── problem_images/
│       └── red_images/
├── models/
│   └── yolo_best.pt
├── .db_storage/
├── .env
└── README.md
```

---


## 🔮 향후 개선 과제

* **Metric Learning 도입**: 의미적 유사도 학습 기반 카테고리 분류 향상
* **사용자 피드백 루프**: 좋아요/싫어요 기반 RAG 개선
* **UI/UX 개선**: 질문/답변 히스토리 관리 기능 추가

