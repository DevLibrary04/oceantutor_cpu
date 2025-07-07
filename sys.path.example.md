import sys
import os

# 현재 파일의 절대 경로를 구하고, 그 경로의 부모 디렉토리(프로젝트 최상위)를 sys.path에 추가합니다.
# 이렇게 하면 프로젝트의 어떤 위치에 있는 모듈이든 'app.models' 와 같이 절대 경로로 임포트할 수 있습니다.
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# -------------------- 변경 전 (app/api_test.py) --------------------
# from models import User
# from database import SessionLocal
# from crud import get_user

# ... (코드 내용)


# -------------------- 변경 후 (scripts/api_test.py) --------------------
import sys
import os

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# 'app' 패키지부터 시작하는 절대 경로로 임포트 구문 수정
from app.models import User
from app.database import SessionLocal
from app.crud import get_user

# ... (코드 내용)
# 이제부터는 SessionLocal(), get_user() 등을 이전과 동일하게 사용할 수 있습니다.