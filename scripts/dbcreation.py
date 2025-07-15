import sys
import os

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from sqlmodel import SQLModel
from app.database import engine
from app import models


def main():
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
