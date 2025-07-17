import sys
import os

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import json
from pathlib import Path
from app.models import (
    GichulSet,
    GichulQna,
    GichulSetType,
    GichulSetInning,
    GichulSetGrade,
    GichulSubject,
)
from app.database import engine
from app.core.config import settings
from dbcreation import main as dropcreate
from sqlmodel import Session
from sqlalchemy.engine import Engine
import re


def normalize_gichulset_name(name: str):
    """
    비일관적인 문자열을 정제
    """
    normalized_type = ""
    if "기관사" in name:
        normalized_type = "기관사"
    elif "항해사" in name:
        normalized_type = "항해사"
    elif "소형" in name:
        normalized_type = "소형선박조종사"
        return (normalized_type, "0")
    else:
        pass
    split = re.split(r"[\.\s?]", name)
    return (normalized_type, split[1].rstrip("급"))


def normalize_gichulqna_qsub(qsub: str):
    split = re.split(r"[\.\s?]", qsub)
    return split[-1]


def insertData(engine: Engine, json_file_path: Path):
    with open(
        json_file_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)
        subject = data["subject"]
        settype, grade = normalize_gichulset_name(subject["name"])
        year = subject["year"]
        inning = subject["inning"]
        with Session(engine) as session:
            set = GichulSet(
                type=GichulSetType(settype),
                grade=GichulSetGrade(grade),
                year=int(year),
                inning=GichulSetInning(str(inning)),
            )
            session.add(set)
            for types in subject["type"]:
                qsubject = normalize_gichulqna_qsub(types["string"])
                for q in types["questions"]:
                    if not q["questionsStr"].strip():
                        continue
                    qna = GichulQna(
                        qnum=int(q["num"]),
                        subject=GichulSubject(qsubject),
                        questionstr=q["questionsStr"],
                        ex1str=q["ex1Str"],
                        ex2str=q["ex2Str"],
                        ex3str=q["ex3Str"],
                        ex4str=q["ex4Str"],
                        answer=q["answer"],
                        gichulset=set,
                    )
                    session.add(qna)
            session.commit()


# 디렉토리 순회
path = settings.BASE_PATH
if __name__ == "__main__":
    dropcreate()
    for questionfolder in path.glob("*/*"):
        json_file_path = questionfolder / f"{questionfolder.name}.json"
        if not json_file_path.exists():
            print(f"{json_file_path} 존재하지 않음")
            continue

        try:
            insertData(engine, json_file_path)

        except Exception as e:

            print(f"오류: {e} at {json_file_path.name}")
