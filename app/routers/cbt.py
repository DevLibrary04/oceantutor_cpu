from typing import Annotated, List
from fastapi import APIRouter, Depends, status, HTTPException, Query
from sqlmodel import Session, select
from .solve import path_getter, dir_maker
from ..database import get_db
from ..models import (
    GichulQna,
    GichulSet,
    GichulSetGrade,
    GichulSetInning,
    GichulSetType,
    GichulSubject,
)
from pathlib import Path
from collections import defaultdict
import os, re, random
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/cbt", tags=["Randomly Mixed Questions"])

base_path_str = os.getenv("BASE_PATH")
if base_path_str is None:
    raise ValueError("BASE_PATH not set")
base_path = Path(base_path_str)


def cbt_imgpath_getter(set: GichulSet):
    directory = dir_maker(
        year=str(set.year), license=set.type, level=set.grade, round=set.inning
    )
    return directory


@router.get("/")
def get_one_random_qna_set(
    license: GichulSetType,
    level: GichulSetGrade,
    *,
    subjects: List[GichulSubject] = Query(),
    db: Annotated[Session, Depends(get_db)],
):
    try:
        sets = db.exec(
            select(GichulSet).where(GichulSet.type == license, GichulSet.grade == level)
        ).all()  # The list of 12 GichulSet objects
        dic = defaultdict(list)
        random_set = defaultdict()
        path_dict = {}
        for set in sets:  # 12개 기출셋 순회
            dir = cbt_imgpath_getter(set)  # 해당 회차 폴더 정보
            path_dict[set.id] = path_getter(
                dir
            )  # 해당 회차 폴더 속 이미지 파일들 경로 -> path_dict[셋id] = {"@pic땡땡": "경로정보"}
            for qna in set.qnas:  # qna객체 순회
                if qna.questionstr and qna.ex1str:
                    joined_text = " ".join([qna.questionstr, qna.ex1str])
                    if joined_text not in dic[qna.subject]:
                        dic[qna.subject].append(
                            qna
                        )  # 문제+선택지1번이 다른 문항만 추가 Append only question that isn't duplicate
        for subject in subjects:
            random_qnas = random.sample(dic[subject], 25)  # 과목별로 25개 뽑기

            qnas_as_dicts = [
                qna.model_dump() for qna in random_qnas
            ]  # solve 속 이미지 경로 뽑는 로직과 대동소이
            pic_marker_reg = re.compile(r"@(\w+)")

            for idx, qna_dict in enumerate(qnas_as_dicts):
                qna_dict["qnum"] = idx + 1
                full_text = " ".join(
                    qna_dict.get(key, " ")
                    for key in ["questionstr", "ex1str", "ex2str", "ex3str", "ex4str"]
                )
                found_pics = pic_marker_reg.findall(full_text)
                if found_pics:
                    img_paths = [
                        path_dict[qna_dict["gichulset_id"]][pic_name]
                        for pic_name in found_pics
                        if pic_name in path_dict[qna_dict["gichulset_id"]]
                    ]
                    qna_dict["imgPaths"] = img_paths
            random_set[subject] = qnas_as_dicts
        return random_set
    except Exception as e:
        print(e)
        raise HTTPException(status_code=418, detail="teapot here")  # 예외처리 미루기
