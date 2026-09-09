# -*- coding: utf-8 -*-
"""学生数据 API"""
from fastapi import APIRouter, HTTPException
from typing import List

from app.models.student import StudentInput, StudentResponse
from app.services.hi_engine import compute_hi_all

router = APIRouter()

# 内存存储（生产环境应替换为数据库）
_students_db: dict = {}


@router.post("/", response_model=StudentResponse)
def create_student(student: StudentInput):
    """录入学生体测数据并计算 HI"""
    hi_scores = compute_hi_all(student.model_dump())
    total_scores = {}
    for g in ["g1", "g2", "g3", "g4"]:
        ts = getattr(student, f"total_score_{g}", None)
        if ts is not None:
            total_scores[g] = ts

    result = StudentResponse(
        student_id=student.student_id,
        gender=student.gender,
        hi_scores=hi_scores,
        total_scores=total_scores,
    )
    _students_db[student.student_id] = {
        "data": student.model_dump(),
        "hi_scores": hi_scores,
        "total_scores": total_scores,
    }
    return result


@router.get("/{student_id}", response_model=StudentResponse)
def get_student(student_id: str):
    """查询学生数据"""
    if student_id not in _students_db:
        raise HTTPException(status_code=404, detail="学生不存在")
    rec = _students_db[student_id]
    return StudentResponse(
        student_id=student_id,
        gender=rec["data"]["gender"],
        hi_scores=rec["hi_scores"],
        total_scores=rec["total_scores"],
    )


@router.get("/", response_model=List[str])
def list_students():
    """列出所有学号"""
    return list(_students_db.keys())
