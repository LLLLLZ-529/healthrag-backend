# -*- coding: utf-8 -*-
"""语义分型 API"""
from fastapi import APIRouter, HTTPException

from app.models.classification import ClassifyRequest, ClassifyResponse, ClassifyBatchResponse
from app.services.classifier import classify_student, PROFILE_DATA
from app.api.students import _students_db

router = APIRouter()


@router.post("/", response_model=ClassifyResponse)
def classify_student_api(req: ClassifyRequest):
    """对单个学生执行 3×3 语义分型"""
    if req.student_id not in _students_db:
        raise HTTPException(status_code=404, detail="学生不存在，请先录入数据")
    rec = _students_db[req.student_id]
    result = classify_student(rec["data"], rec["hi_scores"])
    return ClassifyResponse(
        student_id=req.student_id,
        level=result["level"],
        dynamic=result["dynamic"],
        type_3x3=result["type_3x3"],
        ts_mean=result["ts_mean"],
        slope=result["slope"],
        slope_pval=result["slope_pval"],
    )


@router.get("/distribution", response_model=ClassifyBatchResponse)
def get_distribution():
    """获取 9 类型分布画像"""
    type_dist = {k: v["N"] for k, v in PROFILE_DATA.items()}
    profiles = [
        {"type": k, **v}
        for k, v in PROFILE_DATA.items()
    ]
    return ClassifyBatchResponse(type_distribution=type_dist, profiles=profiles)


@router.get("/types")
def list_types():
    """列出所有 9 种语义分型"""
    return [
        {"type": k, "level": k.split("_")[0], "dynamic": k.split("_")[1],
         "N": v["N"], "advice": v["advice"]}
        for k, v in PROFILE_DATA.items()
    ]
