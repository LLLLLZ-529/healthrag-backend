# -*- coding: utf-8 -*-
"""个性化推荐 API"""
from fastapi import APIRouter, HTTPException

from app.models.recommendation import RecommendRequest, RecommendResponse
from app.services.recommender import generate_personalized_recommendations
from app.api.students import _students_db

router = APIRouter()


@router.post("/", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """生成个性化推荐（分型驱动 + RAG + 早期预警）"""
    if req.student_id not in _students_db:
        raise HTTPException(status_code=404, detail="学生不存在，请先录入数据")
    rec = _students_db[req.student_id]
    result = generate_personalized_recommendations(
        rec["data"], rec["hi_scores"], focus=req.focus
    )
    return RecommendResponse(
        student_id=req.student_id,
        health_type=result["health_type"],
        early_warning=result["early_warning"],
        exercise=[],  # 从 LLM 推荐中解析
        nutrition=[],
        lifestyle=[],
        knowledge_references=result["knowledge_references"],
    )


@router.post("/full")
def recommend_full(req: RecommendRequest):
    """返回完整推荐结果（含 LLM 生成文本）"""
    if req.student_id not in _students_db:
        raise HTTPException(status_code=404, detail="学生不存在，请先录入数据")
    rec = _students_db[req.student_id]
    result = generate_personalized_recommendations(
        rec["data"], rec["hi_scores"], focus=req.focus
    )
    return result
