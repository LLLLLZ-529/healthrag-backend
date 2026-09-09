# -*- coding: utf-8 -*-
"""知识增强评估 API (RAG) + 纯搜索端点"""
from fastapi import APIRouter
import time

from app.models.recommendation import AssessRequest, AssessResponse, RetrievedChunk
from app.services.rag_engine import get_retriever, generate_recommendation
from app.services.chat_engine import search_knowledge
from app.api.students import _students_db

router = APIRouter()


@router.post("/search")
def search_only(req: AssessRequest):
    """纯知识库检索（不含 LLM），返回原始 chunks"""
    t0 = time.time()
    cat_filter = req.category_filter if req.category_filter else None
    results = search_knowledge(req.query, top_k=10, category_filter=cat_filter)
    latency = time.time() - t0
    return {
        "query": req.query,
        "retrieved": results,
        "total": len(results),
        "latency_s": round(latency, 3),
    }


@router.post("/", response_model=AssessResponse)
def assess_query(req: AssessRequest):
    """基于 RAG 的知识增强评估（检索 + LLM 生成）"""
    t0 = time.time()
    retriever = get_retriever()
    results = retriever.search(req.query, top_k=5, category_filter=req.category_filter)

    # 如果关联学生，注入画像
    student_profile = None
    if req.student_id and req.student_id in _students_db:
        rec = _students_db[req.student_id]
        student_profile = {
            "gender": rec["data"]["gender"],
            "hi_score": rec["hi_scores"].get("g1"),
            "health_type": rec["data"].get("type_3x3"),
        }

    recommendation = generate_recommendation(req.query, results, student_profile)
    latency = time.time() - t0

    retrieved_chunks = []
    for r in results:
        score = r.get("rerank_score", r.get("hybrid_score", r.get("bm25_score", 0)))
        score_type = "rerank" if "rerank_score" in r else (
            "hybrid" if "hybrid_score" in r else "bm25")
        retrieved_chunks.append(RetrievedChunk(
            id=r["id"],
            category=r["category"],
            source_title=r.get("source_title", ""),
            content=r["content"][:300],
            score=round(float(score), 4),
            score_type=score_type,
        ))

    return AssessResponse(
        query=req.query,
        retrieved=retrieved_chunks,
        recommendation=recommendation if not recommendation.startswith("[") else None,
        latency_s=round(latency, 3),
    )
