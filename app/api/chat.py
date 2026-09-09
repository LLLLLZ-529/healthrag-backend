# -*- coding: utf-8 -*-
"""AI 对话 API — 支持非流式和 SSE 流式"""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.models.chat import ChatRequest, ChatResponse, ChatMessage
from app.services.chat_engine import chat_completion, chat_stream
from app.api.students import _students_db

router = APIRouter()


@router.post("/", response_model=ChatResponse)
def chat(req: ChatRequest):
    """非流式 AI 对话"""
    student_profile = None
    if req.student_id and req.student_id in _students_db:
        rec = _students_db[req.student_id]
        student_profile = {
            "gender": rec["data"]["gender"],
            "hi_score": rec["hi_scores"].get("g1"),
            "health_type": rec["data"].get("type_3x3"),
        }
    result = chat_completion(
        messages=[m.model_dump() for m in req.messages],
        student_profile=student_profile,
        category_filter=req.category_filter,
    )
    return ChatResponse(
        message=ChatMessage(**result["message"]),
        retrieved=result["retrieved"],
        latency_s=result["latency_s"],
    )


@router.post("/stream")
async def chat_stream_api(req: ChatRequest):
    """SSE 流式 AI 对话"""
    student_profile = None
    if req.student_id and req.student_id in _students_db:
        rec = _students_db[req.student_id]
        student_profile = {
            "gender": rec["data"]["gender"],
            "hi_score": rec["hi_scores"].get("g1"),
            "health_type": rec["data"].get("type_3x3"),
        }
    return StreamingResponse(
        chat_stream(
            messages=[m.model_dump() for m in req.messages],
            student_profile=student_profile,
            category_filter=req.category_filter,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/knowledge/stats")
def knowledge_stats():
    """知识库统计信息"""
    from app.services.chat_engine import get_chunks_stats
    return get_chunks_stats()
