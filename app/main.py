# -*- coding: utf-8 -*-
"""HealthRAG FastAPI 入口"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.database import init_db
from app.api import students, classify, assess, recommend, chat

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="大学生体质健康智能推荐系统 — 基于 RAG 的知识增强评估",
)

# CORS：默认用 settings（本地 localhost）；云上通过环境变量 CORS_ORIGINS 覆盖
_cors_env = os.getenv("CORS_ORIGINS", "").strip()
_cors_origins = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env else settings.CORS_ORIGINS
)
_allow_credentials = "*" not in _cors_origins  # "*" 与 allow_credentials=True 冲突

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(students.router, prefix="/api/students", tags=["学生数据"])
app.include_router(classify.router, prefix="/api/classify", tags=["语义分型"])
app.include_router(assess.router, prefix="/api/assess", tags=["知识增强评估"])
app.include_router(recommend.router, prefix="/api/recommend", tags=["个性化推荐"])
app.include_router(chat.router, prefix="/api/chat", tags=["AI 对话"])


def run_startup():
    """供 uvicorn 与 SCF 入口复用：准备 /tmp 数据目录、从 COS 拉知识库、建表"""
    if os.getenv("RUN_ENV", "").lower() == "scf":
        from app.services.cos_sync import ensure_dirs, ensure_knowledge_dir
        ensure_dirs()
        ensure_knowledge_dir()
    init_db()


@app.on_event("startup")
def on_startup():
    run_startup()


@app.get("/api/health")
def health_check():
    return {"status": "ok", "app": settings.APP_NAME, "version": settings.APP_VERSION}
