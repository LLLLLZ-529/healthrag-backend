# -*- coding: utf-8 -*-
"""推荐 Pydantic 模型"""
from pydantic import BaseModel, Field
from typing import Optional, List


class AssessRequest(BaseModel):
    query: str = Field(..., description="用户健康问题")
    category_filter: Optional[str] = Field(None, description="类别过滤: 运动处方/膳食营养/体测标准/健康政策")
    student_id: Optional[str] = Field(None, description="关联学号（注入画像）")


class RetrievedChunk(BaseModel):
    id: str
    category: str
    source_title: str
    content: str
    score: float
    score_type: str = Field("bm25", description="分数类型: bm25/hybrid/rerank")


class AssessResponse(BaseModel):
    query: str
    retrieved: List[RetrievedChunk] = Field(default_factory=list)
    recommendation: Optional[str] = None
    latency_s: float = 0.0


class RecommendRequest(BaseModel):
    student_id: str = Field(..., description="学号")
    focus: Optional[str] = Field(None, description="关注方向: 运动/营养/生活方式")


class RecommendResponse(BaseModel):
    student_id: str
    health_type: str = Field(..., description="语义分型")
    early_warning: Optional[dict] = None
    exercise: List[str] = Field(default_factory=list, description="运动处方推荐")
    nutrition: List[str] = Field(default_factory=list, description="膳食营养推荐")
    lifestyle: List[str] = Field(default_factory=list, description="生活方式建议")
    knowledge_references: List[dict] = Field(default_factory=list, description="知识来源")
