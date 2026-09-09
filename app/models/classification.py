# -*- coding: utf-8 -*-
"""语义分型 Pydantic 模型"""
from pydantic import BaseModel, Field
from typing import Optional, List


class ClassifyRequest(BaseModel):
    student_id: str = Field(..., description="学号")


class ClassifyResponse(BaseModel):
    student_id: str
    level: str = Field(..., description="水平等级: 优良/及格/不及格")
    dynamic: str = Field(..., description="动态趋势: 改善/平稳/退化")
    type_3x3: str = Field(..., description="3×3 语义分型, 如 优良_改善")
    ts_mean: Optional[float] = Field(None, description="国标总分均值")
    slope: Optional[float] = Field(None, description="HI 轨迹斜率")
    slope_pval: Optional[float] = Field(None, description="斜率 p 值")


class ClassifyBatchResponse(BaseModel):
    type_distribution: dict = Field(default_factory=dict, description="9 类型人数分布")
    profiles: List[dict] = Field(default_factory=list, description="各类画像摘要")
