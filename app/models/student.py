# -*- coding: utf-8 -*-
"""学生数据 Pydantic 模型"""
from pydantic import BaseModel, Field
from typing import Optional


class StudentInput(BaseModel):
    """单学生体测输入"""
    student_id: str = Field(..., description="学号")
    gender: str = Field(..., description="性别: 男/女")
    enrollment_year: int = Field(..., description="入学年份")

    # --- g1 大一 ---
    bmi_g1: Optional[float] = Field(None, description="BMI g1")
    vital_capacity_g1: Optional[float] = Field(None, description="肺活量 g1 (ml)")
    sprint_50m_g1: Optional[float] = Field(None, description="50m跑 g1 (s)")
    standing_long_jump_g1: Optional[float] = Field(None, description="立定跳远 g1 (cm)")
    sit_and_reach_g1: Optional[float] = Field(None, description="体前屈 g1 (cm)")
    endurance_run_sec_g1: Optional[float] = Field(None, description="耐力跑 g1 (s)")
    strength_g1: Optional[float] = Field(None, description="力量 g1 (次)")
    total_score_g1: Optional[float] = Field(None, description="国标总分 g1")

    # --- g2 大二 ---
    bmi_g2: Optional[float] = None
    vital_capacity_g2: Optional[float] = None
    sprint_50m_g2: Optional[float] = None
    standing_long_jump_g2: Optional[float] = None
    sit_and_reach_g2: Optional[float] = None
    endurance_run_sec_g2: Optional[float] = None
    strength_g2: Optional[float] = None
    total_score_g2: Optional[float] = None

    # --- g3 大三 ---
    bmi_g3: Optional[float] = None
    vital_capacity_g3: Optional[float] = None
    sprint_50m_g3: Optional[float] = None
    standing_long_jump_g3: Optional[float] = None
    sit_and_reach_g3: Optional[float] = None
    endurance_run_sec_g3: Optional[float] = None
    strength_g3: Optional[float] = None
    total_score_g3: Optional[float] = None

    # --- g4 大四 ---
    bmi_g4: Optional[float] = None
    vital_capacity_g4: Optional[float] = None
    sprint_50m_g4: Optional[float] = None
    standing_long_jump_g4: Optional[float] = None
    sit_and_reach_g4: Optional[float] = None
    endurance_run_sec_g4: Optional[float] = None
    strength_g4: Optional[float] = None
    total_score_g4: Optional[float] = None


class StudentResponse(BaseModel):
    student_id: str
    gender: str
    hi_scores: dict = Field(default_factory=dict, description="各年级 HI 值")
    total_scores: dict = Field(default_factory=dict, description="各年级国标总分")
