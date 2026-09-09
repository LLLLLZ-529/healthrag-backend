# -*- coding: utf-8 -*-
"""
M0: HI 计算引擎 — 单学生 & 批量模式
7 指标方向校正 → 性别分层 Z-score → 国标权重合成
参考统计量来自 M0 全量计算 (N≈36059)
"""
import numpy as np
from typing import Optional

# ---- 常量（与 m0_pipeline.py 完全一致） ----
INDICATORS = ["bmi", "vital_capacity", "sprint_50m", "standing_long_jump",
              "sit_and_reach", "endurance_run_sec", "strength"]
GRADES = ["g1", "g2", "g3", "g4"]

WEIGHTS = {"bmi": 0.15, "vital_capacity": 0.15, "sprint_50m": 0.20,
           "standing_long_jump": 0.10, "sit_and_reach": 0.10,
           "endurance_run_sec": 0.20, "strength": 0.10}

DIRECTION = {"vital_capacity": +1, "standing_long_jump": +1, "sit_and_reach": +1,
             "strength": +1, "sprint_50m": -1, "endurance_run_sec": -1}

BMI_LO, BMI_HI = 18.5, 23.9

# ---- 性别分层参考统计量（来自 M0 全量计算） ----
# 格式: {gender: {indicator: (mean, std)}} — 方向校正后的值
REF_STATS = {
    "男": {
        "bmi":               (  -0.349,  1.563),
        "vital_capacity":   (4217.800, 721.400),
        "sprint_50m":        (  -7.350,  0.580),
        "standing_long_jump": (226.500, 19.300),
        "sit_and_reach":     (  14.200,  6.400),
        "endurance_run_sec": (-234.500, 28.600),
        "strength":          (  13.500,  5.300),
    },
    "女": {
        "bmi":               (  -0.310,  1.420),
        "vital_capacity":   (2860.200, 530.500),
        "sprint_50m":        (  -8.950,  0.680),
        "standing_long_jump": (174.300, 16.200),
        "sit_and_reach":     (  17.100,  5.800),
        "endurance_run_sec": (-228.300, 24.500),
        "strength":          (  26.800,  8.200),
    },
}


def direction_correct_single(student_data: dict, grade: str) -> dict:
    """对单个学生做方向校正，返回 {indicator: corrected_value}"""
    corrected = {}
    for ind in INDICATORS:
        val = student_data.get(f"{ind}_{grade}")
        if val is None:
            continue
        if ind == "bmi":
            dev = max(BMI_LO - val, 0) + max(val - BMI_HI, 0)
            corrected[ind] = -dev
        else:
            corrected[ind] = DIRECTION[ind] * val
    return corrected


def compute_hi_single(student_data: dict, grade: str) -> Optional[float]:
    """计算单个学生单个年级的 HI 值"""
    gender = student_data.get("gender")
    if gender not in REF_STATS:
        return None
    corrected = direction_correct_single(student_data, grade)
    if len(corrected) < 7:  # 需要全部 7 指标
        return None
    ref = REF_STATS[gender]
    hi = 0.0
    for ind in INDICATORS:
        if ind not in corrected:
            return None
        mean, std = ref[ind]
        z = (corrected[ind] - mean) / std if std > 0 else 0
        hi += WEIGHTS[ind] * z
    return round(hi, 6)


def compute_hi_all(student_data: dict) -> dict:
    """计算单个学生所有年级的 HI"""
    result = {}
    for g in GRADES:
        hi = compute_hi_single(student_data, g)
        if hi is not None:
            result[g] = hi
    return result


def assign_level(total_score: Optional[float]) -> Optional[str]:
    """国标总分 → 水平等级"""
    if total_score is None:
        return None
    if total_score >= 80:
        return "优良"
    elif total_score >= 60:
        return "及格"
    else:
        return "不及格"


def identify_weak_items(student_data: dict, grade: str = "g1") -> list:
    """识别薄弱指标（低于该性别参考均值 1 个标准差以上）"""
    gender = student_data.get("gender")
    if gender not in REF_STATS:
        return []
    corrected = direction_correct_single(student_data, grade)
    weak = []
    ref = REF_STATS[gender]
    for ind in INDICATORS:
        if ind not in corrected:
            continue
        mean, std = ref[ind]
        z = (corrected[ind] - mean) / std if std > 0 else 0
        if z < -1.0:
            weak.append(ind)
    return weak
