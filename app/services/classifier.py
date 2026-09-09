# -*- coding: utf-8 -*-
"""
M1/M2: 3×3 语义分型引擎
水平等级(优良/及格/不及格) × 动态趋势(改善/平稳/退化) = 9 类型
"""
import numpy as np
from scipy import stats as sp_stats
from typing import Optional

from app.services.hi_engine import GRADES, assign_level

GRADE_NUM = {"g1": 1, "g2": 2, "g3": 3, "g4": 4}
DYNAMIC_THRESH = 0.05  # HI 差值阈值

# ---- 9 类型画像摘要（来自 M2 全量计算） ----
PROFILE_DATA = {
    "优良_改善": {"N": 3642, "ts_mean": 87.2, "slope": 0.18, "male_pct": 52.3,
                "weak": [], "advice": "保持当前良好状态，适当增加力量训练"},
    "优良_平稳": {"N": 5280, "ts_mean": 85.1, "slope": 0.01, "male_pct": 48.7,
                "weak": [], "advice": "维持现有运动习惯，注意季节性波动"},
    "优良_退化": {"N": 2103, "ts_mean": 83.5, "slope": -0.15, "male_pct": 45.1,
                "weak": ["endurance_run_sec"], "advice": "重点关注耐力下降，增加有氧训练频次"},
    "及格_改善": {"N": 4120, "ts_mean": 72.3, "slope": 0.22, "male_pct": 49.8,
                "weak": ["strength"], "advice": "改善趋势良好，继续加强薄弱项目"},
    "及格_平稳": {"N": 6950, "ts_mean": 70.8, "slope": 0.00, "male_pct": 47.2,
                "weak": ["sit_and_reach", "strength"], "advice": "需主动干预打破平台期"},
    "及格_退化": {"N": 3820, "ts_mean": 68.4, "slope": -0.19, "male_pct": 44.5,
                "weak": ["endurance_run_sec", "strength"], "advice": "退化风险较高，建议制定针对性训练计划"},
    "不及格_改善": {"N": 1850, "ts_mean": 55.2, "slope": 0.25, "male_pct": 51.6,
                "weak": ["bmi", "endurance_run_sec"], "advice": "改善势头积极，坚持当前方案"},
    "不及格_平稳": {"N": 2470, "ts_mean": 53.1, "slope": 0.00, "male_pct": 48.3,
                "weak": ["bmi", "endurance_run_sec", "strength"], "advice": "亟需专业指导，建议校医院体测门诊"},
    "不及格_退化": {"N": 1824, "ts_mean": 50.8, "slope": -0.21, "male_pct": 46.7,
                "weak": ["bmi", "vital_capacity", "endurance_run_sec", "strength"],
                "advice": "高危群体，需一对一干预+医疗筛查"},
}


def compute_slope(hi_values: dict) -> tuple:
    """用真实观测点拟合 HI 对年级的线性斜率
    hi_values: {grade: hi_score}, 只含真实观测
    返回 (slope, p_value, r_squared, n_points)
    """
    hi_vals, grades = [], []
    for g in GRADES:
        if g in hi_values and hi_values[g] is not None:
            hi_vals.append(hi_values[g])
            grades.append(GRADE_NUM[g])
    if len(hi_vals) < 2:
        return (None, None, None, len(hi_vals))
    x = np.array(grades, dtype=float)
    y = np.array(hi_vals, dtype=float)
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)
    return (round(float(slope), 6), round(float(p_value), 6),
            round(float(r_value ** 2), 6), len(hi_vals))


def assign_dynamic(slope: Optional[float], p_value: Optional[float] = None,
                   n_points: int = 0) -> Optional[str]:
    """斜率 → 动态趋势（混合方案：统计显著+分布分位）"""
    if slope is None:
        return None
    # 斜率三分位参考值（来自 M2 全量计算）
    q33, q67 = -0.0890, 0.0890
    if p_value is not None and n_points >= 3 and p_value < 0.05:
        return "改善" if slope > 0 else "退化"
    if slope < q33:
        return "退化"
    elif slope > q67:
        return "改善"
    else:
        return "平稳"


def classify_student(student_data: dict, hi_scores: dict) -> dict:
    """对单个学生执行完整 3×3 语义分型
    student_data: 含 total_score_g1/g2/g3/g4, g2_real, g4_real
    hi_scores: {g1: HI, g2: HI, ...}
    返回 {level, dynamic, type_3x3, ts_mean, slope, slope_pval, profile}
    """
    # 水平等级：国标总分均值
    scores = []
    for g in GRADES:
        ts = student_data.get(f"total_score_{g}")
        # 只用真实观测
        if g == "g2" and not student_data.get("g2_real", True):
            continue
        if g == "g4" and not student_data.get("g4_real", True):
            continue
        if ts is not None:
            scores.append(ts)
    ts_mean = float(np.mean(scores)) if scores else None
    level = assign_level(ts_mean)

    # 动态趋势
    slope, pval, r2, npts = compute_slope(hi_scores)
    dynamic = assign_dynamic(slope, pval, npts)

    # 合成类型
    if level and dynamic:
        type_3x3 = f"{level}_{dynamic}"
    else:
        type_3x3 = None

    # 画像
    profile = PROFILE_DATA.get(type_3x3, None) if type_3x3 else None

    return {
        "level": level,
        "dynamic": dynamic,
        "type_3x3": type_3x3,
        "ts_mean": round(ts_mean, 2) if ts_mean else None,
        "slope": slope,
        "slope_pval": pval,
        "slope_r2": r2,
        "slope_npts": npts,
        "profile": profile,
    }
