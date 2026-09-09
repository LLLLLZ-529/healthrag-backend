# -*- coding: utf-8 -*-
"""
M3: 个性化推荐引擎 — 分型驱动的推荐 + 早期预警
"""
from typing import Optional
from app.services.hi_engine import identify_weak_items, assign_level, INDICATORS
from app.services.classifier import classify_student, PROFILE_DATA
from app.services.rag_engine import get_retriever, generate_recommendation

INDICATOR_CN = {
    "bmi": "BMI", "vital_capacity": "肺活量", "sprint_50m": "50m跑",
    "standing_long_jump": "立定跳远", "sit_and_reach": "体前屈",
    "endurance_run_sec": "耐力跑", "strength": "力量",
}

# ---- 早期预警评分（简化版 XGBoost 逻辑，基于 E3 SHAP 排名） ----
# Top 特征权重近似: HI_g2, HI_g1, slope_g1_g2, bmi_g1, sprint_50m_g1
EARLY_WARNING_WEIGHTS = {
    "HI_g2": 0.28, "HI_g1": 0.22, "slope_g1_g2": 0.18,
    "bmi_g1": 0.10, "sprint_50m_g1": 0.08, "dHI_g1g2": 0.08,
    "g1_不及格": 0.06,
}


def compute_early_warning_score(student_data: dict, hi_scores: dict) -> dict:
    """计算早期预警分数（简化版）
    返回 {score: 0-100, risk_level: 低/中/高, key_factors: [...]}
    """
    # 基础风险因子
    factors = []

    # g1 水平
    g1_level = assign_level(student_data.get("total_score_g1"))
    if g1_level == "不及格":
        factors.append(("g1 水平不及格", 30))
    elif g1_level == "及格":
        factors.append(("g1 水平及格（近切点）", 15))

    # HI 趋势
    hi_g1 = hi_scores.get("g1")
    hi_g2 = hi_scores.get("g2")
    if hi_g1 is not None:
        if hi_g1 < -1.0:
            factors.append(("HI_g1 显著偏低", 20))
        elif hi_g1 < -0.5:
            factors.append(("HI_g1 偏低", 10))

    if hi_g1 is not None and hi_g2 is not None:
        dhi = hi_g2 - hi_g1
        if dhi < -0.1:
            factors.append(("g1→g2 HI 明显下降", 25))
        elif dhi < -0.05:
            factors.append(("g1→g2 HI 小幅下降", 10))

    # 薄弱指标数量
    weak = identify_weak_items(student_data, "g1")
    if len(weak) >= 3:
        factors.append((f"薄弱指标 ≥3 项", 15))
    elif len(weak) >= 2:
        factors.append((f"薄弱指标 ≥2 项", 8))

    # 计算总分
    total = min(sum(s for _, s in factors), 100)
    if total >= 50:
        risk_level = "高"
    elif total >= 25:
        risk_level = "中"
    else:
        risk_level = "低"

    return {
        "score": total,
        "risk_level": risk_level,
        "key_factors": [{"factor": f, "contribution": s} for f, s in factors],
        "weak_items": [INDICATOR_CN.get(w, w) for w in weak],
    }


def generate_personalized_recommendations(student_data: dict, hi_scores: dict,
                                           focus: str = None) -> dict:
    """生成个性化推荐
    1. 语义分型 → 分型画像
    2. 薄弱指标识别 → 查询构造
    3. RAG 检索 → LLM 生成
    4. 早期预警
    """
    # 1. 分型
    classification = classify_student(student_data, hi_scores)
    health_type = classification.get("type_3x3", "未知")
    profile = classification.get("profile", {})

    # 2. 薄弱指标
    weak_items = identify_weak_items(student_data, "g1")
    weak_cn = [INDICATOR_CN.get(w, w) for w in weak_items]

    # 3. RAG 检索
    queries = []
    if focus == "运动" or not focus:
        for w in weak_cn:
            queries.append(f"{w}不及格怎么改善 大学生训练方法")
    if focus == "营养" or not focus:
        if "BMI" in weak_cn:
            queries.append("大学生BMI偏高如何科学减脂 膳食建议")
        queries.append("大学生体质健康膳食营养建议")
    if focus == "生活方式" or not focus:
        queries.append("大学生健康生活方式 久坐作息建议")

    retriever = get_retriever()
    all_retrieved = []
    for q in queries[:3]:  # 限制查询数
        results = retriever.search(q, top_k=3, category_filter=None)
        all_retrieved.extend(results)

    # 去重
    seen_ids = set()
    unique_retrieved = []
    for r in all_retrieved:
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            unique_retrieved.append(r)

    # 4. LLM 生成
    student_profile = {
        "gender": student_data.get("gender", "未知"),
        "grade": "大一",
        "bmi": student_data.get("bmi_g1"),
        "hi_score": hi_scores.get("g1"),
        "weak_items": ", ".join(weak_cn),
        "health_type": health_type,
    }
    main_query = f"体质健康分型为{health_type}，薄弱指标: {', '.join(weak_cn) if weak_cn else '无'}"
    recommendation = generate_recommendation(main_query, unique_retrieved[:5], student_profile)

    # 5. 早期预警
    warning = compute_early_warning_score(student_data, hi_scores)

    # 6. 基于分型画像的推荐
    profile_advice = profile.get("advice", "") if profile else ""

    return {
        "student_id": student_data.get("student_id", ""),
        "health_type": health_type,
        "classification": classification,
        "early_warning": warning,
        "profile_advice": profile_advice,
        "weak_items": weak_cn,
        "recommendation": recommendation if not recommendation.startswith("[") else None,
        "knowledge_references": [
            {
                "id": r.get("id", ""),
                "category": r.get("category", ""),
                "source_title": r.get("source_title", ""),
                "score": r.get("bm25_score", r.get("hybrid_score", 0)),
            }
            for r in unique_retrieved[:5]
        ],
    }
