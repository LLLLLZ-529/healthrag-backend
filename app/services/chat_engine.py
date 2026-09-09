# -*- coding: utf-8 -*-
"""
M2+: RAG 聊天引擎 — 多轮对话 + 知识库检索增强 + 流式输出
"""
import time
import json
from typing import Optional, AsyncGenerator

from app.config import llm_client_credentials, settings
from app.services.rag_engine import get_retriever, load_chunks


# ---- 知识库统计 ----
_chunks_stats = None

def get_chunks_stats() -> dict:
    global _chunks_stats
    if _chunks_stats is not None:
        return _chunks_stats
    chunks = load_chunks(settings.KNOWLEDGE_DIR)
    from collections import Counter
    cat_counter = Counter(c["category"] for c in chunks)
    _chunks_stats = {
        "total": len(chunks),
        "categories": dict(cat_counter),
        "source_files": len(set(c.get("source_file", c.get("source_title", "")) for c in chunks)),
    }
    return _chunks_stats


# ---- 纯搜索（不含 LLM）----
def search_knowledge(query: str, top_k: int = 10, category_filter: Optional[str] = None) -> list:
    """纯知识库检索，返回原始 chunks"""
    retriever = get_retriever()
    results = retriever.search(query, top_k=top_k, category_filter=category_filter)
    output = []
    for r in results:
        score = r.get("rerank_score", r.get("hybrid_score", r.get("bm25_score", 0)))
        score_type = "rerank" if "rerank_score" in r else (
            "hybrid" if "hybrid_score" in r else "bm25")
        output.append({
            "id": r["id"],
            "category": r["category"],
            "source_title": r.get("source_title", ""),
            "topic": r.get("topic", ""),
            "content": r["content"],
            "score": round(float(score), 4),
            "score_type": score_type,
        })
    return output


# ---- RAG 增强对话 ----
def build_chat_prompt(messages: list, retrieved: list, student_profile: Optional[dict] = None) -> str:
    """构建带知识库上下文的对话 prompt"""
    # 拼接检索结果
    context_parts = []
    for i, c in enumerate(retrieved, 1):
        src = c.get("source_title", c.get("source_file", "未知"))
        cat = c.get("category", "")
        context_parts.append(f"[{i}] 【{cat}·{src}】\n{c['content']}")
    context = "\n\n---\n\n".join(context_parts) if context_parts else "（未检索到相关知识）"

    # 学生画像
    profile_text = ""
    if student_profile:
        profile_text = f"""
## 学生健康画像
- 性别: {student_profile.get('gender', '未知')}
- HI 健康指数: {student_profile.get('hi_score', '未知')}
- 健康分型: {student_profile.get('health_type', '未知')}
- 薄弱指标: {student_profile.get('weak_items', '未知')}
"""

    # 从消息历史中提取最后一轮用户问题
    user_msg = ""
    for m in reversed(messages):
        if m["role"] == "user":
            user_msg = m["content"]
            break

    # 拼接历史（最近 6 轮）
    history_lines = []
    for m in messages[-6:]:
        role_cn = "用户" if m["role"] == "user" else "助手"
        history_lines.append(f"{role_cn}: {m['content']}")
    history = "\n".join(history_lines)

    system_prompt = f"""你是一名专业的大学生健康管理顾问 AI 助手。你的回答必须基于检索到的专业知识，确保建议有据可依。

## 你的能力
1. 回答体质健康相关问题（体测标准、运动处方、膳食营养、健康政策）
2. 根据学生的健康分型和薄弱指标提供个性化建议
3. 解读体测数据和健康指数

## 检索到的专业知识
{context}
{profile_text}

## 对话历史
{history}

## 要求
- 每条建议需标注知识来源（如 [1][2]）
- 语言简洁专业，适合大学生阅读
- 如果检索结果不足以回答问题，诚实说明并给出一般性建议
- 回答控制在 300-500 字以内"""

    return system_prompt


def chat_completion(messages: list, student_profile: Optional[dict] = None,
                    category_filter: Optional[str] = None) -> dict:
    """非流式聊天完成，返回 {message, retrieved, latency_s}"""
    t0 = time.time()

    # 1. 检索
    user_msg = ""
    for m in reversed(messages):
        if m["role"] == "user":
            user_msg = m["content"]
            break

    retriever = get_retriever()
    results = retriever.search(user_msg, top_k=5, category_filter=category_filter)

    # 2. 构建 prompt
    prompt = build_chat_prompt(messages, results, student_profile)

    # 3. 调用 LLM
    reply = _call_llm(prompt)
    latency = time.time() - t0

    # 4. 格式化检索结果
    retrieved = []
    for r in results:
        score = r.get("rerank_score", r.get("hybrid_score", r.get("bm25_score", 0)))
        score_type = "rerank" if "rerank_score" in r else (
            "hybrid" if "hybrid_score" in r else "bm25")
        retrieved.append({
            "id": r["id"],
            "category": r["category"],
            "source_title": r.get("source_title", ""),
            "content": r["content"][:300],
            "score": round(float(score), 4),
            "score_type": score_type,
        })

    return {
        "message": {"role": "assistant", "content": reply},
        "retrieved": retrieved,
        "latency_s": round(latency, 3),
    }


async def chat_stream(messages: list, student_profile: Optional[dict] = None,
                       category_filter: Optional[str] = None) -> AsyncGenerator[str, None]:
    """流式聊天，SSE 格式输出"""
    t0 = time.time()

    # 1. 检索
    user_msg = ""
    for m in reversed(messages):
        if m["role"] == "user":
            user_msg = m["content"]
            break

    retriever = get_retriever()
    results = retriever.search(user_msg, top_k=5, category_filter=category_filter)

    # 2. 先发送检索结果（SSE event: retrieved）
    retrieved_data = []
    for r in results:
        score = r.get("rerank_score", r.get("hybrid_score", r.get("bm25_score", 0)))
        score_type = "rerank" if "rerank_score" in r else (
            "hybrid" if "hybrid_score" in r else "bm25")
        retrieved_data.append({
            "id": r["id"],
            "category": r["category"],
            "source_title": r.get("source_title", ""),
            "content": r["content"][:200],
            "score": round(float(score), 4),
            "score_type": score_type,
        })

    yield f"event: retrieved\ndata: {json.dumps(retrieved_data, ensure_ascii=False)}\n\n"

    # 3. 构建并流式调用 LLM
    prompt = build_chat_prompt(messages, results, student_profile)

    if settings.LLM_PROVIDER == "dashscope":
        async for chunk in _stream_dashscope(prompt):
            yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
    else:
        async for chunk in _stream_openai(prompt):
            yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"

    latency = round(time.time() - t0, 3)
    yield f"event: done\ndata: {json.dumps({'latency_s': latency}, ensure_ascii=False)}\n\n"


async def _stream_dashscope(prompt: str):
    """DashScope 流式输出"""
    try:
        import dashscope
        from dashscope import Generation
        if not settings.DASHSCOPE_API_KEY:
            yield "[未设置 DASHSCOPE_API_KEY]"
            return
        dashscope.api_key = settings.DASHSCOPE_API_KEY
        # 使用 stream_mode
        responses = Generation.call(
            model=settings.LLM_MODEL,
            prompt=prompt,
            max_tokens=2048,
            temperature=0.7,
            result_format='message',
            stream=True,
        )
        for resp in responses:
            if resp.status_code == 200:
                content = resp.output.choices[0].message.content
                if content:
                    yield content
            else:
                yield f"[错误] {resp.code}: {resp.message}"
                break
    except ImportError:
        yield "[dashscope 未安装]"
    except Exception as e:
        yield f"[错误] {e}"


async def _stream_openai(prompt: str):
    """OpenAI 兼容流式输出（含 DeepSeek）"""
    try:
        from openai import AsyncOpenAI
        api_key, base_url = llm_client_credentials()
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        stream = await client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.7,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"[错误] {e}"


def _call_llm(prompt: str) -> str:
    """非流式 LLM 调用"""
    if settings.LLM_PROVIDER == "dashscope":
        try:
            import dashscope
            from dashscope import Generation
            if not settings.DASHSCOPE_API_KEY:
                return "[未设置 DASHSCOPE_API_KEY]"
            dashscope.api_key = settings.DASHSCOPE_API_KEY
            resp = Generation.call(
                model=settings.LLM_MODEL,
                prompt=prompt,
                max_tokens=2048,
                temperature=0.7,
                result_format='message',
            )
            if resp.status_code == 200:
                return resp.output.choices[0].message.content
            return f"[错误] {resp.code}: {resp.message}"
        except ImportError:
            return "[dashscope 未安装]"
        except Exception as e:
            return f"[错误] {e}"
    else:
        try:
            from openai import OpenAI
            api_key, base_url = llm_client_credentials()
            client = OpenAI(api_key=api_key, base_url=base_url)
            resp = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.7,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"[错误] {e}"
