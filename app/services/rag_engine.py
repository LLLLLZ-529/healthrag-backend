# -*- coding: utf-8 -*-
"""
M2: RAG 引擎 — 知识增强评估
BM25 + Dense 混合检索 + 可选 Rerank + LLM 生成
"""
import os
import sys
import time
import math
import json
import tempfile
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Optional

from app.config import llm_client_credentials, settings

# ---- 知识库加载 ----
def load_chunks(chunks_dir: str) -> list:
    """加载分块知识库"""
    chunks_dir = Path(chunks_dir)
    chunks = []
    if not chunks_dir.exists():
        return chunks
    for cat_dir in sorted(chunks_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        for cf in sorted(cat_dir.glob("*.md")):
            text = cf.read_text(encoding="utf-8")
            meta, body = {}, text
            if text.startswith("---"):
                parts = text.split("---", 2)
                if len(parts) >= 3:
                    for line in parts[1].strip().split("\n"):
                        if ":" in line:
                            k, v = line.split(":", 1)
                            meta[k.strip()] = v.strip().strip('"')
                    body = parts[2].strip()
            chunks.append({
                "id": str(cf.relative_to(chunks_dir)).replace("\\", "/"),
                "category": meta.get("category", cat_dir.name),
                "topic": meta.get("topic", ""),
                "source_file": meta.get("source_file", ""),
                "source_title": meta.get("source_title", ""),
                "chunk_size_chars": int(meta.get("chunk_size_chars", 0)),
                "content": body,
            })
    return chunks


# ---- BM25 ----
class BM25Retriever:
    def __init__(self, chunks):
        import jieba
        self.chunks = chunks
        self.doc_tokens = []
        self.doc_lengths = []
        for c in chunks:
            tokens = [t.strip() for t in jieba.cut(c["content"]) if len(t.strip()) > 1]
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))

        self.N = len(chunks)
        self.avgdl = sum(self.doc_lengths) / max(self.N, 1)

        self.df = defaultdict(int)
        for tokens in self.doc_tokens:
            for t in set(tokens):
                self.df[t] += 1
        self.idf = {}
        for t, df in self.df.items():
            self.idf[t] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

        self.inverted = defaultdict(list)
        for i, tokens in enumerate(self.doc_tokens):
            tf_map = defaultdict(int)
            for t in tokens:
                tf_map[t] += 1
            for t, tf in tf_map.items():
                self.inverted[t].append((i, tf))

    def search(self, query, top_k=30, category_filter=None):
        import jieba
        k1, b = 1.5, 0.75
        q_tokens = [t.strip() for t in jieba.cut(query) if len(t.strip()) > 1]
        scores = defaultdict(float)
        for qt in q_tokens:
            if qt not in self.inverted:
                continue
            idf = self.idf.get(qt, 0)
            for doc_idx, tf in self.inverted[qt]:
                dl = self.doc_lengths[doc_idx]
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self.avgdl))
                scores[doc_idx] += idf * tf_norm
        results = []
        for doc_idx, score in sorted(scores.items(), key=lambda x: -x[1]):
            c = self.chunks[doc_idx]
            if category_filter and c["category"] != category_filter:
                continue
            results.append({**c, "bm25_score": float(score)})
            if len(results) >= top_k:
                break
        return results


# ---- Dense (可选) ----
class DenseRetriever:
    def __init__(self, chunks, embed_model=None, vector_dir=None):
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer

        embed_model = embed_model or settings.EMBED_MODEL
        vector_dir = Path(vector_dir or settings.VECTOR_STORE_DIR)
        vector_dir.mkdir(parents=True, exist_ok=True)
        self.chunks = chunks

        index_path = vector_dir / "faiss.index"
        tmp_dir = Path(tempfile.gettempdir()) / "faiss_tmp"
        tmp_dir.mkdir(exist_ok=True)
        tmp_index = tmp_dir / "faiss.index"

        if index_path.exists():
            shutil.copy2(str(index_path), str(tmp_index))
            self.index = faiss.read_index(str(tmp_index))
            self.model = SentenceTransformer(embed_model)
        else:
            self.model = SentenceTransformer(embed_model)
            texts = [c["content"] for c in chunks]
            embeddings = self.model.encode(texts, batch_size=32,
                                           show_progress_bar=True,
                                           normalize_embeddings=True)
            embeddings = np.array(embeddings, dtype=np.float32)
            dim = self.model.get_embedding_dimension()
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
            faiss.write_index(self.index, str(tmp_index))
            shutil.copy2(str(tmp_index), str(index_path))

    def search(self, query, top_k=10, category_filter=None):
        import numpy as np
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)
        n_search = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(q_emb, n_search)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            c = self.chunks[idx]
            if category_filter and c["category"] != category_filter:
                continue
            results.append({**c, "dense_score": float(score)})
            if len(results) >= top_k:
                break
        return results


# ---- 混合检索 ----
class HybridRetriever:
    def __init__(self, bm25, dense=None, reranker=None, alpha=0.5):
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.alpha = alpha

    def search(self, query, top_k=5, category_filter=None):
        bm25_results = self.bm25.search(query, top_k=settings.BM25_TOP_K,
                                         category_filter=category_filter)
        if not self.dense:
            results = bm25_results[:top_k]
        else:
            dense_results = self.dense.search(query, top_k=settings.DENSE_TOP_K,
                                               category_filter=category_filter)
            all_ids = {}
            for r in bm25_results:
                all_ids[r["id"]] = {"chunk": r, "bm25_norm": 0.0, "dense_norm": 0.0}
            for r in dense_results:
                if r["id"] not in all_ids:
                    all_ids[r["id"]] = {"chunk": r, "bm25_norm": 0.0, "dense_norm": 0.0}
            bm25_scores = [r["bm25_score"] for r in bm25_results]
            max_bm25 = max(bm25_scores) if bm25_scores else 1
            dense_scores = [r["dense_score"] for r in dense_results]
            max_dense = max(dense_scores) if dense_scores else 1
            for r in bm25_results:
                all_ids[r["id"]]["bm25_norm"] = r["bm25_score"] / max(max_bm25, 1e-6)
            for r in dense_results:
                all_ids[r["id"]]["dense_norm"] = r["dense_score"] / max(max_dense, 1e-6)
            for v in all_ids.values():
                v["hybrid_score"] = self.alpha * v["bm25_norm"] + (1 - self.alpha) * v["dense_norm"]
            ranked = sorted(all_ids.values(), key=lambda x: -x["hybrid_score"])
            results = []
            for v in ranked[:top_k]:
                r = v["chunk"].copy()
                r["bm25_norm"] = v["bm25_norm"]
                r["dense_norm"] = v["dense_norm"]
                r["hybrid_score"] = v["hybrid_score"]
                results.append(r)

        if self.reranker and results:
            pairs = [(query, r["content"]) for r in results]
            scores = self.reranker.predict(pairs)
            for r, s in zip(results, scores):
                r["rerank_score"] = float(s)
            results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        return results[:top_k]


# ---- LLM 生成 ----
def generate_recommendation(query: str, retrieved: list, student_profile: dict = None) -> str:
    context_parts = []
    for i, c in enumerate(retrieved, 1):
        src = c.get("source_title", c.get("source_file", "未知"))
        cat = c.get("category", "")
        context_parts.append(f"[{i}] 【{cat}·{src}】\n{c['content']}")
    context = "\n\n---\n\n".join(context_parts)

    profile_text = ""
    if student_profile:
        profile_text = f"""
## 学生健康画像
- 性别: {student_profile.get('gender', '未知')}
- 年级: {student_profile.get('grade', '未知')}
- BMI: {student_profile.get('bmi', '未知')}
- 体质健康综合得分: {student_profile.get('hi_score', '未知')}
- 薄弱指标: {student_profile.get('weak_items', '未知')}
- 健康分型: {student_profile.get('health_type', '未知')}
"""

    prompt = f"""你是一名专业的大学生健康管理顾问。请基于以下检索到的专业知识，为大学生提供个性化的健康推荐。

{profile_text}

## 学生问题
{query}

## 检索到的专业知识
{context}

## 要求
请从以下三个方面给出具体、可操作的建议：
1. **运动处方**：推荐具体的运动类型、频率、强度、时长
2. **膳食营养**：推荐具体的饮食调整方案、营养素摄入建议
3. **生活方式**：作息、心理调适等建议

每条建议需标注知识来源（如 [1][2]），确保有据可依。语言简洁专业，适合大学生阅读。
"""

    if settings.LLM_PROVIDER == "dashscope":
        try:
            import dashscope
            from dashscope import Generation
            if not settings.DASHSCOPE_API_KEY:
                return "[SKIP] 未设置 DASHSCOPE_API_KEY"
            dashscope.api_key = settings.DASHSCOPE_API_KEY
            resp = Generation.call(model=settings.LLM_MODEL, prompt=prompt,
                                   max_tokens=2048, temperature=0.7,
                                   result_format='message')
            if resp.status_code == 200:
                return resp.output.choices[0].message.content
            return f"[ERROR] {resp.code}: {resp.message}"
        except ImportError:
            return "[SKIP] dashscope 未安装"
        except Exception as e:
            return f"[ERROR] {e}"
    else:
        try:
            from openai import OpenAI
            api_key, base_url = llm_client_credentials()
            client = OpenAI(api_key=api_key, base_url=base_url)
            resp = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048, temperature=0.7,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"[ERROR] {e}"


# ---- 单例管理 ----
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """懒加载检索器（全局单例）"""
    global _retriever
    if _retriever is not None:
        return _retriever

    chunks = load_chunks(settings.KNOWLEDGE_DIR)
    bm25 = BM25Retriever(chunks)

    dense = None
    if settings.USE_DENSE:
        dense = DenseRetriever(chunks)

    reranker = None
    if settings.USE_RERANKER:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(settings.RERANK_MODEL)

    _retriever = HybridRetriever(bm25, dense, reranker, alpha=settings.HYBRID_ALPHA)
    return _retriever
