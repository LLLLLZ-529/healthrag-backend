<div align="center">

# 🏥 HealthRAG Backend

> 大学生体质健康智能推荐系统 · 基于 RAG 的知识增强评估后端

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Web%20Backend-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

</div>

基于 **检索增强生成（RAG）** 的健康评估与问答后端：管理学生体质数据，做语义分型、知识增强评估、个性化推荐，并提供 AI 对话接口。

## ✨ 功能特性

- 🎓 **学生数据管理**：`/api/students` 学生与体质数据的读写
- 🏷️ **语义分型**：`/api/classify` 体质状况语义分类
- 📚 **知识增强评估**：`/api/assess` 结合知识库检索 + 大模型生成评估结论
- 🎯 **个性化推荐**：`/api/recommend` 按分型与评估生成建议
- 💬 **AI 对话**：`/api/chat` RAG 增强问答
- 🧠 **混合检索**：BM25（默认）+ 可选稠密向量（`bge-large-zh-v1.5`）+ 可选重排（`bge-reranker-large`）
- ☁️ **COS 知识库同步**：云函数（SCF）环境启动时自动从腾讯云 COS 拉取知识库
- 🔌 **多 LLM 后端**：DashScope（qwen-plus 默认）/ DeepSeek / OpenAI 兼容接口
- 🐳 **Docker 部署**：内置 CloudBase 云托管镜像

## 🛠️ 技术栈

- **后端**：Python 3.10 · FastAPI · Pydantic v2 · SQLAlchemy · SQLite
- **RAG**：jieba + BM25；可选 faiss / sentence-transformers + bge 系列模型
- **部署**：Docker（CloudBase 云托管）/ 腾讯云 SCF + COS

## 🚀 快速开始

### 本地运行

```bash
# 1. 安装依赖（云端精简依赖）
pip install -r requirements-cloud.txt

# 2. 配置环境变量（复制 .env.example 为 .env，填入 LLM 密钥等）
export DASHSCOPE_API_KEY=你的Key        # 或 DEEPSEEK_API_KEY / OPENAI_API_KEY
export DATA_DIR=/你的/本地/数据路径      # ⚠️ 必填：默认值是 Windows 路径
export KNOWLEDGE_DIR=/你的/知识库/chunks
export VECTOR_STORE_DIR=/你的/向量库路径

# 3. 启动
uvicorn app.main:app --reload
```

> ⚠️ **重要**：`app/config.py` 的默认数据路径是 `D:\大创\...`（Windows），macOS/Linux 本地运行必须通过环境变量覆盖，否则启动会失败。

### 环境变量速查

| 变量 | 说明 | 默认 |
|---|---|---|
| `DATABASE_URL` | 数据库连接串 | `sqlite:///./healthrag.db` |
| `DATA_DIR` / `KNOWLEDGE_DIR` / `VECTOR_STORE_DIR` | 数据与知识库路径 | Windows 路径（**需覆盖**） |
| `LLM_PROVIDER` | `dashscope` / `deepseek` / openai 兼容 | `dashscope` |
| `LLM_MODEL` | 模型名 | `qwen-plus` |
| `DASHSCOPE_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` | LLM 密钥 | 空 |
| `USE_DENSE` / `USE_RERANKER` | 是否启用稠密检索/重排 | `0` |
| `CORS_ORIGINS` | 允许的前端来源（逗号分隔） | localhost:3000 |

### API 文档

启动后访问 `http://127.0.0.1:8000/docs`（Swagger UI），接口一览：

| 路由 | 说明 |
|---|---|
| `GET /api/health` | 健康检查 |
| `/api/students/*` | 学生数据 |
| `/api/classify/*` | 语义分型 |
| `/api/assess/*` | 知识增强评估 |
| `/api/recommend/*` | 个性化推荐 |
| `/api/chat/*` | AI 对话 |

## 🐳 Docker 部署（CloudBase 云托管）

```bash
docker build -t healthrag-backend .
docker run -p 8000:8000 \
  -e RUN_ENV=scf \
  -e DASHSCOPE_API_KEY=xxx \
  healthrag-backend
```

镜像默认执行 `uvicorn app.main:app --port 8000`，启动时若 `RUN_ENV=scf` 会自动从 COS 拉取知识库到 `/tmp/healthrag`。

## 📁 项目结构

```
healthrag-backend/
├── app/
│   ├── main.py           # FastAPI 入口，注册路由与 CORS
│   ├── config.py         # 配置（环境变量优先）
│   ├── api/              # 路由层：students / classify / assess / recommend / chat
│   ├── models/           # 数据模型
│   ├── services/         # 业务层：rag_engine / classifier / recommender / chat_engine / hi_engine / cos_sync
│   └── db/               # SQLAlchemy 数据库
├── Dockerfile            # CloudBase 云托管镜像
├── requirements-cloud.txt
└── .gitignore
```


## 📄 许可

MIT License（以仓库 LICENSE 文件为准）。
