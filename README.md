<div align="center">

# HealthRAG
> 基于RAG的医疗知识库问答Web应用，支持网页对话、知识库上传、流式输出

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Web%20Backend-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)
[![Live Demo](https://img.shields.io/badge/🔗Online-Demo-purple)](https://healthrag.pages.dev)

</div>

## 📖 项目简介
本项目是**检索增强生成RAG网页端Agent**，用户上传知识库文档，基于向量检索召回参考资料，调用大模型做流式问答。
后端使用FastAPI，前端部署在pages.dev，知识库托管腾讯云CloudBase。
> 解决痛点：大模型幻觉，基于私有文档做回答，附带流式SSE对话接口。

## ✨ 功能亮点
- ✅ 文档入库：PDF/TXT文档解析、文本分块、向量化存入向量库
- ✅ 流式对话接口：SSE流式输出（就是你踩坑CORS那个！简历重点）
- ✅ Web前端页面，公网可访问演示链接
- ✅ 可切换LLM后端（MiniMax）
- ✅ 会话记忆管理

## 🏗️ 系统架构

## 🛠️ 技术栈
- 后端：Python, FastAPI, Pydantic
- RAG：文本分块、Embedding向量检索
- 部署：腾讯云CloudBase云函数、pages.dev静态前端托管
- 前端：Vue/静态网页
- 大模型：MiniMax API

## 🚀 快速启动
```bash
# 克隆仓库
git clone https://github.com/xxx/healthrag.git
cd healthrag

# 安装依赖
pip install -r requirements.txt

# 启动后端
uvicorn main:app --reload
