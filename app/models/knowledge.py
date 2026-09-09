# -*- coding: utf-8 -*-
"""知识库 Pydantic 模型"""
from pydantic import BaseModel, Field
from typing import Optional


class KnowledgeChunk(BaseModel):
    id: str
    category: str
    source_title: str = ""
    topic: str = ""
    content: str
    chunk_size_chars: int = 0
