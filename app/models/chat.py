# -*- coding: utf-8 -*-
"""聊天 Pydantic 模型"""
from pydantic import BaseModel, Field
from typing import Optional, List


class ChatMessage(BaseModel):
    role: str = Field(..., description="角色: user/assistant/system")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="对话历史")
    student_id: Optional[str] = Field(None, description="关联学号（注入画像）")
    category_filter: Optional[str] = Field(None, description="知识库类别过滤")
    stream: bool = Field(False, description="是否流式输出")


class ChatResponse(BaseModel):
    message: ChatMessage
    retrieved: List[dict] = Field(default_factory=list, description="检索到的知识块")
    latency_s: float = 0.0
