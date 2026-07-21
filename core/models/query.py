"""
Query model for RAG request/response schemas.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from uuid import UUID

from core.models.document import ContentType  # من الـ file اللي عملناه


class UserQuery(BaseModel):
    """اللي بييجي من المستخدم — نص بس"""
    user_query_text: str = Field(..., min_length=3, max_length=500)


class ParsedIntent(BaseModel):
    """السيستم استنتجه من السؤال"""
    original_question: str
    content_type: Optional[ContentType] = None  # ممكن ميعرفش
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


class RetrievedSource(BaseModel):
    """chunk واحد رجع من الـ VectorDB"""
    chunk_id: str
    text: str
    source_filename: str   # من أنهي PDF
    score: float           # 0.0 → 1.0
    page_number: Optional[int] = None
    content_type: Optional[ContentType] = None


class RAGResponse(BaseModel):
    """الرد النهائي"""
    answer: str                          # ← ده كان ناقص
    sources: List[RetrievedSource]
    original_question: str

