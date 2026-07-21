"""
Document model for RAG documents schemas.
"""

# core/models/document.py
#
# ============================================================
# إزاي بنيت الـ file ده؟
#
# ٣ أسئلة بس بنجاوب عليهم:
#   ١. إيه اللي بييجي من المستخدم؟      → RawDocument
#   ٢. إيه اللي بيتخزن في الـ VectorDB? → EmbeddedChunk
#   ٣. إيه اللي بيتخزن في الـ SQL DB?   → DocumentRecord
#
# الـ metadata بنبنيها على:
#   - وقت الـ PDF (created_at / valid_until)
#   - محتوى الـ PDF (content_type)
# ============================================================

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ============================================================
# STEP 1: الـ Enums
# ليه؟ عشان نحدد القيم المسموح بيها بدل ما نسيب الـ field
# يقبل أي string — ده بيمنع الأخطاء من الأساس
# ============================================================

class ContentType(str, Enum):
    """
    أنواع المحتوى في الـ Marketing System.
    
    ليه str + Enum مع بعض؟
    - str    → عشان يتخزن كـ string في الـ DB والـ JSON
    - Enum   → عشان يقبل القيم دي بس ومحدش يكتب "Blog Post" بدل "blog_post"
    """
    CAMPAIGN_REPORT  = "campaign_report"   # تقارير أداء الحملات
    AD_COPY          = "ad_copy"           # نصوص الإعلانات
    MARKET_RESEARCH  = "market_research"   # أبحاث السوق
    BRAND_GUIDELINES = "brand_guidelines"  # إرشادات البراند
    EMAIL_TEMPLATE   = "email_template"    # قوالب الإيميلات
    SOCIAL_CONTENT   = "social_content"    # محتوى السوشيال ميديا


class DocumentStatus(str, Enum):
    """
    حالة الـ document في الـ pipeline.
    
    الـ PDF بيعدي على مراحل:
    PENDING → PROCESSING → INDEXED
                  ↓
               FAILED  (لو في error)
    """
    PENDING    = "pending"     # اتستلم بس لسه ما اتعملش حاجة
    PROCESSING = "processing"  # بيتعمل chunk + embedding دلوقتي
    INDEXED    = "indexed"     # اتخزن في VectorDB وجاهز للـ search
    FAILED     = "failed"      # في error حصل


# ============================================================
# STEP 2: الـ Metadata
#
# ده أهم model في الـ file.
# ليه؟ لأن الـ metadata هي اللي بتتفلتر بيها في الـ search.
#
# قررنا نحط فيه:
#   - content_type  ← لأن المستخدم قال "فلتر بالمحتوى"
#   - created_at    ← لأن المستخدم قال "فلتر بالوقت"
#   - valid_until   ← مهم في الـ marketing (الـ campaign بتنتهي)
#
# باقي الـ fields مساعدة للـ RAG نفسه (chunk_index, etc.)
# ============================================================

class DocumentMetadata(BaseModel):
    """
    المعلومات اللي بتوصف الـ document.
    
    بتتخزن جنب كل chunk في الـ VectorDB كـ "payload"
    عشان لما بنسرش نقدر نفلتر بيها.
    
    مثال على الفلترة:
        "جيبلي كل الـ campaign_reports من يناير لمارس"
        → filter: content_type = "campaign_report"
                  created_at between Jan and Mar
    """

    # ── فلاتر المحتوى ───────────────────────────────────────
    content_type: ContentType = ContentType.CAMPAIGN_REPORT
    # ده اللي قالنا عليه المستخدم — نوع المحتوى

    brand: Optional[str] = None
    # اسم البراند لو الشركة شغالة على أكتر من براند

    # ── فلاتر الوقت ─────────────────────────────────────────
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # امتى اتعمل الـ document ده — الفلتر الزمني الأساسي

    valid_until: Optional[datetime] = None
    # الـ campaign reports ليها تاريخ انتهاء
    # لو None يعني المحتوى مفيش له expiry

    # ── معلومات المصدر ──────────────────────────────────────
    source_filename: str = ""
    # اسم الـ PDF الأصلي — مهم للـ traceability

    source_url: Optional[str] = None
    # لو المحتوى جاي من URL

    author: Optional[str] = None
    # مين كتب الـ document

    # ── معلومات الـ Chunk نفسه ───────────────────────────────
    # دي مش للفلترة — دي للـ RAG pipeline نفسه
    chunk_index: int = 0
    # رقم الـ chunk ده في الـ document الأصلي
    # مثال: لو الـ PDF اتقسم لـ 10 chunks، ده رقم 3

    total_chunks: int = 1
    # إجمالي الـ chunks في الـ document
    # بنستخدمه عشان نعرف لو الـ document اتعمل index كامل

    page_number: Optional[int] = None
    # رقم الصفحة في الـ PDF الأصلي
    # مفيد لما المستخدم يسأل "من الصفحة التالتة"

    # ── Extra ────────────────────────────────────────────────
    extra: Dict[str, Any] = Field(default_factory=dict)
    # لو في معلومات تانية مش محددة دلوقتي
    # أحسن من إننا نكسر الـ schema بعدين


# ============================================================
# STEP 3: رحلة الـ Document (Pipeline Models)
#
# كل model بيمثل مرحلة في الـ pipeline.
# ليه منفصلين؟ عشان كل مرحلة بياخد input مختلف
# ويرجع output مختلف.
# ============================================================

class RawDocument(BaseModel):
    """
    STAGE 1: اللي بييجي من المستخدم لما يرفع الـ PDF.

    ده أبسط model — بياخد بس المعلومات الضرورية
    اللي المستخدم لازم يديها.
    """
    filename:    str
    # اسم الـ PDF: "q1_campaign_report.pdf"

    content:     str
    # النص المستخرج من الـ PDF بعد الـ parsing
    # (الـ extraction بتتعمل قبل ما نعمل الـ model ده)

    content_type: ContentType = ContentType.CAMPAIGN_REPORT
    # المستخدم بيختار نوع المحتوى لما يرفع

    source_url:  Optional[str] = None
    # لو جاي من رابط مش upload


class DocumentChunk(BaseModel):
    """
    STAGE 2: بعد ما بنقسم الـ document لـ chunks.

    ليه بنقسم؟ عشان الـ LLM مش بياخد الـ PDF كله —
    بياخد جزء صغير ذي صلة بالسؤال بس.

    الـ chunk_id بنعمله هنا عشان نتتبعه في الـ VectorDB.
    """
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    # ID فريد لكل chunk — بنستخدمه عشان نرجعله لو محتاج

    text: str
    # نص الـ chunk — جزء من الـ PDF الأصلي

    metadata: DocumentMetadata
    # المعلومات اللي بتوصف الـ chunk ده
    # (content_type, created_at, page_number, etc.)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        """
        الـ chunk لازم يكون فيه محتوى فعلي.
        بنعمل strip عشان نشيل المسافات الفاضية.
        """
        v = v.strip()
        if len(v) < 20:
            # 20 حرف minimum — أقل من كده مش مفيد للـ RAG
            raise ValueError(f"Chunk too short ({len(v)} chars). Min is 20.")
        return v

    @model_validator(mode="after")
    def chunk_index_valid(self) -> "DocumentChunk":
        """
        chunk_index لازم يكون أقل من total_chunks.
        مثال: لو عندك 5 chunks، الـ index بيكون 0,1,2,3,4
        """
        if self.metadata.chunk_index >= self.metadata.total_chunks:
            raise ValueError(
                f"chunk_index ({self.metadata.chunk_index}) must be "
                f"less than total_chunks ({self.metadata.total_chunks})"
            )
        return self


class EmbeddedChunk(BaseModel):
    """
    STAGE 3: بعد ما بنعمل embedding للـ chunk.

    ده اللي بيتخزن في الـ VectorDB (Qdrant/Pinecone).
    الـ embedding هو الـ vector (قائمة أرقام) اللي بيمثل
    معنى النص — بيه بنعمل الـ semantic search.
    """
    chunk_id:  str
    # نفس الـ ID اللي اتعمل في DocumentChunk

    text:      str
    # النص الأصلي — بنخزنه جنب الـ vector عشان نرجعه للمستخدم

    embedding: List[float]
    # الـ vector نفسه — قائمة أرقام
    # text-embedding-3-small → 1536 رقم
    # text-embedding-3-large → 3072 رقم

    metadata:  DocumentMetadata
    # بتتخزن كـ "payload" في Qdrant
    # عشان نفلتر بـ content_type و created_at

    @field_validator("embedding")
    @classmethod
    def valid_dimension(cls, v: List[float]) -> List[float]:
        """
        بنتأكد إن الـ embedding dimension صح.
        لو غلط يبقى في مشكلة في الـ embedding model.
        """
        allowed = (1536, 3072)
        if len(v) not in allowed:
            raise ValueError(
                f"Invalid embedding dimension: {len(v)}. "
                f"Expected one of {allowed}"
            )
        return v


# ============================================================
# STEP 4: الـ Database Record (Relational DB)
#
# ده بيتخزن في PostgreSQL — مش الـ VectorDB.
# ليه محتاجينه؟ عشان نتتبع الـ documents على مستوى الـ file
# مش على مستوى الـ chunk.
#
# مثال: الـ PDF "q1_report.pdf" اتقسم لـ 47 chunk —
# في الـ VectorDB عندنا 47 record.
# في PostgreSQL عندنا record واحد للـ PDF كله.
# ============================================================

class DocumentRecord(BaseModel):
    """
    الـ document كـ وحدة كاملة في الـ SQL Database.

    model_config = ConfigDict(from_attributes=True)
    ده بيخلي Pydantic يقرأ من SQLAlchemy ORM object مباشرة
    بدون ما نعمل .dict() يدوي.
    """
    model_config = ConfigDict(from_attributes=True)

    id:            UUID           = Field(default_factory=uuid4)
    filename:      str
    content_type:  ContentType
    status:        DocumentStatus = DocumentStatus.PENDING

    # الوقت
    created_at:    datetime       = Field(default_factory=datetime.utcnow)
    updated_at:    datetime       = Field(default_factory=datetime.utcnow)
    valid_until:   Optional[datetime] = None

    # إحصائيات
    total_chunks:  int            = 0
    # بيتحدث بعد ما الـ indexing يخلص

    # Traceability
    uploaded_by:   Optional[UUID] = None
    # مين رفع الـ document — مفيد للـ audit

    error_message: Optional[str]  = None
    # لو الـ status = FAILED — إيه المشكلة


# ============================================================
# STEP 5: الـ API Schemas
#
# دول الـ models اللي بتتعامل مع الـ HTTP requests/responses.
# منفصلين عن الباقي عشان الـ API schema ممكن يتغير
# من غير ما نأثر على الـ internal models.
# ============================================================

class DocumentUploadRequest(BaseModel):
    """
    اللي المستخدم بيبعته لما يرفع PDF.
    الـ content نفسه بييجي كـ file upload (multipart/form-data)
    مش في الـ JSON body.
    """
    content_type: ContentType        = ContentType.CAMPAIGN_REPORT
    source_url:   Optional[str]      = None
    author:       Optional[str]      = None
    valid_until:  Optional[datetime] = None
    extra:        Dict[str, Any]     = Field(default_factory=dict)


class DocumentUploadResponse(BaseModel):
    """
    اللي بيرجع للمستخدم بعد الـ upload.
    بيديه الـ document_id عشان يتابع الـ status.
    """
    document_id: UUID
    status:      DocumentStatus
    message:     str
    # مثال: "Document received. Processing started."


class DocumentStatusResponse(BaseModel):
    """
    لما المستخدم يسأل: "الـ PDF بتاعي اتعمل index؟"
    """
    document_id:   UUID
    status:        DocumentStatus
    total_chunks:  int
    created_at:    datetime
    error_message: Optional[str] = None