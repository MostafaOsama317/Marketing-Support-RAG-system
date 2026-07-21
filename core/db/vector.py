# core/db/vector.py
#
# ============================================================
# الـ file ده بيعمل ٣ حاجات بس:
#
#   ١. يتصل بـ Qdrant         → QdrantClientManager
#   ٢. يخزن الـ EmbeddedChunks → store()
#   ٣. يسرش عن أقرب chunks    → search()
#
# بيتكامل مع:
#   - document.py → EmbeddedChunk, DocumentMetadata, ContentType
#   - query.py    → ParsedIntent, RetrievedSource
# ============================================================

from config.Setting import VECTOR_SIZE

from typing import List, Optional
from uuid import UUID


from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from core.models.document import ContentType, EmbeddedChunk
from core.models.query import ParsedIntent, RetrievedSource


# ============================================================
# STEP 1: الـ Constants
#
# بنحطهم هنا فوق عشان لو حبينا نغير حاجة
# منغيرهاش في أكتر من مكان
# ============================================================

COLLECTION_NAME  = "marketing_docs"
VECTOR_SIZE      = VECTOR_SIZE   # text-embedding-3-small بتاع OpenAI
DISTANCE_METRIC  = Distance.COSINE
# COSINE = بيقيس التشابه في الاتجاه مش المسافة
# الأنسب للـ text embeddings


# ============================================================
# STEP 2: الـ QdrantClientManager
#
# ليه class مش functions؟
# عشان نعمل الـ connection مرة واحدة بس
# ونستخدمه في كل الـ operations
# ============================================================

class QdrantClientManager:
    """
    بيتحكم في الاتصال بـ Qdrant وكل العمليات عليه.

    الاستخدام:
        manager = QdrantClientManager(url="http://localhost:6333")
        manager.create_collection_if_not_exists()
        manager.store(chunks)
        results = manager.search(intent, top_k=5)
    """

    def __init__(self, url: str = "http://localhost:6333"):
        """
        url: عنوان الـ Qdrant server
             لو شغال local → "http://localhost:6333"
             لو cloud      → بتاخده من الـ Qdrant dashboard
        """
        # ── بنعمل الـ connection هنا مرة واحدة ────────────
        self.client = QdrantClient(url=url)
        self.collection_name = COLLECTION_NAME

    # ──────────────────────────────────────────────────────
    # CREATE COLLECTION
    # ──────────────────────────────────────────────────────

    def create_collection_if_not_exists(self) -> None:
        """
        بنعمل الـ collection لو مش موجودة.

        الـ collection زي الـ table في SQL —
        بس بدل rows فيها vectors.

        بنستدعي الميثود دي مرة واحدة بس
        لما السيستم يشتغل أول مرة.
        """
        # بنجيب الـ collections الموجودة
        existing = [
            col.name
            for col in self.client.get_collections().collections
        ]

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=DISTANCE_METRIC,
                ),
            )
            print(f"✅ Collection '{self.collection_name}' created.")
        else:
            print(f"ℹ️ Collection '{self.collection_name}' already exists.")

    # ──────────────────────────────────────────────────────
    # STORE
    # ──────────────────────────────────────────────────────

    def store(self, chunks: List[EmbeddedChunk]) -> None:
        """
        بيخزن قائمة من EmbeddedChunks في Qdrant.

        كل chunk بيتحول لـ PointStruct:
            - id      → chunk_id
            - vector  → embedding (الـ 1536 رقم)
            - payload → metadata (content_type, created_at, etc.)

        الـ payload هو اللي بنفلتر بيه وقت الـ search.

        مثال:
            chunks = embedder.embed(document_chunks)
            manager.store(chunks)
        """
        # ── بنحول كل EmbeddedChunk لـ PointStruct ─────────
        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=chunk.embedding,
                payload={
                    # ── النص الأصلي ───────────────────────
                    "text": chunk.text,

                    # ── فلاتر المحتوى ─────────────────────
                    "content_type":    chunk.metadata.content_type.value,
                    # .value عشان نخزن "campaign_report" مش الـ enum object

                    # ── فلاتر الوقت ───────────────────────
                    "created_at":      chunk.metadata.created_at.isoformat(),
                    # isoformat() → "2024-01-15T10:30:00"

                    "valid_until": (
                        chunk.metadata.valid_until.isoformat()
                        if chunk.metadata.valid_until
                        else None
                    ),

                    # ── معلومات المصدر ────────────────────
                    "source_filename": chunk.metadata.source_filename,
                    "page_number":     chunk.metadata.page_number,
                    "author":          chunk.metadata.author,

                    # ── معلومات الـ chunk ──────────────────
                    "chunk_index":     chunk.metadata.chunk_index,
                    "total_chunks":    chunk.metadata.total_chunks,
                }
            )
            for chunk in chunks
        ]

        # ── بنرفع كل الـ points دفعة واحدة ────────────────
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        # upsert = insert لو مش موجود، update لو موجود

        print(f"✅ Stored {len(points)} chunks in '{self.collection_name}'.")

    # ──────────────────────────────────────────────────────
    # SEARCH
    # ──────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: List[float],
        intent: ParsedIntent,
        top_k: int = 5,
    ) -> List[RetrievedSource]:
        """
        بيسرش عن أقرب chunks للسؤال في Qdrant.

        query_embedding: الـ vector بتاع سؤال المستخدم
        intent:          ParsedIntent — السيستم استنتجه من السؤال
                         (content_type, date_from, date_to)
        top_k:           كام نتيجة ترجع (default 5)

        بيرجع: List[RetrievedSource] — اللي عرفناه في query.py
        """

        # ── STEP A: بنبني الـ Filter ───────────────────────
        #
        # الفلتر بيتبني على أساس الـ ParsedIntent —
        # اللي السيستم استنتجه من سؤال المستخدم.
        # لو السيستم ملقاش intent معين، مبنفلترش بيه.
        #
        filter_conditions = []

        # فلتر المحتوى — لو السيستم عرف نوع المحتوى
        if intent.content_type:
            filter_conditions.append(
                FieldCondition(
                    key="content_type",
                    match=MatchValue(value=intent.content_type.value),
                )
            )

        # فلتر الوقت — لو السيستم عرف نطاق زمني
        if intent.date_from or intent.date_to:
            filter_conditions.append(
                FieldCondition(
                    key="created_at",
                    range=Range(
                        gte=intent.date_from.isoformat() if intent.date_from else None,
                        lte=intent.date_to.isoformat()   if intent.date_to   else None,
                    ),
                )
            )

        # ── بنحول الـ conditions لـ Filter object ──────────
        qdrant_filter = (
            Filter(must=filter_conditions)
            if filter_conditions
            else None
            # لو مفيش conditions → مفيش filter → بيسرش في كل حاجة
        )

        # ── STEP B: بنعمل الـ Search ───────────────────────
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k,
        ).points

        # ── STEP C: بنحول النتايج لـ RetrievedSource ───────
        #
        # RetrievedSource هو الـ model اللي عرفناه في query.py —
        # بيحتوي على النص والمصدر والـ score
        #
        retrieved = []
        for point in results:
            payload = point.payload or {}
            retrieved.append(
                RetrievedSource(
                    chunk_id=str(point.id),
                    text=payload.get("text", ""),
                    source_filename=payload.get("source_filename", ""),
                    score=point.score,
                    page_number=payload.get("page_number"),
                    content_type=(
                        ContentType(payload["content_type"])
                        if payload.get("content_type")
                        else None
                    ),
                )
            )

        return retrieved

    # ──────────────────────────────────────────────────────
    # DELETE (مساعد)
    # ──────────────────────────────────────────────────────

    def delete_by_source(self, source_filename: str) -> None:
        """
        بيمسح كل الـ chunks اللي جاية من PDF معين.

        مفيد لو:
        - المستخدم رفع نسخة جديدة من نفس الـ PDF
        - محتاج يحدّث المحتوى

        مثال:
            manager.delete_by_source("q1_report.pdf")
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source_filename",
                        match=MatchValue(value=source_filename),
                    )
                ]
            ),
        )
        print(f"🗑️ Deleted all chunks from '{source_filename}'.")