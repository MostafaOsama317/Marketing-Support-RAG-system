# core/db/relational.py
#
# ============================================================
# الـ file ده فيه ٣ أجزاء:
#
#   ١. ORM Model    → DocumentORM   (الـ table في PostgreSQL)
#   ٢. Database     → DatabaseManager (الـ connection)
#   ٣. Repository   → DocumentRepository (العمليات على الـ DB)
#
# بيتكامل مع:
#   - document.py → DocumentRecord, DocumentStatus, ContentType
# ============================================================

from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Column,
    DateTime,
    Enum as SAEnum,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ── بنستورد من document.py اللي عملناه ─────────────────────
from src.core.models.document import ContentType, DocumentRecord, DocumentStatus


# ============================================================
# STEP 1: الـ Base
#
# كل ORM model لازم يورث منه —
# بيقول لـ SQLAlchemy إن الـ class ده بيمثل table
# ============================================================

class Base(DeclarativeBase):
    pass


# ============================================================
# STEP 2: الـ ORM Model
#
# ده بيمثل الـ table في PostgreSQL.
#
# ليه عندنا ORM Model و Pydantic Model مع بعض؟
#
#   DocumentORM    → SQLAlchemy — بيتكلم مع الـ DB
#   DocumentRecord → Pydantic   — بيتكلم مع الـ API والـ code
#
# الـ flow:
#   DocumentRecord → نحوله لـ ORM → نخزنه في DB
#   ORM من DB     → نحوله لـ DocumentRecord → نبعته للـ API
# ============================================================

class DocumentORM(Base):
    """
    الـ table بتاع الـ documents في PostgreSQL.

    كل row = PDF كامل (مش chunk).
    الـ chunks بتاعته بتتخزن في Qdrant.
    """
    __tablename__ = "documents"

    # ── Primary Key ─────────────────────────────────────────
    id = Column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        # as_uuid=True → بيرجع UUID object مش string
    )

    # ── معلومات الـ PDF ──────────────────────────────────────
    filename = Column(String(255), nullable=False)
    # اسم الـ PDF: "q1_campaign_report.pdf"

    file_type = Column(String(50), nullable=False, default="pdf")
    # دلوقتي pdf بس — ممكن يتوسع بعدين

    content_type = Column(
        SAEnum(ContentType, name="content_type_enum"),
        nullable=False,
    )
    # نوع المحتوى — من الـ ContentType enum في document.py

    # ── الـ Status ───────────────────────────────────────────
    status = Column(
        SAEnum(DocumentStatus, name="document_status_enum"),
        nullable=False,
        default=DocumentStatus.PENDING,
    )
    # pending → processing → indexed
    #                ↓
    #             failed

    error_message = Column(Text, nullable=True)
    # لو status = failed — إيه المشكلة

    # ── إحصائيات ────────────────────────────────────────────
    total_chunks = Column(Integer, default=0)
    # بيتحدث بعد ما الـ chunking يخلص

    # ── التواريخ ─────────────────────────────────────────────
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    valid_until = Column(DateTime, nullable=True)
    # onupdate → بيتحدث تلقائي لما أي field يتغير

    # ── Traceability ─────────────────────────────────────────
    uploaded_by = Column(PGUUID(as_uuid=True), nullable=True)
    # مين رفع الـ PDF — مفيد للـ audit


# ============================================================
# STEP 3: الـ DatabaseManager
#
# بيعمل الـ connection بـ PostgreSQL ويديك sessions.
#
# الـ Session هي اللي بتستخدمها عشان تعمل
# أي عملية على الـ DB (insert, select, update, delete).
# ============================================================

class DatabaseManager:
    """
    بيتحكم في الاتصال بـ PostgreSQL.

    الاستخدام:
        db = DatabaseManager(url="postgresql://user:pass@localhost/marketing_rag")
        db.create_tables()

        with db.get_session() as session:
            # اعمل أي عملية هنا
            pass
    """

    def __init__(self, database_url: str):
        """
        database_url مثال:
            "postgresql://postgres:password@localhost:5432/marketing_rag"
        """
        self.engine = create_engine(
            database_url,
            echo=False,
            # echo=True → بيطبع كل SQL query في الـ console
            # مفيد للـ debugging، بس مش للـ production
        )
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_tables(self) -> None:
        """
        بيعمل الـ tables في الـ DB لو مش موجودة.
        بنستدعيها مرة واحدة لما السيستم يشتغل.
        """
        Base.metadata.create_all(bind=self.engine)
        print("✅ Tables created successfully.")

    def get_session(self) -> Session:
        """
        بيديك session تعمل بيها العمليات.

        بنستخدمه كـ context manager:
            with db.get_session() as session:
                session.add(...)
                session.commit()

        لو في error → بيعمل rollback تلقائي.
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# ============================================================
# STEP 4: الـ Repository
#
# ده اللي بيعمل العمليات الفعلية على الـ DB.
#
# ليه Repository pattern؟
# عشان الـ business logic ما تعرفش أي DB بنستخدم —
# بتكلم الـ Repository بس.
#
# مثال:
#   بدل ما تكتب SQL في كل حتة:
#       session.query(DocumentORM).filter(...)
#   بتكتب:
#       repo.get_by_id(document_id)
# ============================================================

class DocumentRepository:
    """
    كل العمليات على الـ documents table.

    الاستخدام:
        repo = DocumentRepository(session)
        doc  = repo.save(document_record)
        doc  = repo.get_by_id(doc_id)
        repo.update_status(doc_id, DocumentStatus.INDEXED)
    """

    def __init__(self, session: Session):
        self.session = session

    # ──────────────────────────────────────────────────────
    # SAVE
    # ──────────────────────────────────────────────────────

    def save(self, record: DocumentRecord) -> DocumentRecord:
        """
        بيخزن document جديد في الـ DB.

        بياخد DocumentRecord (Pydantic) →
        بيحوله لـ DocumentORM (SQLAlchemy) →
        بيخزنه في الـ DB →
        بيرجعه كـ DocumentRecord تاني

        مثال:
            record = DocumentRecord(
                filename="q1_report.pdf",
                content_type=ContentType.CAMPAIGN_REPORT,
                uploaded_by=user_id,
            )
            saved = repo.save(record)
            print(saved.id)  # UUID اتعمل تلقائي
        """
        # ── Pydantic → ORM ──────────────────────────────────
        orm_doc = DocumentORM(
            id=record.id,
            filename=record.filename,
            file_type="pdf",
            content_type=record.content_type,
            status=record.status,
            total_chunks=record.total_chunks,
            created_at=record.created_at,
            updated_at=record.updated_at,
            valid_until=record.valid_until,
            uploaded_by=record.uploaded_by,
            error_message=record.error_message,
        )

        self.session.add(orm_doc)
        self.session.flush()
        # flush → بيبعت الـ SQL للـ DB بس مش commit بعد
        # بنعمل commit في الـ DatabaseManager.get_session()

        # ── ORM → Pydantic ──────────────────────────────────
        return self._to_record(orm_doc)

    # ──────────────────────────────────────────────────────
    # GET BY ID
    # ──────────────────────────────────────────────────────

    def get_by_id(self, document_id: UUID) -> Optional[DocumentRecord]:
        """
        بيجيب document بالـ ID بتاعه.

        بيرجع None لو مش موجود — مش بيـraise exception.

        مثال:
            doc = repo.get_by_id(some_uuid)
            if doc is None:
                raise HTTPException(404, "Document not found")
        """
        orm_doc = (
            self.session.query(DocumentORM)
            .filter(DocumentORM.id == document_id)
            .first()
        )
        return self._to_record(orm_doc) if orm_doc else None

    # ──────────────────────────────────────────────────────
    # UPDATE STATUS
    # ──────────────────────────────────────────────────────

    def update_status(
        self,
        document_id: UUID,
        status: DocumentStatus,
        total_chunks: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> Optional[DocumentRecord]:
        """
        بيحدث حالة الـ document في الـ pipeline.

        بنستدعيه في ٣ حالات:
            - لما الـ processing يبدأ  → PROCESSING
            - لما الـ indexing يخلص   → INDEXED + total_chunks
            - لما في error يحصل       → FAILED + error_message

        مثال:
            repo.update_status(doc_id, DocumentStatus.INDEXED, total_chunks=47)
            repo.update_status(doc_id, DocumentStatus.FAILED, error_message="Parse error")
        """
        orm_doc = (
            self.session.query(DocumentORM)
            .filter(DocumentORM.id == document_id)
            .first()
        )

        if not orm_doc:
            return None

        # ── بنحدث الـ fields ─────────────────────────────────
        orm_doc.status     = status
        orm_doc.updated_at = datetime.utcnow()

        if total_chunks is not None:
            orm_doc.total_chunks = total_chunks

        if error_message is not None:
            orm_doc.error_message = error_message

        self.session.flush()
        return self._to_record(orm_doc)

    # ──────────────────────────────────────────────────────
    # LIST ALL
    # ──────────────────────────────────────────────────────

    def list_all(
        self,
        status: Optional[DocumentStatus] = None,
        content_type: Optional[ContentType] = None,
        limit: int = 50,
    ) -> List[DocumentRecord]:
        """
        بيجيب قائمة بالـ documents مع فلترة اختيارية.

        مثال:
            # كل الـ documents
            docs = repo.list_all()

            # الـ documents اللي لسه processing
            docs = repo.list_all(status=DocumentStatus.PROCESSING)

            # campaign reports بس
            docs = repo.list_all(content_type=ContentType.CAMPAIGN_REPORT)
        """
        query = self.session.query(DocumentORM)

        if status:
            query = query.filter(DocumentORM.status == status)

        if content_type:
            query = query.filter(DocumentORM.content_type == content_type)

        orm_docs = (
            query
            .order_by(DocumentORM.created_at.desc())
            # الأحدث الأول
            .limit(limit)
            .all()
        )

        return [self._to_record(doc) for doc in orm_docs]

    # ──────────────────────────────────────────────────────
    # DELETE
    # ──────────────────────────────────────────────────────

    def delete(self, document_id: UUID) -> bool:
        """
        بيمسح document من الـ DB.

        بيرجع True لو اتمسح، False لو مش موجود.

        ملاحظة: لازم تمسح الـ chunks من Qdrant كمان
        باستخدام vector.py → delete_by_source()
        """
        orm_doc = (
            self.session.query(DocumentORM)
            .filter(DocumentORM.id == document_id)
            .first()
        )

        if not orm_doc:
            return False

        self.session.delete(orm_doc)
        self.session.flush()
        return True

    # ──────────────────────────────────────────────────────
    # HELPER: ORM → Pydantic
    # ──────────────────────────────────────────────────────

    def _to_record(self, orm_doc: DocumentORM) -> DocumentRecord:
        """
        بيحول DocumentORM → DocumentRecord (Pydantic).

        from_attributes=True في DocumentRecord
        بيخلي الـ conversion ده يشتغل تلقائي.
        """
        return DocumentRecord.model_validate(orm_doc)
        # model_validate مع from_attributes=True →
        # بيقرأ الـ attributes من الـ ORM object مباشرة