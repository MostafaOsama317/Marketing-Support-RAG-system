
# core/rag/chunker.py
#
# ============================================================
# الـ file ده بيعمل حاجة واحدة بس:
#
#   بياخد RawDocument (النص الكامل من الـ PDF)
#   ويرجع List[DocumentChunk] (أجزاء صغيرة جاهزة للـ embedding)
#
# بيتكامل مع:
#   - document.py → RawDocument, DocumentChunk, DocumentMetadata
#
# ليه langchain text splitter؟
#   عشان هو الأذكى في التقسيم — بيحاول يقسم على
#   الجمل والفقرات مش في النص الأوسط
# ============================================================

import re
from typing import List
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.models.document import DocumentChunk, DocumentMetadata, RawDocument
from src.config.Setting import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS_AR, SEPARATORS_EN



class DocumentChunker:
    """
    بيقسم نص الـ PDF لـ chunks مناسبة للـ RAG.

    بيتعامل مع العربي والإنجليزي تلقائياً.

    الاستخدام:
        chunker = DocumentChunker()
        chunks  = chunker.chunk(raw_document)
    """

    def __init__(
        self,
        chunk_size: int    = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    # ──────────────────────────────────────────────────────
    # MAIN METHOD
    # ──────────────────────────────────────────────────────

    def chunk(self, document: RawDocument) -> List[DocumentChunk]:
        """
        الميثود الرئيسية — بتاخد RawDocument وترجع chunks.

        STEP A: بتنظف النص
        STEP B: بتكتشف اللغة
        STEP C: بتقسم بناءً على اللغة
        STEP D: بتحول كل جزء لـ DocumentChunk

        مثال:
            raw = RawDocument(
                filename="q1_report.pdf",
                content="نص الـ PDF كامل هنا...",
                content_type=ContentType.CAMPAIGN_REPORT,
            )
            chunks = chunker.chunk(raw)
            print(len(chunks))  # عدد الـ chunks
        """

        # ── STEP A: تنظيف النص ──────────────────────────────
        cleaned_text = self._clean_text(document.content)

        if not cleaned_text:
            return []

        # ── STEP B: كشف اللغة ───────────────────────────────
        language = self._detect_language(cleaned_text)

        # ── STEP C: التقسيم ─────────────────────────────────
        text_chunks = self._split_text(cleaned_text, language)

        if not text_chunks:
            return []

        # ── STEP D: تحويل لـ DocumentChunk ──────────────────
        total = len(text_chunks)
        document_chunks = []

        for index, chunk_text in enumerate(text_chunks):
            metadata = DocumentMetadata(
                content_type=document.content_type,
                source_filename=document.filename,
                source_url=document.source_url,
                language=language,
                chunk_index=index,
                total_chunks=total,
            )

            chunk = DocumentChunk(
                chunk_id=str(uuid4()),
                text=chunk_text,
                metadata=metadata,
            )
            document_chunks.append(chunk)

        return document_chunks

    # ──────────────────────────────────────────────────────
    # STEP A: تنظيف النص
    # ──────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """
        بينظف النص قبل الـ chunking.

        بيشيل:
        - السطور الفاضية الزيادة
        - المسافات الزيادة
        - الـ special characters الغير مفيدة
        """
        if not text:
            return ""

        # بيشيل الـ special characters ما عدا علامات الترقيم المهمة
        text = re.sub(r'[^\w\s\n\.\!\?\،\؟\:\-\(\)]', ' ', text)

        # بيحول أكتر من سطر فاضي لسطرين بس
        text = re.sub(r'\n{3,}', '\n\n', text)

        # بيشيل المسافات الزيادة في السطر الواحد
        text = re.sub(r'[ \t]+', ' ', text)

        return text.strip()

    # ──────────────────────────────────────────────────────
    # STEP B: كشف اللغة
    # ──────────────────────────────────────────────────────

    def _detect_language(self, text: str) -> str:
        """
        بيكتشف لغة النص — عربي أو إنجليزي.

        الطريقة: بيحسب نسبة الحروف العربية في النص.
        لو أكتر من 30% عربي → يعتبره عربي.

        بيرجع: "ar" أو "en"
        """
        # الحروف العربية Unicode range
        arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
        total_chars  = len([c for c in text if c.isalpha()])

        if total_chars == 0:
            return "en"

        arabic_ratio = len(arabic_chars) / total_chars

        return "ar" if arabic_ratio > 0.3 else "en"
        # 0.3 = 30% — لو ربع النص عربي يبقى عربي

    # ──────────────────────────────────────────────────────
    # STEP C: التقسيم
    # ──────────────────────────────────────────────────────

    def _split_text(self, text: str, language: str) -> List[str]:
        """
        بيقسم النص لـ chunks باستخدام langchain splitter.

        بيختار الـ separators المناسبة للغة:
            عربي  → ؟ و ، وغيرهم
            إنجليزي → ? و . وغيرهم

        RecursiveCharacterTextSplitter بيشتغل إزاي:
            ١. بيجرب يقسم على "\n\n" (فقرات)
            ٢. لو الـ chunk لسه كبير، بيقسم على "\n" (أسطر)
            ٣. لو لسه كبير، بيقسم على "." (جمل)
            ٤. وهكذا...
        عشان كده "Recursive" — بيجرب الأكبر الأول
        """
        separators = SEPARATORS_AR if language == "ar" else SEPARATORS_EN

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
            length_function=len,
            # length_function=len → بيحسب بالـ characters
        )

        chunks = splitter.split_text(text)

        # بنشيل الـ chunks الفاضية أو القصيرة جداً
        chunks = [c.strip() for c in chunks if len(c.strip()) >= 20]

        return chunks