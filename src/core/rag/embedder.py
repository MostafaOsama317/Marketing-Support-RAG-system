# core/rag/embedder.py
#
# ============================================================
# الـ file ده بيعمل حاجة واحدة بس:
#
#   بياخد List[DocumentChunk] (من chunker.py)
#   ويرجع List[EmbeddedChunk] (جاهزة للـ vector.py)
#
# بيتكامل مع:
#   - document.py  → DocumentChunk, EmbeddedChunk
#   - chunker.py   → بياخد الـ output بتاعه
#   - vector.py    → بيبعتله الـ output بتاعه
#
# ليه Qwen3-Embedding؟
#   - بيدعم العربي والإنجليزي بشكل ممتاز
#   - open source — مش محتاج API key
#   - بيشتغل local على الـ machine بتاعتك
# ============================================================

from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

from src.core.models.document import DocumentChunk, EmbeddedChunk
from src.config.Setting import DEFAULT_MODEL, BATCH_SIZE , EMBEDDING_DIM


# ============================================================
# STEP 2: الـ Embedder
# ============================================================

class DocumentEmbedder:
    """
    بيحول DocumentChunks لـ EmbeddedChunks.

    بيلود الـ model مرة واحدة في الـ __init__
    ويستخدمه لكل الـ chunks.

    الاستخدام:
        embedder = DocumentEmbedder()
        embedded = embedder.embed(chunks)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        بيلود الـ Qwen3 model من HuggingFace.

        أول مرة بيشتغل → بيحمل الـ model (دقايق)
        بعد كده → بيستخدم الـ cached model (ثواني)

        model_kwargs:
            device → "cpu" أو "cuda" لو عندك GPU
        encode_kwargs:
            normalize_embeddings → مهم للـ cosine similarity
            بيخلي كل vectors بنفس الـ scale
        """
        self.model_name = model_name

        print(f"⏳ Loading embedding model: {model_name}")

        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "device": "cuda",
                
            },
            encode_kwargs={
                "normalize_embeddings": True,
                # normalize → مهم عشان الـ cosine similarity يشتغل صح
                "batch_size": BATCH_SIZE,
            },
        )

        print(f"✅ Model loaded: {model_name}")

    # ──────────────────────────────────────────────────────
    # MAIN METHOD
    # ──────────────────────────────────────────────────────

    def embed(self, chunks: List[DocumentChunk]) -> List[EmbeddedChunk]:
        """
        بيحول قائمة chunks لـ EmbeddedChunks.

        STEP A: بيستخرج النصوص من الـ chunks
        STEP B: بيبعتهم للـ model في batches
        STEP C: بيركب الـ EmbeddedChunk من النص + الـ vector

        مثال:
            chunks   = chunker.chunk(raw_document)
            embedded = embedder.embed(chunks)
            # embedded جاهز يتبعت لـ vector.py
        """
        if not chunks:
            return []

        # ── STEP A: استخرج النصوص ───────────────────────────
        texts = [chunk.text for chunk in chunks]
        # بنبعت النصوص بس للـ model — مش الـ metadata

        # ── STEP B: عمل الـ embeddings ──────────────────────
        print(f"⏳ Embedding {len(texts)} chunks...")

        embeddings = self.model.embed_documents(texts)
        # embed_documents → بيتعامل مع الـ batching تلقائي
        # بيرجع List[List[float]]
        # كل List[float] هي الـ vector بتاع chunk واحد

        print(f"✅ Done. Embedding dim: {len(embeddings[0])}")

        # ── STEP C: ركّب الـ EmbeddedChunks ─────────────────
        embedded_chunks = []

        for chunk, embedding in zip(chunks, embeddings):
            # zip → بيربط كل chunk بالـ embedding بتاعه
            # chunk[0] ↔ embedding[0]
            # chunk[1] ↔ embedding[1]
            # ...

            embedded_chunk = EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                embedding=embedding,
                metadata=chunk.metadata,
                # نفس الـ metadata من الـ DocumentChunk
                # content_type, created_at, source_filename, etc.
            )
            embedded_chunks.append(embedded_chunk)

        return embedded_chunks

    # ──────────────────────────────────────────────────────
    # EMBED QUERY (للـ search)
    # ──────────────────────────────────────────────────────

    def embed_query(self, text: str) -> List[float]:
        """
        بيحول سؤال المستخدم لـ vector عشان نسرش بيه.

        الفرق بين embed_documents و embed_query:
            embed_documents → للـ chunks اللي بتتخزن
            embed_query     → للـ سؤال اللي بنسرش بيه

        في Qwen3 الفرق مهم —
        الـ query بياخد instruction مختلفة عن الـ document.

        مثال:
            query_vector = embedder.embed_query("ايه أفضل حملة في يناير؟")
            results      = qdrant.search(query_vector, ...)
        """
        return self.model.embed_query(text)