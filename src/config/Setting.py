VECTOR_SIZE=1536
DISTANCE_METRIC="Cosine"

CHUNK_SIZE    = 1000   
CHUNK_OVERLAP = 200 
SEPARATORS_EN = ["\n\n", "\n", ".", "!", "?", " "]
SEPARATORS_AR = ["\n\n", "\n", ".", "!", "؟", "،", " "]


DEFAULT_MODEL = "Qwen/Qwen3-Embedding-4B"
 
# الـ batch size — كام chunk بنبعت للـ model في المرة
# لو الـ RAM قليلة نقلله، لو كتيرة نكبره
BATCH_SIZE = 32
 
# الـ embedding dimension بتاع Qwen3
# 4B → 2560 dimension
# 8B → 4096 dimension
EMBEDDING_DIM = {
    "Qwen/Qwen3-Embedding-4B": 2560,
    "Qwen/Qwen3-Embedding-8B": 4096,
}