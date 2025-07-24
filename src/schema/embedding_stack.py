from src.schema.base import BaseModel
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding


class EmbeddingStack(BaseModel):
    dense: TextEmbedding
    sparse: SparseTextEmbedding
    late_interaction: LateInteractionTextEmbedding
