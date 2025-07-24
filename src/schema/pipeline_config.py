from src.schema.base import BaseModel

from src.doc_processors import BaseProcessor
from src.schema.embedding_stack import EmbeddingStack


class PipelineConfig(BaseModel):
    processor: BaseProcessor
    embeddings: EmbeddingStack
