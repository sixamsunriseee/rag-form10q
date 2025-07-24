from chonkie import TokenChunker
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

from src.doc_processors import PlumberProcessor
from src.schema.embedding_stack import EmbeddingStack
from src.schema.pipeline_config import PipelineConfig


pipeline_fast = PipelineConfig(
    processor=PlumberProcessor(
        page_sep='\n\n',
        chunker=TokenChunker(chunk_size=1024, chunk_overlap=128)
    ),
    embeddings=EmbeddingStack(
        dense=TextEmbedding('sentence-transformers/all-MiniLM-L6-v2'),
        sparse=SparseTextEmbedding("Qdrant/bm25"),
        late_interaction=LateInteractionTextEmbedding('colbert-ir/colbertv2.0'),
    )
)
