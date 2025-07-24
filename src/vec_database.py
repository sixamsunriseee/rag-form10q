import uuid
from typing import Iterable

from qdrant_client import models
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from fastembed.rerank.cross_encoder import TextCrossEncoder

from schema.chunk import Chunk
from schema.embedding_stack import EmbeddingStack


class QdrantDatabase:
    def __init__(self, url: str, embeddings: EmbeddingStack):
        self.client = QdrantClient(path=url)
        self.embeddings = embeddings


    def create_collection(self, collection_name: str):
        dense_vec_params = models.VectorParams(
            size=self.embeddings.dense.embedding_size,
            distance=models.Distance.COSINE,
        )

        sparse_vec_params = models.SparseVectorParams(
            modifier=models.Modifier.IDF
        )

        late_vec_params = models.VectorParams(
            size=self.embeddings.late_interaction.embedding_size,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0)
        )

        self.client.create_collection(
            collection_name,
            vectors_config={
                self.embeddings.dense.model_name: dense_vec_params,
                self.embeddings.late_interaction.model_name: late_vec_params
            },
            sparse_vectors_config={
                self.embeddings.sparse.model_name: sparse_vec_params
            }
        )

    def text_to_embeddings(self, text: str) -> Iterable[models.Document]:
        dense_vec = models.Document(text=text, model=self.embeddings.dense.model_name)
        sparse_vec = models.Document(text=text, model=self.embeddings.sparse.model_name)
        late_vec = models.Document(text=text, model=self.embeddings.late_interaction.model_name)

        return dense_vec, sparse_vec, late_vec


    def chunk_to_point(self, chunk: Chunk):
        dense_vec, sparse_vec, late_vec = self.text_to_embeddings(chunk.content_to_embed)

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                self.embeddings.dense.model_name: dense_vec,
                self.embeddings.late_interaction.model_name: late_vec,
                self.embeddings.sparse.model_name: sparse_vec,
            },
            payload=chunk.model_dump()
        )

        return point

    def upsert_chunks(self, collection_name: str, chunks: Iterable[Chunk]):
        self.client.upsert(
            collection_name=collection_name,
            points=[self.chunk_to_point(chunk) for chunk in chunks]
        )


    def query(self, collection_name: str, query: str) -> Iterable[str]:
        dense_vec, sparse_vec, late_vec = self.text_to_embeddings(query)

        dense_prefetch = models.Prefetch(query=dense_vec, using=self.embeddings.dense.model_name, limit=40)
        sparse_prefetch = models.Prefetch(query=sparse_vec, using=self.embeddings.sparse.model_name, limit=40)

        points = self.client.query_points(
            collection_name=collection_name,
            prefetch=[dense_prefetch, sparse_prefetch],
            query=late_vec,
            using=self.embeddings.late_interaction.model_name,
            limit=20
        ).points

        reranker = TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')
        docs = [str(point.payload) for point in points]
        new_points = reranker.rerank(query, documents=docs)
        scores = [(score, doc) for score, doc in zip(new_points, docs)]
        scores.sort(key=lambda x: x[0], reverse=True)

        return map(lambda x: x[1], scores[:10])