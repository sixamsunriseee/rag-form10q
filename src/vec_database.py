import uuid
from typing import Iterable

from qdrant_client import models
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

from src.schema.chunk import Chunk
from config import QDRANT_CONN_STRING, DENSE_MODEL, SPARSE_MODEL, LATE_MODEL


class QdrantDatabase:
    def __init__(self):
        self.client = QdrantClient(path=QDRANT_CONN_STRING)
        self.dense = TextEmbedding(DENSE_MODEL)
        self.sparse = SparseTextEmbedding(SPARSE_MODEL)
        self.late = LateInteractionTextEmbedding(LATE_MODEL)


    def create_collection(self, collection_name: str):
        dense_vec_params = models.VectorParams(
            size=self.dense.embedding_size,
            distance=models.Distance.COSINE,
        )

        sparse_vec_params = models.SparseVectorParams(
            modifier=models.Modifier.IDF
        )

        late_vec_params = models.VectorParams(
            size=self.late.embedding_size,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0)
        )

        self.client.create_collection(
            collection_name,
            vectors_config={
                self.dense.model_name: dense_vec_params,
                self.late.model_name: late_vec_params
            },
            sparse_vectors_config={
                self.sparse.model_name: sparse_vec_params
            }
        )

    def text_to_embeddings(self, text: str) -> Iterable[models.Document]:
        dense_vec = models.Document(text=text, model=self.dense.model_name)
        sparse_vec = models.Document(text=text, model=self.sparse.model_name)
        late_vec = models.Document(text=text, model=self.late.model_name)

        return dense_vec, sparse_vec, late_vec


    def chunk_to_point(self, chunk: Chunk):
        dense_vec, sparse_vec, late_vec = self.text_to_embeddings(chunk.content_to_embed)

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                self.dense.model_name: dense_vec,
                self.late.model_name: late_vec,
                self.sparse.model_name: sparse_vec,
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

        # dense_prefetch = models.Prefetch(query=dense_vec, using=self.dense.model_name, limit=20)
        # sparse_prefetch = models.Prefetch(query=sparse_vec, using=self.sparse.model_name, limit=10)
        #
        # points = self.client.query_points(
        #     collection_name=collection_name,
        #     prefetch=[dense_prefetch, sparse_prefetch],
        #     query=late_vec,
        #     using=self.late.model_name,
        #     limit=10
        # ).points

        points = self.client.query_points(
            collection_name=collection_name,
            query=dense_vec,
            using=self.dense.model_name,
            limit=10
        ).points

        return (str(point.payload) for point in points)