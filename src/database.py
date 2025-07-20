import os.path
from hashlib import sha1

from qdrant_client import models
from qdrant_client import QdrantClient
from qdrant_client.http.models import QueryResponse
from qdrant_client.models import VectorParams, Distance, PointStruct

from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding



class QdrantDatabase:
    def __init__(
        self,
        url: str,
        dense_model: TextEmbedding,
        sparse_model: SparseTextEmbedding,
        late_model: LateInteractionTextEmbedding
    ):
        self.client = QdrantClient(path=url)
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.late_model = late_model


    def create_collection(self, collection_name: str):
        dense_vector_params = models.VectorParams(
            size=self.dense_model.embedding_size,
            distance=models.Distance.COSINE,
        )

        sparse_vector_params = models.SparseVectorParams(
            modifier=models.Modifier.IDF
        )

        late_vector_params = models.VectorParams(
            size=self.late_model.embedding_size,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0)
        )

        self.client.create_collection(
            collection_name,
            vectors_config={
                self.dense_model.model_name: dense_vector_params,
                self.late_model.model_name: late_vector_params
            },
            sparse_vectors_config={
                self.sparse_model.model_name: sparse_vector_params
            }
        )


    def upsert_chunks(self, collection_name: str, chunks: list[dict]):
        points = []
        i = 0
        for chunk in chunks:
            i += 1
            dense_vector = next(self.dense_model.embed(chunk['text']))
            sparse_vector = next(self.sparse_model.embed(chunk['text'])).as_object()
            late_vector = next(self.late_model.embed(chunk['text']))

            point = PointStruct(
                id=i,
                vector={
                    self.dense_model.model_name: dense_vector,
                    self.late_model.model_name: late_vector,
                    self.sparse_model.model_name: sparse_vector,
                },
                payload=chunk
            )

            points.append(point)

        self.client.upsert(
            collection_name=collection_name,
            points=points
        )


    def query(self, collection_name: str, text: str) -> QueryResponse:
        dense_vector = next(self.dense_model.embed(text))
        sparse_vector = next(self.sparse_model.embed(text)).as_object()
        late_vector = next(self.late_model.embed(text))

        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using=self.dense_model.model_name,
                limit=40
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_vector),
                using=self.sparse_model.model_name,
                limit=20
            )
        ]

        points = self.client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=late_vector,
            using=self.late_model.model_name,
            limit=20
        )

        return points