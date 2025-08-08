import uuid
from typing import Iterable, override, Any

from qdrant_client import models
from qdrant_client.models import PointStruct
from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding

from src.embedding.base import BaseEmbedding
from src.schema import Chunk, QueryRoute
from src.vec_database.base import BaseDatabase


class HybridDatabase(BaseDatabase):
    def __init__(self, conn_string: str, dense: BaseEmbedding, sparse: SparseTextEmbedding, late: LateInteractionTextEmbedding):
        super().__init__(conn_string)
        self.dense = dense
        self.sparse = sparse
        self.late = late


    @override
    async def create_collection(self, collection_name: str):
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

        await self.client.create_collection(
            collection_name,
            vectors_config={
                self.dense.model_name: dense_vec_params,
                self.late.model_name: late_vec_params
            },
            sparse_vectors_config={
                self.sparse.model_name: sparse_vec_params
            }
        )


    async def text_to_embeddings(self, text: str) -> Iterable[Any]:
        dense_vec = await self.dense.embed(text)
        sparse_vec = models.SparseVector(
            **next(iter(self.sparse.embed(text))).as_object()
        )
        late_vec = next(iter(self.late.embed(text)))

        return dense_vec, sparse_vec, late_vec


    @override
    async def chunk_to_point(self, chunk: Chunk) -> PointStruct:
        dense_vec, sparse_vec, late_vec = await self.text_to_embeddings(chunk.content)

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


    @override
    async def query(self, collection_name: str, query: str, limit: int, route: QueryRoute) -> Iterable[Chunk]:
        dense_vec, sparse_vec, late_vec = await self.text_to_embeddings(query)
        query_filter = await self.get_query_filter(route)

        dense_prefetch = models.Prefetch(
            query=dense_vec,
            using=self.dense.model_name,
            filter=query_filter,
            limit=limit
        )

        sparse_prefetch = models.Prefetch(
            query=sparse_vec,
            using=self.sparse.model_name,
            filter=query_filter,
            limit=limit
        )

        points = await self.client.query_points(
            collection_name=collection_name,
            prefetch=[dense_prefetch, sparse_prefetch],
            query=late_vec,
            using=self.late.model_name,
            limit=limit
        )

        return (Chunk(**point.payload) for point in points.points)
