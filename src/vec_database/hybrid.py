import os
import uuid
from typing import override

import numpy as np
from qdrant_client import models
from qdrant_client.models import PointStruct
from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding

from src.embedding.base import BaseEmbedding
from src.schema import Chunk, Route
from src.vec_database.base import BaseDatabase


class HybridDatabase(BaseDatabase):
    def __init__(
        self,
        dense: BaseEmbedding,
        sparse: SparseTextEmbedding,
        late: LateInteractionTextEmbedding,
        prefetch_limit = 20
    ):
        super().__init__(os.getenv("HYBRID_CONN_STRING"))
        self.dense = dense
        self.sparse = sparse
        self.late = late
        self.prefetch_limit = prefetch_limit


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


    def _get_sparse_embeddings(self, content: str) -> models.SparseVector:
        return models.SparseVector(
            **next(iter(self.sparse.embed(content))).as_object()
        )


    def _get_late_embeddings(self, content: str) -> list[float] | np.ndarray:
        return next(iter(self.late.embed(content)))


    @override
    async def chunk_to_point(self, chunk: Chunk) -> PointStruct:
        dense_vec = await self.dense.embed(chunk.content)
        sparse_vec = self._get_sparse_embeddings(chunk.content)
        late_vec = self._get_late_embeddings(chunk.content)

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                self.dense.model_name: dense_vec,
                self.sparse.model_name: sparse_vec,
                self.late.model_name: late_vec,
            },
            payload=chunk.model_dump()
        )

        return point


    @override
    async def get_ordered_chunks(self, collection_name: str, query: str, route: Route, limit: int) -> list[Chunk]:
        query_filter = self.get_query_filter(route)

        dense_prefetch = models.Prefetch(
            query=await self.dense.embed(query),
            using=self.dense.model_name,
            filter=query_filter,
            limit=self.prefetch_limit
        )

        sparse_prefetch = models.Prefetch(
            query=self._get_sparse_embeddings(query),
            using=self.sparse.model_name,
            filter=query_filter,
            limit=self.prefetch_limit
        )

        points = await self.client.query_points(
            collection_name=collection_name,
            prefetch=[dense_prefetch, sparse_prefetch],
            query=self._get_late_embeddings(query),
            using=self.late.model_name,
            limit=limit
        )

        chunks = [Chunk(**point.payload) for point in points.points]
        chunks.sort(key=lambda chunk: chunk.index)

        for chunk in chunks:
            await self.bundle_chunk_inplace(collection_name, chunk, route)

        return chunks
