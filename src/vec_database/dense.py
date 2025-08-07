import uuid
from typing import override

from qdrant_client import models
from qdrant_client.models import PointStruct

from src.schema import Chunk
from src.embedding.base import BaseEmbedding
from src.vec_database.base import BaseDatabase
from config import DENSE_CONN_STRING


class DenseDatabase(BaseDatabase):
    def __init__(self, dense: BaseEmbedding):
        super().__init__(DENSE_CONN_STRING)
        self.dense = dense

    @override
    async def create_collection(self, collection_name: str):
        await self.client.create_collection(
            collection_name,
            vectors_config=models.VectorParams(
                size=self.dense.embedding_size,
                distance=models.Distance.COSINE,
            )
        )

    @override
    async def chunk_to_point(self, chunk: Chunk) -> PointStruct:
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=self.dense.embed(chunk.content_to_embed),
            payload=chunk.model_dump()
        )

        return point

    @override
    async def query(self, collection_name: str, query: str, limit: int):
        points = await self.client.query_points(
            collection_name=collection_name,
            query=self.dense.embed(query),
            limit=limit
        )

        return (str(point.payload) for point in points.points)
