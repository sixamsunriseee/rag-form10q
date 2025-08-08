import uuid
from typing import override, Iterable

from qdrant_client import models
from qdrant_client.models import PointStruct

from src.schema import Chunk, QueryRoute
from src.embedding.base import BaseEmbedding
from src.vec_database.base import BaseDatabase


class DenseDatabase(BaseDatabase):
    def __init__(self, conn_string: str, dense: BaseEmbedding):
        super().__init__(conn_string)
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
            vector=await self.dense.embed(chunk.content),
            payload=chunk.model_dump()
        )

        return point


    @override
    async def query(self, collection_name: str, query: str, limit: int, route: QueryRoute) -> Iterable[Chunk]:
        points = await self.client.query_points(
            collection_name=collection_name,
            query=await self.dense.embed(query),
            query_filter=await self.get_query_filter(route),
            limit=limit,
        )

        return (Chunk(**point.payload) for point in points.points)
