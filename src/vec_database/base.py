import asyncio
from abc import ABC, abstractmethod
from typing import Iterable

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import PointStruct, Filter

from src.schema import Chunk, QueryRoute


class BaseDatabase(ABC):
    def __init__(self, conn_string: str):
        self.client = AsyncQdrantClient(path=conn_string)


    @staticmethod
    async def get_query_filter(route: QueryRoute) -> Filter:
        return models.Filter(
            must=[
                models.FieldCondition(key='year', match=models.MatchAny(any=route.years)),
                models.FieldCondition(key='quarter', match=models.MatchAny(any=route.quarters)),
                models.FieldCondition(key='company', match=models.MatchAny(any=route.companies))
            ]
        )


    @abstractmethod
    async def create_collection(self, collection_name: str): ...

    @abstractmethod
    async def chunk_to_point(self, chunk: Chunk) -> PointStruct: ...

    @abstractmethod
    async def query(self, collection_name: str, query: str, limit: int, route: QueryRoute) -> Iterable[Chunk]: ...


    async def upsert_chunks(self, collection_name: str, chunks: Iterable[Chunk]):
        await self.client.upsert(
            collection_name=collection_name,
            points=await asyncio.gather(*[self.chunk_to_point(chunk) for chunk in chunks])
        )
