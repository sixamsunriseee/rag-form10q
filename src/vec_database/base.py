import asyncio
from abc import ABC, abstractmethod

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

from src.schema import Chunk, Route


class BaseDatabase(ABC):
    def __init__(self, conn_string: str):
        self.client = AsyncQdrantClient(path=conn_string)


    @abstractmethod
    async def create_collection(self, collection_name: str): ...

    @abstractmethod
    async def chunk_to_point(self, chunk: Chunk) -> PointStruct: ...


    async def upsert_chunks(self, collection_name: str, chunks: list[Chunk]):
        await self.client.upsert(
            collection_name=collection_name,
            points=await asyncio.gather(
                *[self.chunk_to_point(chunk) for chunk in chunks]
            )
        )


    @staticmethod
    def get_route_field_conditions(route: Route) -> list[FieldCondition]:
        return [
            FieldCondition(key='route.year', match=MatchValue(value=route.year)),
            FieldCondition(key='route.quarter', match=MatchValue(value=route.quarter)),
            FieldCondition(key='route.company', match=MatchValue(value=route.company))
        ]


    @staticmethod
    def get_query_filter(route: Route) -> Filter:
        return Filter(must=BaseDatabase.get_route_field_conditions(route))


    async def bundle_chunk(self, collection_name: str, chunk: Chunk, route: Route):
        conditions = BaseDatabase.get_route_field_conditions(route)

        before = await self.client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key='index',
                        match=MatchValue(value=chunk.index - 1)
                    ),
                    *conditions
                ]
            )
        )

        after = await self.client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key='index',
                        match=MatchValue(value=chunk.index + 1)
                    ),
                    *conditions
                ]
            )
        )

        if before[0]:
            chunk.content = before[0][0].payload['content'] + chunk.content

        if after[0]:
            chunk.content += after[0][0].payload['content']


    @abstractmethod
    async def get_ordered_chunks(self, collection_name: str, query: str, route: Route, limit: int) -> list[Chunk]: ...
