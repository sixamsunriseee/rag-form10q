from abc import ABC, abstractmethod
from typing import Iterable, override

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from src.schema import Chunk
from config import HYBRID_CONN_STRING


class BaseDatabase(ABC):
    def __init__(self, conn_string: str):
        self.client = AsyncQdrantClient(path=conn_string)

    @abstractmethod
    async def create_collection(self, collection_name: str): ...

    @abstractmethod
    async def chunk_to_point(self, chunk: Chunk) -> PointStruct: ...

    async def upsert_chunks(self, collection_name: str, chunks: Iterable[Chunk]):
        await self.client.upsert(
            collection_name=collection_name,
            points=[await self.chunk_to_point(chunk) for chunk in chunks]
        )

    @abstractmethod
    async def query(self, collection_name: str, query: str, limit: int): ...
