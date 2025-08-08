from abc import ABC, abstractmethod
from typing import Iterable

from src.schema import QueryRoute, Chunk


class BaseLLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    async def get_query_route(self, query: str) -> QueryRoute: ...

    @abstractmethod
    async def query(self, instructions: str, chunks: Iterable[Chunk], query: str) -> str: ...
