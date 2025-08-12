from abc import ABC, abstractmethod

from src.schema import Route, QueryChunks


class BaseLLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    async def get_subqueries(self, query: str) -> list[str]: ...

    @abstractmethod
    async def get_route(self, query: str) -> Route: ...

    @abstractmethod
    async def get_answer_single(self, query_chunks: QueryChunks) -> str: ...

    @abstractmethod
    async def get_answer_multiple(self, initial_query: str, query_chunks: list[QueryChunks]) -> str: ...
