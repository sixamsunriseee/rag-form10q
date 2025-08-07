from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    async def query(self, instructions: str, query: str) -> str: ...