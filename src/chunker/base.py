from abc import ABC, abstractmethod


class BaseChunker(ABC):
    @abstractmethod
    def split_to_chunks(self, content: str) -> list[str]: ...
