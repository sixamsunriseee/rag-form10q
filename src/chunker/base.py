from abc import ABC, abstractmethod
from typing import Iterable


class BaseChunker(ABC):
    @abstractmethod
    def split_to_chunks(self, content: str) -> Iterable[str]: ...
