from abc import ABC, abstractmethod
from typing import override, Iterable

from chonkie import SentenceChunker


class BaseChunker(ABC):
    @abstractmethod
    def __call__(self, content: str) -> Iterable[str]: ...


class TextChunker(BaseChunker):
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunker = SentenceChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    @override
    def __call__(self, content: str) -> Iterable[str]:
        return (chunk.text for chunk in self.chunker(content))