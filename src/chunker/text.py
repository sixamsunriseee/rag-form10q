from typing import override, Iterable

from chonkie import SentenceChunker

from src.chunker.base import BaseChunker


class TextChunker(BaseChunker):
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunker = SentenceChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )


    @override
    def split_to_chunks(self, content: str) -> Iterable[str]:
        return (chunk.text for chunk in self.chunker(content))
