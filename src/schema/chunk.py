from typing import override

from src.schema.base import BaseModel


class Chunk(BaseModel):
    filename: str
    text: str

    @property
    def content_to_embed(self) -> str: return self.text


class ChunkWithContext(Chunk):
    context: str

    @override
    @property
    def content_to_embed(self) -> str: return self.context
