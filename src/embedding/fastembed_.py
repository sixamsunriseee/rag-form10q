from typing import override

import numpy as np
from fastembed import TextEmbedding

from src.embedding.base import BaseEmbedding


class MiniLmEmbedding(BaseEmbedding):
    def __init__(self):
        super().__init__(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            embedding_size=384
        )

        self.model = TextEmbedding(self.model_name)


    @override
    async def embed(self, query: str) -> list[float] | np.ndarray:
        return next(iter(self.model.embed(query)))
