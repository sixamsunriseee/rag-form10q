from typing import override

import numpy as np
from fastembed import TextEmbedding
from src.embedding.base import BaseEmbedding
from config import DENSE_MODEL


class FastEmbedding(BaseEmbedding):
    def __init__(self):
        self.model = TextEmbedding(DENSE_MODEL)
        super().__init__(DENSE_MODEL, self.model.embedding_size)


    @override
    def embed(self, content: str) -> list[float] | np.ndarray:
        return next(self.model.embed(content))
