from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedding(ABC):
    def __init__(self, model_name: str, embedding_size: int):
        self.model_name = model_name
        self.embedding_size = embedding_size

    @abstractmethod
    async def embed(self, content: str) -> list[float] | np.ndarray: ...
