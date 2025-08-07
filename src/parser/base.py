from abc import ABC, abstractmethod


class BaseParser(ABC):
    @abstractmethod
    def parse_to_string(self, filename: str) -> str: ...
