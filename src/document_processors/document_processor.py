import os
from abc import ABC, abstractmethod


class DocumentProcessor(ABC):
    @abstractmethod
    def parse(self, filename: str | os.PathLike) -> str: ...

    @abstractmethod
    def to_chunks(self, filename: str | os.PathLike) -> list[dict]: ...

    @staticmethod
    def extract_doc_keys(filename: str | os.PathLike) -> dict:
        filename = os.path.basename(filename)
        year, quarter, company = filename.removesuffix('.pdf').split()
        return {
            'year': year,
            'quarter': quarter,
            'company': company
        }
