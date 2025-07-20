from typing import override
import os

import pdfplumber

from chonkie import BaseChunker, SentenceChunker

from . import DocumentProcessor

class PlumberProcessor(DocumentProcessor):
    def __init__(self, page_sep = '\n\n', chunker: BaseChunker = None):
        self.page_sep = page_sep

        if chunker is None:
            chunk_size = 2048
            chunk_overlap = int(chunk_size * 0.2)
            chunker = SentenceChunker.from_recipe(
                'markdown',
                lang='en',
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        self.chunker = chunker


    @override
    def parse(self, filename: str | os.PathLike) -> str:
        with pdfplumber.open(filename) as pdf:
            texts = map(lambda page: page.extract_text(), pdf.pages)
            return self.page_sep.join(texts)


    @override
    def to_chunks(self, filename: str | os.PathLike) -> list[dict]:
        text = self.parse(filename)
        doc_keys = self.extract_doc_keys(filename)
        chunks = []

        for chunk in self.chunker(text):
            chunks.append(
                {
                    **doc_keys,
                    'text': chunk.text
                }
            )

        return chunks

