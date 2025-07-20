# std
import os
from typing import override

from chonkie import BaseChunker, SentenceChunker
# 3-rd party

from marker.models import create_model_dict
from marker.converters.pdf import PdfConverter
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered

from . import DocumentProcessor


class MarkerProcessor(DocumentProcessor):
    def __init__(self, page_sep: str = '\n\n', chunker: BaseChunker = None):
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

        config_parser = ConfigParser(
            # https://github.com/datalab-to/marker?tab=readme-ov-file#convert-a-single-file
            cli_options={
                'output_format': 'markdown',
                'disable_image_extraction': True
            }
        )

        self.converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service()
        )

    @override
    def parse(self, filename: str | os.PathLike) -> str:
        rendered = self.converter(filename)
        text, output_type, images = text_from_rendered(rendered)

        return text


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
