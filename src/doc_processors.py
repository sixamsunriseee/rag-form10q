import os
from abc import ABC, abstractmethod
from typing import Iterable, override

import pdfplumber
from chonkie import BaseChunker
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

from src.schema.chunk import Chunk



class BaseProcessor(ABC):
    @abstractmethod
    def parse_to_string(self, filename: str | os.PathLike) -> str: ...


    @abstractmethod
    def split_to_chunks(self, filename: str | os.PathLike, prefetch: str = None) -> Iterable[Chunk]: ...



class PlumberProcessor(BaseProcessor):
    def __init__(self, page_sep: str, chunker: BaseChunker):
        self.page_sep = page_sep
        self.chunker = chunker


    @override
    def parse_to_string(self, filename: str | os.PathLike) -> str:
        with pdfplumber.open(filename) as pdf:
            pages_as_string = (page.extract_text() for page in pdf.pages)

            return self.page_sep.join(pages_as_string)


    @override
    def split_to_chunks(self, filename: str | os.PathLike, prefetch: str = None) -> Iterable[Chunk]:
        text = prefetch if prefetch else self.parse_to_string(filename)

        for chunk in self.chunker(text):
            yield Chunk(filename=os.path.basename(filename), text=chunk.text)



class MarkerProcessorMarkdown(BaseProcessor):
    def __init__(self, chunker: BaseChunker):
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
    def parse_to_string(self, filename: str | os.PathLike) -> str:
        rendered = self.converter(filename)
        text, output_type, images = text_from_rendered(rendered)

        return text


    @override
    def split_to_chunks(self, filename: str | os.PathLike, prefetch: str = None) -> Iterable[Chunk]:
        text = prefetch if prefetch else self.parse_to_string(filename)

        for chunk in self.chunker(text):
            yield Chunk(filename=os.path.basename(filename), text=chunk.text)
