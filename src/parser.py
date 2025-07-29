# std
from abc import ABC, abstractmethod
from typing import override

# 3rd party
import pdfplumber

from marker.models import create_model_dict
from marker.converters.pdf import PdfConverter
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered


class BaseParser(ABC):
    @abstractmethod
    def parse_to_string(self, filename: str) -> str: ...


class TextParser(BaseParser):
    @override
    def parse_to_string(self, filename: str) -> str:
        with pdfplumber.open(filename) as pdf:
            pages_as_string = (page.extract_text() for page in pdf.pages)
            return '\n\n'.join(pages_as_string)


class MarkdownParser(BaseParser):
    def __init__(self):
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
    def parse_to_string(self, filename: str) -> str:
        rendered = self.converter(filename)
        text, output_type, images = text_from_rendered(rendered)
        return text
