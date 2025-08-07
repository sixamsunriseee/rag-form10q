from typing import override

import pdfplumber

from src.parser.base import BaseParser


class TextParser(BaseParser):
    @override
    def parse_to_string(self, filename: str) -> str:
        with pdfplumber.open(filename) as pdf:
            return '\n\n'.join(page.extract_text() for page in pdf.pages)
