import os

from src.parser.base import BaseParser
from src.parser.markdown import MarkdownParser
from src.parser.text import TextParser
from config import DOCS_PATH, MARKDOWNS_PATH, TEXTS_PATH


def main(parser: BaseParser, output_directory: str, suffix: str):
    for filename in os.listdir(DOCS_PATH):
        if not filename.endswith('.pdf'):
            continue

        contents = parser.parse_to_string(os.path.join(DOCS_PATH, filename))
        filename = filename.removesuffix('.pdf') + suffix

        with open(os.path.join(output_directory, filename), 'w') as parsed_file:
            parsed_file.write(contents)


if __name__ == '__main__':
    main(MarkdownParser(), MARKDOWNS_PATH, '.md')
    main(TextParser(), TEXTS_PATH, '.txt')
