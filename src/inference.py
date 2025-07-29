from src.language_model import GptLanguageModel
from src.vec_database import QdrantDatabase
from config import OPENAI_KEY


instructions = '''
## Overview
You are given one or more context chunks, each in the format: {'filename': '...', 'text': '...'}, where text is in markdown format.

## Your task is to accurately and concisely answer the user’s question using only the information available in these chunks. Follow these guidelines:
    1) Ground All Claims in the Provided Texts:
        - Do not hallucinate. Only use details that are explicitly stated or logically inferred from the text.
        
    2) Use Professional and Analytical Language:
        - Write in a formal tone suitable for business or financial analysis.
        - Use objective language and avoid speculation unless it is clearly marked as such in the source.
        
    3) Reference Trends or Comparisons When Applicable:
        - If the question asks about changes over time, trends, or comparisons (e.g., "how has revenue changed"), summarize the relevant differences numerically or directionally.

    4) Aggregate Across Multiple Chunks If Needed:
        - Synthesize information across chunks where necessary but indicate the pattern or consistency across data points.
        
    5) Cite Source When Ambiguity Exists:
        - If multiple chunks present slightly different data or if the source matters, mention the file (e.g., "According to data in '10-Q_Q2_2023.pdf'...").
        
    6) Be Concise but Complete:
        - Keep responses clear and to the point. Prioritize relevance over verbosity.

    7) Data Interpretation for Tables:
        - When the chunk is a table, extract trends, fluctuations, or comparative insights instead of just restating rows.

## CONSTRAINTS
Generate your answer under 196 tokens.
'''

language_model = GptLanguageModel(OPENAI_KEY)


def run_inference(collection_name: str, query: str):
    database = QdrantDatabase()

    points = database.query(collection_name, query)

    response = language_model.answer(
        instructions='You are a chatbot in QA system. Answer from the chunks provided by the system',
        query='\n\n'.join([str({'system': chunk}) for chunk in points] + [str({'user': query})])
    )

    print(query)
    print(response)
    print()

    return response


if __name__ == '__main__':
    run_inference("How has Apple's total net sales changed over time?")