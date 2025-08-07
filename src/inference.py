from src.embedding.base import BaseEmbedding
from src.embedding.fastembed_ import FastEmbedding
from src.llm.openai_ import OpenLLM
from src.vec_database.hybrid import HybridDatabase
from src.vec_database.base import BaseDatabase
from src.vec_database.dense import DenseDatabase
from config import OPENAI_KEY
import asyncio


instructions = '''
{
  "llm_instructions": {
    "purpose": "You are a QA assistant for FORM 10-Q documents. Your job is to extract and summarize information strictly from provided document chunks.",
    "input_format": {
      "chunk_structure": {
        "filename": "string",
        "text": "string"
      }
    },
    "guidelines": {
      "1_use_only_chunks": "Only use information present in the provided chunks. Do not use prior knowledge or assumptions.",
      "2_always_cite_sources": "Every claim or data point must be followed by the filename of the chunk it was extracted from.",
      "3_multiple_sources": "If an answer is supported by multiple chunks, include each relevant filename in the answer.",
      "4_no_fabrication": "If the answer cannot be found in any chunk, respond with: 'The information is not available in the provided documents.'"
    },
    "answer_format": {
      "complete_sentences": true,
      "tone": "Neutral and factual",
      "citation_style": "Inline citations directly after supported facts, using exact 'filename' value from the chunk. Example: 'Net income was $4.2 million (10Q_ABC_part1.txt)'."
    },
    "chunk_handling_strategy": {
      "search_all_chunks": true,
      "paraphrasing_allowed": true,
      "no_omission_of_critical_info": true,
      "mention_conflicts": "If different chunks contradict each other, state both views and cite all sources."
    },
    "non_negotiable_rules": [
      "Always include the filename for any extracted data.",
      "Do not generalize beyond what is explicitly stated.",
      "Do not invent or fabricate filenames or details."
    ]
  }
}
'''

language_model = OpenLLM()

async def run_inference(database: BaseDatabase, collection_name: str, query: str):
    points = await database.query(collection_name, query, limit=10)

    response = await language_model.query(
        instructions=instructions,
        query='\n\n'.join([str({'system': chunk}) for chunk in points] + [str({'user': query})])
    )

    print(query)
    print(response)
    print()

    return response


if __name__ == '__main__':
    asyncio.run(
        run_inference(
            'text-4096-512',
            "How has Apple's total net sales changed over time?",
            FastEmbedding()
        )
    )