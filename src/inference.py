from src.llm.base import BaseLLM
from src.schema import QueryChunks
from src.vec_database.base import BaseDatabase


async def run_inference(database: BaseDatabase, collection_name: str, retrieve_limit: int, llm: BaseLLM, query: str) -> str:
    decomposed_queries = await llm.get_subqueries(query)

    query_chunks = []

    for decomposed_query in decomposed_queries:
        route = await llm.get_route(decomposed_query)
        chunks = await database.get_ordered_chunks(collection_name, decomposed_query, route, retrieve_limit)
        query_chunks.append(QueryChunks(query=decomposed_query, chunks=chunks))

    answer = await llm.get_answer_multiple(initial_query=query, query_chunks=query_chunks)

    print(answer + '\n')

    return answer
