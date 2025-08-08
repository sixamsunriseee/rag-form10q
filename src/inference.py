from src.llm.base import BaseLLM
from src.schema import Chunk
from src.vec_database.base import BaseDatabase


async def run_inference(
    llm: BaseLLM,
    instructions: str,
    database: BaseDatabase,
    collection_name: str,
    query: str,
    retrieve_limit: int
) -> tuple[str, list[Chunk]]:
    route = await llm.get_query_route(query)
    chunks = list(await database.query(collection_name, query, limit=retrieve_limit, route=route))

    response = await llm.query(
        instructions=instructions,
        chunks=chunks,
        query=query,
    )

    return response, chunks
