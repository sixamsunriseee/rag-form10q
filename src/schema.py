from typing import Literal

from pydantic import BaseModel


Year = Literal[2022, 2023]
Quarter = Literal['Q1', 'Q2', 'Q3']
Company = Literal['AAPL', 'AMZN', 'INTC', 'MSFT', 'NVDA']


class Route(BaseModel):
    year: Year
    quarter: Quarter
    company: Company


class Chunk(BaseModel):
    index: int
    filename: str
    content: str
    route: Route


class Subqueries(BaseModel):
    subqueries: list[str]


class QueryChunks(BaseModel):
    query: str
    chunks: list[Chunk]
