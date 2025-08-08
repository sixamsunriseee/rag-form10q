from typing import Literal

from pydantic import BaseModel


Year = Literal[2022, 2023]
Quarter = Literal['Q1', 'Q2', 'Q3']
Company = Literal['AAPL', 'AMZN', 'INTC', 'MSFT', 'NVDA']


class Chunk(BaseModel):
    filename: str
    year: Year
    quarter: Quarter
    company: Company
    content: str


class QueryRoute(BaseModel):
    companies: list[Company]
    quarters: list[Quarter]
    years: list[Year]
