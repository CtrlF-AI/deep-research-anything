from dataclasses import dataclass
from typing import List


@dataclass
class RetrievalEvent:
    query: str
    retrieval_result: List
