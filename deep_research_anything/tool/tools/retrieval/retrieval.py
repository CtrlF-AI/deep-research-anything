import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict

from deep_research_anything.tool.tool import ToolExecutionResult, Tool
from deep_research_anything.tool.tool_parameter import ToolParameter
from deep_research_stock.derived_tools.retrieval.models import RetrievalEvent
from deep_research_stock.rag.retrieve import query_data


# current_dir = os.path.dirname(os.path.abspath(__file__))
# json_file_path = os.path.join(current_dir, "../../resources/embeddings/akshare/stock_embeddings.json")
# COLLECTIONS = {
#     "akshare_stock": load_collection_from_json(json_file_path)
# }


@dataclass
class RetrievalResult:
    query: str
    result: Dict[str, Any]
    text_result: str


class RetrievalQueryParameter(ToolParameter):
    def __init__(self):
        name = "retrieval_query"
        code = "retrieval_query"
        description = "Vector retrieval query statement, use short keywords for queries, do not include stock names; try to use only"
        super().__init__(name=name, code=code, description=description,
                         optional=False, value=None, parameter_type=str)


# class CollectionParameter(EnumParameter):
#     def __init__(self):
#         name = "collection"
#         code = "collection"
#         enum_values = list(COLLECTIONS.keys())
#         description = "collection的名字"
#         super().__init__(name=name, code=code, description=description,
#                          optional=False, value=None, parameter_type=str, enum_values=enum_values)
        
        
class Retrieval(Tool):
    def __init__(self):
        super().__init__(
            name="retrieval",
            code="retrieval",
            description="akshare document retrieval tool, input collection and retrieval statement to query relevant akshare documents",
            documentation=""""""
        )
        
        # self.collections = COLLECTIONS
        self.parameters = {
            "retrieval_query": RetrievalQueryParameter(),
            # "collection": CollectionParameter()
        }
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_file_path = os.path.join(current_dir, "../../resources/embeddings/akshare/stock_embeddings.json")
        self.client, self.collection = load_collection_from_json(json_file_path)

    async def _execute(self, agent_args) -> ToolExecutionResult:
        results = await asyncio.to_thread(query_data, self.collection, query=self.parameters["retrieval_query"].value, n_results=3)
        event = RetrievalEvent(query=self.parameters["retrieval_query"].value, retrieval_result=results)
        await agent_args._notify_progress(event)
        return ToolExecutionResult.success(
            tool_code=self.code,
            result=RetrievalResult(query=self.parameters["retrieval_query"].value, result=results, text_result=self.get_results(results)),
            message="Retrieval completed"
        )

    def get_results(self, results):
        text = "\n\n".join([f"Result {i + 1}:\nDocument: {results['documents'][i]}\nID: {results['ids'][i]}\nDistance: {results['distances'][i]}" for i in range(len(results["documents"]))])
        return text
