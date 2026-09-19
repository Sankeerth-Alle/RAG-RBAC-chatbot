from typing import List
from typing_extensions import TypedDict, Annotated
from langchain_core.documents import Document


class AnswerWithSources(TypedDict):
    answer: str
    sources: Annotated[
        List[str],
        ...,
        "List of sources used to answer the question"
    ]


class State(TypedDict):
    question: str
    access_level: str
    context: List[Document]
    answer: AnswerWithSources