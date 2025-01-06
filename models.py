from pydantic import BaseModel
from typing import List

class Query(BaseModel):
    text: str

class Document(BaseModel):
    id: int
    values: List[str]
    content: str

class IndexRequest(BaseModel):
    documents: List[Document]

class InferenceRequest(BaseModel):
    queries: List[Query]

class InferenceResponse(BaseModel):
    results: List[List[int]]