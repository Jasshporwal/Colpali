from pinecone import Pinecone, ServerlessSpec
import os
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME

def initialize_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("Pinecone API key not found in environment variables")

    pc = Pinecone(api_key=api_key)
    index_name = os.getenv("PINECONE_INDEX_NAME", "default_index")
    dimension=768

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists. Using the existing index.")

    return pc.Index(index_name)


def upsert_vectors(index, vectors, ids):
    index.upsert(vectors=list(zip(ids, vectors)))

def query_vectors(index, query_vector, top_k=10):
    results = index.query(vector=query_vector, top_k=top_k)
    return [match.id for match in results.matches]