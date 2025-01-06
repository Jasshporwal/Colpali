from fastapi import FastAPI, HTTPException
from models import InferenceRequest, InferenceResponse, IndexRequest
from inference import ColPaliInference

app = FastAPI()
colpali_inference = ColPaliInference()

@app.post("/index", response_model=dict)
async def index_documents(request: IndexRequest):
    try:
        documents = [doc.content for doc in request.documents]
        colpali_inference.index_documents(documents)
        return {"message": f"Successfully indexed {len(documents)} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    try:
        results = []
        for query in request.queries:
            retrieved_docs = colpali_inference.retrieve_documents(query.text, top_k=10)
            results.append(retrieved_docs)
        return InferenceResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))