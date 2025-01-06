import os
import torch
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "vidore/colpali"
BASE_MODEL = "vidore/colpaligemma-3b-mix-448-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")