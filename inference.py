import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.utils.colpali_processing_utils import process_images, process_queries
from PIL import Image
from config import MODEL_NAME, BASE_MODEL, DEVICE
from utils import initialize_pinecone, upsert_vectors, query_vectors

class ColPaliInference:
    def __init__(self):
        self.model = ColPali.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE).eval()
        self.model.load_adapter(MODEL_NAME)
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.pinecone_index = initialize_pinecone()

    def process_documents(self, documents):
        dummy_image = Image.new("RGB", (448, 448), (255, 255, 255))
        dataloader = DataLoader(
            documents,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda x: process_images(self.processor, [dummy_image] * len(x)),
        )
        ds = []
        for batch_doc in dataloader:
            with torch.no_grad():
                batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                embeddings_doc = self.model(**batch_doc)
            ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
        return ds

    def process_queries(self, queries):
        dummy_image = Image.new("RGB", (448, 448), (255, 255, 255))
        dataloader = DataLoader(
            queries,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda x: process_queries(self.processor, x, dummy_image),
        )
        qs = []
        for batch_query in dataloader:
            with torch.no_grad():
                batch_query = {k: v.to(self.model.device) for k, v in batch_query.items()}
                embeddings_query = self.model(**batch_query)
            qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))
        return qs

    def index_documents(self, documents):
        embeddings = self.process_documents(documents)
        upsert_vectors(self.pinecone_index, embeddings, [str(i) for i in range(len(documents))])

    def retrieve_documents(self, query, top_k=10):
        query_embedding = self.process_queries([query])[0]
        return query_vectors(self.pinecone_index, query_embedding.tolist(), top_k)