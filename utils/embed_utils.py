from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def embed_chunks(chunks):
    return model.encode(chunks)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_faiss(index, query_embedding, top_k=5):
    D, I = index.search(query_embedding, top_k)
    return I[0]
