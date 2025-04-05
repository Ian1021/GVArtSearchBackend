import faiss
import numpy as np

embedding_dim = 512 + 2048
index = faiss.IndexFlatL2(embedding_dim)
id_map = {}
file_path = "art.index"

def add_to_index(embedding: np.ndarray, image_id: str, idx: int):
    index.add(np.expand_dims(embedding, axis=0))
    id_map[idx] = image_id

def search_index(query_embedding: np.ndarray, top_k: int = 5):
    D, I = index.search(np.expand_dims(query_embedding, axis=0), top_k)
    return [id_map[i] for i in I[0]]

def store_index(file_path: str):
    faiss.write_index(index, file_path)