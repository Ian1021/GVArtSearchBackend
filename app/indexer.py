import faiss
import numpy as np
import os
import pickle

embedding_dim = 2560
index = faiss.IndexFlatL2(embedding_dim)
id_map = {}

INDEX_FILE = "data/art.index"
IDMAP_FILE = "data/id_map.pkl"

def add_to_index(embedding: np.ndarray, filename: str):
    index.add(np.expand_dims(embedding, axis=0))
    faiss_id = index.ntotal - 1
    id_map[faiss_id] = filename

def search_index(query_embedding: np.ndarray, top_k: int = 6):
    D, I = index.search(np.expand_dims(query_embedding, axis=0), top_k)
    print("Returned FAISS indices:", I[0])
    print("Current id_map keys:", list(id_map.keys()))
    return [id_map.get(i, f"unknown_id_{i}") for i in I[0]]

def save_index(index_path: str = INDEX_FILE, idmap_path: str = IDMAP_FILE):
    faiss.write_index(index, index_path)
    with open(idmap_path, "wb") as f:
        pickle.dump(id_map, f)
    print("Index and ID map saved.")

def load_index(index_path: str = INDEX_FILE, idmap_path: str = IDMAP_FILE):
    global index, id_map
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        print(f"Loaded FAISS index from {index_path}")
    else:
        index = faiss.IndexFlatL2(embedding_dim)
        print("Created new FAISS index")

    if os.path.exists(idmap_path):
        with open(idmap_path, "rb") as f:
            id_map = pickle.load(f)
        print(f"Loaded ID map from {idmap_path}")
    else:
        id_map = {}
        print("Initialized empty ID map")
