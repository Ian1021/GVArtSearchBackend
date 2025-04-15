import os
import faiss
import pickle
import numpy as np

from app.gcs import download_blob, upload_blob

embedding_dim = 2560
index = faiss.IndexFlatL2(embedding_dim)
id_map = {}

# GCS paths
GCS_INDEX_PATH = "data/art.index"
GCS_IDMAP_PATH = "data/id_map.pkl"

# Local temporary paths (Cloud Run allows writing to /tmp)
INDEX_FILE = "/tmp/art.index"
IDMAP_FILE = "/tmp/id_map.pkl"


def add_to_index(embedding: np.ndarray, filename: str):
    image_id = os.path.splitext(filename)[0]  # Remove extension
    if image_id in id_map.values():
        print(f"Skipping already indexed image: {image_id}")
        return
    index.add(np.expand_dims(embedding, axis=0))
    faiss_id = index.ntotal - 1
    id_map[faiss_id] = image_id


def search_index(query_embedding: np.ndarray, top_k: int = 6):
    D, I = index.search(np.expand_dims(query_embedding, axis=0), top_k)
    return [id_map.get(i, f"unknown_id_{i}") for i in I[0]]


def save_index(index_path: str = INDEX_FILE, idmap_path: str = IDMAP_FILE):
    faiss.write_index(index, index_path)
    with open(idmap_path, "wb") as f:
        pickle.dump(id_map, f)
    print("Index and ID map saved locally.")

    # Upload to GCS
    upload_blob(index_path, GCS_INDEX_PATH)
    upload_blob(idmap_path, GCS_IDMAP_PATH)
    print("Index and ID map uploaded to GCS.")


def load_index(index_path: str = INDEX_FILE, idmap_path: str = IDMAP_FILE):
    global index, id_map

    # Attempt to download from GCS
    try:
        download_blob(GCS_INDEX_PATH, index_path)
        download_blob(GCS_IDMAP_PATH, idmap_path)
        print("Downloaded index and ID map from GCS.")
    except Exception as e:
        print(f"Could not download index or ID map from GCS: {e}")

    # Load FAISS index
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        print(f"Loaded index from '{index_path}' with {index.ntotal} vectors.")
    else:
        index = faiss.IndexFlatL2(embedding_dim)
        print("Initialized new FAISS index.")

    # Load ID map
    if os.path.exists(idmap_path):
        with open(idmap_path, "rb") as f:
            id_map = pickle.load(f)
        print(f"Loaded ID map from '{idmap_path}'.")
    else:
        id_map = {}
        print("Initialized empty ID map.")