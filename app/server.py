from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from PIL import Image
import io

from app.embedding import extract_embedding
from app.indexer import add_to_index, search_index, load_index, save_index

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_index()
    print("Index loaded at startup")
    yield

app = FastAPI(lifespan=lifespan)

# Endpoint to query the index with an image and return similar images
@app.post("/query")
async def query_image(file: UploadFile = File(...)):
    try:
        print(f"Received query for file: {file.filename}")
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        emb = extract_embedding(img)
        results = search_index(emb)
        print(f"Search results: {results}")
        return {"results": results}
    except Exception as e:
        print(f"Error during query: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# Endpoint to add images to the index
@app.post("/index")
async def index_images(files: List[UploadFile] = File(...)):
    try:
        for file in files:
            print(f"Indexing file: {file.filename}")
            img = Image.open(io.BytesIO(await file.read())).convert("RGB")
            emb = extract_embedding(img)
            add_to_index(emb, file.filename)
        save_index()
        print(f"Index saved. Now contains {len(search_index())} images.")
        return {"status": "indexed", "ids": [file.filename for file in files]}
    except Exception as e:
        print(f"Error during indexing: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})