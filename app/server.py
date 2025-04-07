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
    load_index("data/art.index")
    print("Index loaded at startup")
    yield

app = FastAPI(lifespan=lifespan)
PORT = 8000

@app.post("/query")
async def query_image(file: UploadFile = File(...)):
    try:
        print("Received a query request with file:", file.filename)
        
        # Read the image from the file
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        print("Image opened and converted to RGB.")
        
        # Extract the embedding from the image
        emb = extract_embedding(img)
        print("Extracted embedding from image:", emb)
        
        # Search the index using the extracted embedding
        results = search_index(emb)
        print("Search results from index:", results)
        
        return {"results": results}
    except Exception as e:
        print(f"Error during query processing: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/index")
async def index_images(files: List[UploadFile] = File(...)):
    try:
        for file in files:
            print("Received an index request with file:", file.filename)

            img = Image.open(io.BytesIO(await file.read())).convert("RGB")
            print("Image opened and converted to RGB.")

            emb = extract_embedding(img)
            print("Extracted embedding from image:", emb)

            add_to_index(emb, file.filename)
            print(f"Added embedding for {file.filename} to index.")
            
        save_index("data/art.index")
        print("Index saved to file.")
        return {"status": "indexed", "ids": [file.filename for file in files]}
    except Exception as e:
        print(f"Error during indexing: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})