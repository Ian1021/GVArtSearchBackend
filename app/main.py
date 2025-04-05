from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import zipfile
import os

from embedding import extract_embedding
from indexer import add_to_index, search_index

app = FastAPI()
counter = 0  # To assign incremental index keys
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
    global counter
    try:
        for file in files:
            print("Received an index request with file:", file.filename)
            
            # Read the image from the file
            img = Image.open(io.BytesIO(await file.read())).convert("RGB")
            print("Image opened and converted to RGB.")
            
            # Extract the embedding from the image
            emb = extract_embedding(img)
            print("Extracted embedding from image:", emb)
            
            # Add the embedding to the index
            add_to_index(emb, file.filename, counter)
            print(f"Added embedding for {file.filename} to index with key {counter}.")
            
            # Increment counter for the next index
            counter += 1
            print(f"Counter incremented. New counter value: {counter}")
        
        return {"status": "indexed", "ids": [file.filename for file in files]}
    except Exception as e:
        print(f"Error during indexing: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/index_folder")
async def index_folder(file: UploadFile = File(...)):
    global counter
    try:
        # Save the uploaded zip file temporarily
        zip_path = f"temp_{file.filename}"
        with open(zip_path, "wb") as f:
            f.write(await file.read())

        # Extract the zip file
        extracted_folder = f"extracted_{file.filename.split('.')[0]}"
        os.makedirs(extracted_folder, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder)
        
        print(f"Extracted files to {extracted_folder}")

        # Process each image in the extracted folder
        for img_file in os.listdir(extracted_folder):
            img_path = os.path.join(extracted_folder, img_file)
            
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Open and process the image
                with open(img_path, 'rb') as img_file_obj:
                    img = Image.open(img_file_obj).convert("RGB")
                    print(f"Processing image: {img_file}")
                    
                    # Extract embedding
                    emb = extract_embedding(img)
                    print(f"Extracted embedding for {img_file}: {emb}")

                    # Add the image embedding to the index
                    add_to_index(emb, img_file, counter)
                    print(f"Indexed {img_file} with key {counter}")

                    # Increment counter for the next index
                    counter += 1

        # Clean up the zip file and extracted folder
        os.remove(zip_path)
        for img_file in os.listdir(extracted_folder):
            os.remove(os.path.join(extracted_folder, img_file))
        os.rmdir(extracted_folder)

        return {"status": "indexed all images", "folder": file.filename}
    
    except Exception as e:
        print(f"Error during folder indexing: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})