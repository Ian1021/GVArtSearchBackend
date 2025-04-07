## Getting Started
Follow these steps to run the server locally:

### 1. Clone the Repository

### 2. Navigate into the Project Directory

```bash
cd GVARTSEARCHBACKEND
```

### 3. Install Dependencies

Recommened to be in a virtual environment (but not necessary), then install all required packages with the following command:

```bash
pip install -r requirements.txt
```

### 4. Start the Server

Run the FastAPI server using Uvicorn:

```bash
uvicorn app.server:app
```

You can add the `--reload` flag to automatically restart the server when you make changes in your code

### 5. Verify It's Running

You should see a line like:

```
Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### 6. To test the API is setup

Open your browser and go to:

```
http://127.0.0.1:8000/docs
```
(Note: If you see a different URL in step 5, navigate to that instead and append with `/docs`)

This will bring up the interactive UI where you can test endpoints directly.

## Avaialble Endpoints

Both endpoints use the `POST` method:

`/index` - Index Images
- Upload one or more image files using the key `files`.
- These images will be processed then added to the vector database.

 `/query` - Search Similar Images
 - Upload a single image file using the key `file`.
 - The API will return 6 similar images based on their embeddings.