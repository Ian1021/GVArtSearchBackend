from google.cloud import storage

BUCKET_NAME = "gvart-image-embeddings"

def get_storage_client():
    return storage.Client()

def download_blob(blob_name: str, destination_path: str):
    client = get_storage_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(destination_path)
    print(f"Downloaded {blob_name} to {destination_path}")

def upload_blob(source_path: str, blob_name: str):
    client = get_storage_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(source_path)
    print(f"Uploaded {source_path} to {blob_name}")