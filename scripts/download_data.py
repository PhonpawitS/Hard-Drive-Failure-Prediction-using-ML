import os
import requests
import zipfile
from io import BytesIO

# ====== CONFIG ======
BASE_URL = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q3_2025.zip"
OUTPUT_DIR = "data/raw"

# ====================

def download_and_extract(url, output_dir):
    print(f"Downloading from {url} ...")

    response = requests.get(url)
    response.raise_for_status()

    print("Download complete. Extracting...")

    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(output_dir)

    print(f"Data extracted to {output_dir}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    download_and_extract(BASE_URL, OUTPUT_DIR)
