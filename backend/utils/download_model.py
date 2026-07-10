import os
import requests

MODEL_URL = (
    "https://huggingface.co/sambhavmishra1/"
    "medisense-xray-model/resolve/main/xray_model.hdf5"
)

MODEL_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "ml",
        "models"
    )
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "xray_model.hdf5"
)


def download_xray_model():
    """
    Downloads the X-ray CNN model only if it does not exist.
    """

    if os.path.exists(MODEL_PATH):
        print("[Download] Model already exists.")
        return MODEL_PATH

    print("[Download] Creating model directory...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("[Download] Downloading X-ray model...")
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                if total:
                    percent = downloaded * 100 / total
                    print(f"\rDownloading... {percent:.1f}%", end="")

    print("\n[Download] Model downloaded successfully.")

    return MODEL_PATH