import os

class Config:

    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    GEMINI_MODEL = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash"
    )

    XRAY_MODEL_URL = os.getenv("XRAY_MODEL_URL")

    DEBUG = False