"""
Configuration management for RBAC-RAG Chatbot backend.
Loads settings from environment variables with safe defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load the repository .env regardless of the process working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class Config:
    """Application configuration from environment variables."""

    # LLM Models
    GEMINI_CHAT_MODEL = os.getenv(
        "GEMINI_CHAT_MODEL",
        "gemini-3.6-flash"
    )
    # Local model used for both indexing and querying Chroma.
    EMBEDDING_MODEL = os.getenv(
        "LOCAL_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Database
    VECTOR_DB_PATH = os.getenv(
        "VECTOR_DB_PATH",
        "vector_store_db"
    )
    VECTOR_DB_COLLECTION_NAME = os.getenv(
        "CHROMA_COLLECTION_NAME",
        "documentation"
    )

    # JWT authentication
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
    TOKEN_EXPIRY_MINUTES = int(os.getenv("TOKEN_EXPIRY_MINUTES", "1440"))

    # Backend Server
    BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
    BACKEND_PORT = int(os.getenv("BACKEND_PORT", "6001"))
    ALLOWED_ORIGINS = [
        origin.strip()
        for origin in os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:8501,http://127.0.0.1:8501"
        ).split(",")
        if origin.strip()
    ]

    # Frontend Backend URL
    BACKEND_URL = os.getenv(
        "BACKEND_URL",
        "http://localhost:6001"
    )

def get_vector_db_path() -> Path:
    """
    Get the vector database path.
    Returns absolute path if relative path is provided.
    """
    db_path = Config.VECTOR_DB_PATH
    
    if not os.path.isabs(db_path):
        # If relative, make it relative to project root (parent of backend/)
        project_root = Path(__file__).parent.parent.parent
        db_path = project_root / db_path
    
    return Path(db_path)


def validate_config() -> list:
    """
    Validate configuration at startup.
    Returns a list of warning/error messages (empty if all OK).
    """
    issues = []

    # Check Google API Key
    if not os.getenv("GOOGLE_API_KEY"):
        issues.append(
            "❌ ERROR: GOOGLE_API_KEY not set in .env. "
            "Gemini API will fail without this."
        )

    if not Config.JWT_SECRET_KEY:
        issues.append(
            "ERROR: JWT_SECRET_KEY is not set. Configure a private signing secret."
        )
    elif len(Config.JWT_SECRET_KEY) < 32:
        issues.append(
            "ERROR: JWT_SECRET_KEY must contain at least 32 characters."
        )

    # Check vector DB path
    db_path = get_vector_db_path()
    if not db_path.exists():
        issues.append(
            f"⚠️  WARNING: Vector database path does not exist: {db_path}. "
            f"Run indexing script to create it."
        )
    else:
        chroma_db = db_path / "chroma.sqlite3"
        if not chroma_db.exists():
            issues.append(
                f"⚠️  WARNING: Chroma database not found at {chroma_db}. "
                f"Run indexing script to create it."
            )

    # Validate chat model
    deprecated_models = ["gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.0-pro", "gemini-3.0-pro", "gemini-2.5-flash"]
    if Config.GEMINI_CHAT_MODEL in deprecated_models:
        issues.append(
            f"❌ ERROR: Chat model '{Config.GEMINI_CHAT_MODEL}' is no longer available "
            f"to new users. Update GEMINI_CHAT_MODEL in .env to one of: "
            f"gemini-3.0-pro, gemini-3.0-flash, gemini-3.0-flash-lite, gemini-4.0"
        )

    return issues


def test_models_availability() -> dict:
    """
    Test if the configured models are actually available.
    Returns dict with 'chat_model' and 'embedding_model' keys (True/False).
    
    This is called optionally to verify models work before starting the app.
    """
    results = {"chat_model": False, "embedding_model": False}

    # Test chat model
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = ChatGoogleGenerativeAI(model=Config.GEMINI_CHAT_MODEL)
        # Try to initialize (don't invoke)
        results["chat_model"] = True
    except Exception as e:
        results["chat_model"] = False
        print(f"Chat model test failed: {e}")

    # Verify the local embedding integration can be initialized.
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embed = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        results["embedding_model"] = True
    except Exception as e:
        results["embedding_model"] = False
        print(f"Embedding model test failed: {e}")

    return results


if __name__ == "__main__":
    # Test configuration
    print("Configuration:")
    print(f"  Chat Model: {Config.GEMINI_CHAT_MODEL}")
    print(f"  Local Embedding Model: {Config.EMBEDDING_MODEL}")
    print(f"  Vector DB Path: {get_vector_db_path()}")
    print(f"  Backend: {Config.BACKEND_HOST}:{Config.BACKEND_PORT}")
    print(f"  Frontend Backend URL: {Config.BACKEND_URL}")
    print()
    print("Validation Issues:")
    issues = validate_config()
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  None - configuration OK")
