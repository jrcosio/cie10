import os, dotenv
from qdrant_client import QdrantClient
from google import genai

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "Variable de entorno GEMINI_API_KEY no definida. "
        "Ejecútalo con: set GEMINI_API_KEY=tu_clave (Windows) o export GEMINI_API_KEY=tu_clave (Linux/Mac)"
    )

MODEL_NAME       = "gemini-2.5-flash-lite"
# MODEL_NAME       = "gemini-3.1-flash-lite-preview"
EMBEDDING_MODEL  = "gemini-embedding-001"
COLLECTION_NAME  = "cie10_catalogo"

client = genai.Client(api_key=GEMINI_API_KEY)
qdrant = QdrantClient(url="http://localhost:6333")
