APP_TITLE = "ResearchEase AI"

DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

REQUEST_TIMEOUT_SECONDS = 300
ACADEMIC_REQUEST_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------
# Complete-paper analysis
# ---------------------------------------------------------

SUMMARY_CHUNK_SIZE = 6000
SUMMARY_CHUNK_OVERLAP = 500
MAX_SUMMARY_CHUNKS = 20


# ---------------------------------------------------------
# Uploaded-paper RAG
# ---------------------------------------------------------

RAG_CHUNK_SIZE = 1800
RAG_CHUNK_OVERLAP = 250

EMBEDDING_BATCH_SIZE = 16

DEFAULT_TOP_K = 5
MIN_RETRIEVAL_SCORE = 0.15

MAX_CONTEXT_CHARACTERS = 10000
MAX_CHAT_HISTORY_TURNS = 4


# ---------------------------------------------------------
# External academic research
# ---------------------------------------------------------

DEFAULT_EXTERNAL_RESULTS_PER_SOURCE = 5
MAX_EXTERNAL_RESULTS_PER_SOURCE = 10
MAX_EXTERNAL_CONTEXT_CHARACTERS = 14000

SEMANTIC_SCHOLAR_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/search"
)

ARXIV_API_URL = (
    "https://export.arxiv.org/api/query"
)

CROSSREF_API_URL = (
    "https://api.crossref.org/works"
)

APP_USER_AGENT = (
    "ResearchEase-AI/3.0 "
    "(academic research assistant; local Streamlit application)"
)
# ---------------------------------------------------------
# Version 5: Literature review
# ---------------------------------------------------------

LITERATURE_CHUNK_SIZE = 5000
LITERATURE_CHUNK_OVERLAP = 400

# Limits the number of Ollama calls for each uploaded paper.
MAX_LITERATURE_CHUNKS_PER_PAPER = 12

# Includes the primary paper, supporting PDFs, and
# external abstract-level records.
MAX_LITERATURE_SOURCES = 8