from pathlib import Path
import torch

# Определяем базовую директорию проекта (orion_assistant/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR_NAME = "data"
RAW_DATA_FOLDER_NAME = "raw"
FOLDER_STRUCTURE_FILE = "folder_structure.json"

# Полные пути
DATA_PATH = BASE_DIR / DATA_DIR_NAME
RAW_DATA_PATH = DATA_PATH / RAW_DATA_FOLDER_NAME

# Яндекс диск
YANDEX_DISK_PUBLIC_KEY = 'https://disk.360.ru/d/ZWfcuA3Wi1BCiA'
YANDEX_DISK_BASE_URL = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'

# Имена файлов/папок в процессе загрузки
YAD_ZIP_FILENAME = "AI_Boostcamp.zip"
YAD_EXTRACTED_FOLDER = "AI BoostCamp"
PDF_ZIP_EXTRACTED_FOLDER = "All_PDFs_merged_1"

# чанкинг
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = [
    "\n\n",
    "\n",
    " ",
]

# Модель эмбедингов
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ChromaDB
VECTOR_DB_PATH = DATA_PATH / "vectordb"
COLLECTION_NAME = "orion_assistant_docs"

# Retriever
TOP_K_CHUNKS = 5 

# LLM
LLM_API_URL = "https://inference.product.nova.neurotech.k2.cloud"
LLM_TOKEN = "qwen2 oOv0w4yv5QxeAlgm8VL"
LLM_MODEL_NAME = "Qwen2.5-32B"
LLM_MAX_TOKENS = 1024