from typing import List, Optional
from pathlib import Path
import chromadb

from langchain.schema.document import Document
from chromadb.api.models.Collection import Collection

from src.core.config import VECTOR_DB_PATH, BASE_DIR, COLLECTION_NAME
from src.ingestion.embedder import Embedder
from src.ingestion.text_splitter import TextSplitter
from src.ingestion.downloader import DataLoader

class VectorStoreManager:
    def __init__(self, db_path: Path = VECTOR_DB_PATH):
        """
        Инициализирует Chroma DB
        """
        self.db_path = db_path
        self.db_path.mkdir(parents=True, exist_ok=True)

        print(f"Инициализация ChromaDB клиент, путь: {self.db_path}")

        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.embedder = Embedder()
        self.embedding_dimension = self.embedder.get_embedding_dimension()
        self.collection = COLLECTION_NAME

    def get_or_create_collection(self) -> Optional[Collection]:
        """
        Получает существующую коллекцию или создает новую, 
        используя размерность векторов.
        """
        if self.embedding_dimension == 0:
            print("Ошибка: Размерность эмбеддингов равна нулю. Невозможно создать коллекцию.")
            return None

        print(f"Получение/создание коллекции '{COLLECTION_NAME}'.")
        
        collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        current_count = collection.count()
        print(f"Коллекция '{COLLECTION_NAME}' готова. Текущее количество документов: {current_count}")
        return collection

    def index_documents(self, chunks: List[Document]) -> bool:
        """
        Генерирует эмбеддинги для чанков и добавляет их в ChromaDB.
        
        Аргументы:
            chunks: Список объектов LangChain Document (текстовые чанки).
            
        Возвращает:
            bool: True, если индексация прошла успешно.
        """
        collection = self.get_or_create_collection()
        if not collection:
            return False

        if not chunks:
            print("Список чанков пуст. Индексация отменена.")
            return False

        # Генерация эмбеддингов
        embeddings_list = self.embedder.embed_documents(chunks)

        if not embeddings_list or len(embeddings_list) != len(chunks):
            print("Не удалось сгенерировать эмбеддинги для всех чанков.")
            return False

        # Подготовка данных для ChromaDB
        ids = [f"doc_{i}" for i in range(len(chunks))]
        documents = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Добавление данных в коллекцию
        print(f"Добавление {len(documents)} документов в ChromaDB.")
        try:
            collection.add(
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print("Индексация завершена успешно.")
            print(f"Общее количество документов в коллекции: {collection.count()}")
            return True
            
        except Exception as e:
            print(f"Ошибка при добавлении в ChromaDB: {e}")
            return False

# if __name__ == "__main__":
#     manager = VectorStoreManager()
#     manager.index_documents(chunks)