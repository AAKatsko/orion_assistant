from typing import List
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import torch

from src.core.config import (
    RAW_DATA_PATH, EMBEDDING_MODEL_NAME, DEVICE
)

class Embedder:
    """
    Класс для загрузки модели эмбеддингов и генерации векторных представлений текста.
    """
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: str = DEVICE):
        """
        Инициализирует модель эмбеддингов.
        """
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            print(f"Модель успешно загружена. Размерность эмбеддингов: {self.embedding_dimension}")
            
        except Exception as e:
            print(f"Ошибка при загрузке модели {model_name}: {e}")
            self.model = None
            self.embedding_dimension = 0

    def embed_documents(self, chunks: List[Document]) -> List[List[float]]:
        """
        Генерирует эмбеддинги для списка объектов LangChain Document.
        
        Аргументы:
            chunks: Список чанков (LangChain Document).

        Возвращает:
            List[List[float]]: Список векторов (эмбеддингов).
        """
        if not self.model:
            print("Модель эмбеддингов не загружена.")
            return []
            
        texts = [chunk.page_content for chunk in chunks]
        print(f"Начало векторизации {len(texts)} текстовых чанков.")
        
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=False,
                show_progress_bar=True)

            print(f"Генерация завершена. Создано {len(embeddings)} векторов.")
            return embeddings.tolist()
        except Exception as e:
            print(f"Ошибка при генерации эмбеддингов: {e}")
            return []

    # Для векторизации запроса пользователя
    def embed_query(self, query: str) -> List[float]:
        """
        Генерирует эмбеддинг для одного запроса.
        """
        if not self.model:
            print('Ошибка: Модель эмбеддингов не загружена.')
            return []

        # query_with_prefix = [f"query: {query}"]
        
        try:
            embedding = self.model.encode(
                query,
                convert_to_tensor=False
            )
            # Возвращаем первый и единственный вектор из списка
            return embedding[0].tolist()
            
        except Exception as e:
            print(f"Ошибка при генерации эмбеддинга запроса: {e}")
            return None

    def get_embedding_dimension(self):
        return self.embedding_dimension

# if __name__ == "__main__":
    
#     embedder = Embedder()
#     embeddings = embedder.embed_documents(chunks)

#     if embeddings:
#         print('Check result')
#         print(f'Embeddings genered: {len(embeddings)}')
#         print(f'Embedding dimension: {embedder.get_embedding_dimension}')
#         print('Example')
#         print(f'Size of 0 embed: {len(embeddings[0])}')
#         print(embeddings[0][0])