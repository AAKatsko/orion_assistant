from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain.schema.document import Document
from chromadb.api.models.Collection import Collection

from ingestion.embedder import Embedder
from ingestion.vector_store import VectorStoreManager, COLLECTION_NAME
from core.config import (VECTOR_DB_PATH, TOP_K_CHUNKS)

class Retriever:
    """
    Класс для Retrieval в ChromaDB.
    """
    def __init__(self, db_path: Path = VECTOR_DB_PATH, k: int = TOP_K_CHUNKS):
        """
        Инициализирует ретривер, подключаясь к ChromaDB и загружая модель эмбеддингов.
        
        Аргументы:
            db_path: Путь к папке, где хранится ChromaDB.
            k: Количество чанков, которое нужно извлечь.
        """
        self.k = k
        self.embedder: Embedder = Embedder()
        self.manager: VectorStoreManager = VectorStoreManager(db_path=db_path)
        
        # Получаем доступ к коллекции ChromaDB
        self.collection: Optional[Collection] = self.manager.get_or_create_collection()
        
        if self.collection:
            print(f"Ретривер инициализирован: подключен к коллекции '{COLLECTION_NAME}' (K={self.k}).")
        else:
            print("Ошибка: Не удалось подключиться к коллекции ChromaDB.")

    def retrieve(self, query: str) -> List[Document]:
        """
        Извлекает k наиболее релевантных чанков из векторной базы по запросу.

        Аргументы:
            query: Пользовательский текстовый запрос.

        Возвращает:
            List[Document]: Список объектов LangChain Document, содержащих 
                            релевантный текст и метаданные.
        """
        if not self.collection:
            print('Коллекция не найдена.')
            return []

        print(f"Поиск релевантного контекста для запроса: '{query[:50]}.'")

        # 1. Векторизация запроса
        # Создаем временный Document для векторизации
        query_embedding = self.embedder.embed_query(query)
        
        # 2. Поиск в ChromaDB
        results: Dict[str, Any] = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.k,
            include=['documents', 'metadatas', 'distances']
        )
        # посмотреть результат results
        
        retrieved_documents: List[Document] = []

        # 3. Форматирование результатов в объекты Document
        if results['documents'] and results['metadatas']:
            # Проходим по результатам (они приходят в виде списков в списках)
            docs = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            print(f"Найдено {len(docs)} релевантных фрагментов.")

            for doc_content, meta, dist in zip(docs, metadatas, distances):
                # Добавляем дистанцию как метаданные для отладки
                meta['distance'] = dist
                
                retrieved_documents.append(
                    Document(page_content=doc_content, metadata=meta)
                )
        
        return retrieved_documents
    
    # @staticmethod
    # def format_context(documents: List[Document]) -> str:
    #     """
    #     Форматирует список найденных документов в единую строку, 
    #     удобную для передачи в промпт LLM.
    #     """
    #     context_parts = []
    #     for i, doc in enumerate(documents):
    #         # Сначала посмотреть что выдает: doc.metadata.get('source', 'Неизвестный источник')
    #         source = doc.metadata.get('source', 'Неизвестный источник')
    #         page = doc.metadata.get('page', '?')
            
    #         # Формат: [1] Источник: filename.pdf (стр. 50)
    #         header = f"[{i+1}] Источник: {source} (стр. {page})"
    #         context_parts.append(f"{header}\n---\n{doc.page_content}")
            
        return "\n\n" + "\n---\n\n".join(context_parts)

if __name__ == "__main__":
    
    # нужно запустить python ingest.py
    if not VECTOR_DB_PATH.exists():
        print("Запустите 'python ingest.py' для наполнения базы.")
    else:
        # Инициализация ретривера
        retriever = Retriever(k=3)
        
        # Пример запроса
        test_query = "Каковы преимущества использования модулей Orion S-Terra и V-Terra?"
        
        # 1. Поиск
        retrieved_chunks = retriever.retrieve(test_query)
        
        if retrieved_chunks:
            # 2. Форматирование контекста для промпта
            context = Retriever.format_context(retrieved_chunks)
            
            print("\n" + "="*50)
            print("КОНТЕКСТ ДЛЯ LLM:")
            print("="*50)
            print(context)

            # Вывод деталей
            print("\n--- Детализация (K=3) ---")
            for i, chunk in enumerate(retrieved_chunks):
                print(f"[{i+1}] Файл: {chunk.metadata.get('filename')}, Стр. {chunk.metadata.get('page')}, Расстояние: {chunk.metadata.get('distance'):.4f}")