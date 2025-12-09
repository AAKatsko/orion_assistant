# orion_assistant/scripts/check_db_documents.py (или можно добавить в vector_store.py)

import sys
from pathlib import Path

# Добавляем корневую папку src в PYTHONPATH для импорта
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from langchain.schema.document import Document
from src.ingestion.vector_store import VectorStoreManager, COLLECTION_NAME
from src.core.config import VECTOR_DB_PATH

def retrieve_and_display_documents(n_documents: int = 2):
    """
    Подключается к ChromaDB, извлекает первые N документов и выводит их в формате LangChain Document.
    """
    print("=" * 60)
    print(f"      👀 ПРОСМОТР ДОКУМЕНТОВ ИЗ CHROMADB ({COLLECTION_NAME})")
    print("=" * 60)

    # Проверка наличия базы
    if not VECTOR_DB_PATH.exists():
        print(f"❌ База данных не найдена по пути: {VECTOR_DB_PATH}")
        print("Пожалуйста, сначала запустите 'python ingest.py'.")
        return

    # 1. Инициализация менеджера и получение коллекции
    manager = VectorStoreManager()
    collection = manager.get_or_create_collection()
    
    if not collection:
        print("❌ Не удалось получить коллекцию.")
        return

    total_count = collection.count()
    print(f"Текущее количество документов в коллекции: {total_count}")

    if total_count == 0:
        print("Коллекция пуста.")
        return

    # 2. Получение данных
    # Используем метод get(), чтобы извлечь документы и метаданные.
    # Ограничиваемся первыми N документами.
    
    # NOTE: ChromaDB не гарантирует порядок, но мы можем использовать get() для извлечения.
    # Чтобы получить 'первые' N документов, нам нужно получить их ID, 
    # но поскольку при индексации мы использовали IDs вида 'doc_0', 'doc_1',
    # мы можем просто взять их для извлечения.

    # Используем get() для извлечения данных по ID
    ids_to_fetch = [f"doc_{i}" for i in range(min(n_documents, total_count))]

    try:
        results = collection.get(
            ids=ids_to_fetch,
            include=['documents', 'metadatas']
        )
    except Exception as e:
        # Если IDs не найдены (например, база была перезаписана)
        print(f"Ошибка при извлечении документов по ID: {e}")
        # Попробуем извлечь без указания ID (если IDs не последовательные)
        results = collection.peek(limit=n_documents)

    # 3. Преобразование результатов в формат Document
    if results['documents']:
        print(f"\nВывод первых {len(results['documents'])} документов:")
        
        for i, (content, meta) in enumerate(zip(results['documents'], results['metadatas'])):
            
            # Создаем объект Document
            doc = Document(
                page_content=content,
                metadata=meta
            )
            
            # Выводим в нужном формате
            print("-" * 30)
            print(f"[{i+1}] Извлеченный объект Document:")
            print("Document(")
            # Используем repr() для корректного отображения строк
            print(f"    page_content={repr(doc.page_content[:120] + '...')},")
            print(f"    metadata={doc.metadata}")
            print(")")
        print("-" * 30)
    else:
        print("Не удалось извлечь документы.")


if __name__ == "__main__":
    # Укажите, сколько документов вы хотите просмотреть
    retrieve_and_display_documents(n_documents=3)