import sys
from pathlib import Path

# Добавляем корневую папку src в PYTHONPATH, чтобы импортировать модули
# Это необходимо, если скрипт запускается извне папки src
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from src.ingestion.downloader import DataLoader
from src.ingestion.text_splitter import TextSplitter
from src.ingestion.vector_store import VectorStoreManager
from src.core.config import RAW_DATA_PATH

def run_ingestion_pipeline():
    """
    Оркестрирует полный пайплайн индексации документов:
    Загрузка -> Разбиение -> Векторизация и Сохранение в ChromaDB.
    """
    print("Запуск Ingestion-пайплайна")
    
    # загрузка данных
    print("1. Проверка и загрузка исходных PDF-документов")
    
    # Проверка наличия PDF-файлов
    if not list(RAW_DATA_PATH.rglob("*.pdf")):
        print("Исходные PDF-файлы не найдены в data/raw. Запускаем загрузчик.")
        downloader = DataLoader()
        if not downloader.download_and_prepare_data():
            print("Ошибка загрузки данных. Пайплайн остановлен.")
            return

    # ЗАГРУЖАЮ ТОЛЬКО ЧАСТЬ ДЛЯ ТЕСТА
    TEST_DATA_PATH = RAW_DATA_PATH / "zvirt-metrics"
    
    # разбиение на чанки
    print("2. Загрузка документов и разбиение их на чанки")
    splitter = TextSplitter()
    loaded_pages = splitter.load_documents(TEST_DATA_PATH)
    if not loaded_pages:
        print("Документы не загружены. Пайплайн остановлен.")
        return

    # Разбиваем страницы на мелкие чанки
    chunks = splitter.split_documents(loaded_pages)
    if not chunks:
        print("Не удалось создать чанки. Пайплайн остановлен.")
        return

    # Эмбеддинги и векторизация
    print("3. Генерация эмбеддингов и сохранение в ChromaDB")
    
    manager = VectorStoreManager()
    
    if manager.index_documents(chunks):
        # Эмбеддинги генерируются при вызове index_documents(chunks)
        print("Эмбеддинги сгенерированы")
        print(f"Документация готова к поиску в коллекции '{manager.collection}'.")
    else:
        print("Ошибка генерации эмбеддингов.")

if __name__ == "__main__":
    run_ingestion_pipeline()

# запуск: python3 -m src.ingestion.ingest.py