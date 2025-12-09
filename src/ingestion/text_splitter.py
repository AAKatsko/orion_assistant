# ПОКА ЧТО ИГНОРИРУЕМ ФОТО
# Вариант для улучшения: PyMuPDF
from typing import List
from pathlib import Path
import pypdf
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from src.core.config import (
    RAW_DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS
)

class TextSplitter:
    """
    Класс для загрузки PDF-документов с помощью pypdf и разбиения их на чанки.
    """
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Инициализирует сплиттер с заданными параметрами.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )
        print(f"TextSplitter инициализирован: размер чанка={chunk_size}, перекрытие={chunk_overlap}")

    def load_documents(self, data_path: Path = RAW_DATA_PATH) -> List[Document]:
        """
        Рекурсивно загружает все PDF-файлы из указанной папки, используя pypdf.

        Аргументы:
            data_path: Базовый путь, откуда начинать поиск PDF-файлов (data/raw).

        Возвращает:
            List[Document]: Список объектов LangChain Document, где каждый объект — это страница.
        """
        print(f"Начало загрузки документов из: {data_path}")
        all_documents: List[Document] = []
        
        # поиск всех PDF-файлов в подпапках
        pdf_files = list(data_path.rglob("*.pdf"))

        for file_path in pdf_files:
            try:
                reader = pypdf.PdfReader(file_path)
                
                # Извлекаем текст постранично
                for i, page in enumerate(reader.pages):
                    page_content = page.extract_text()
                    
                    if page_content:
                        # Создаем объект Document для каждой страницы
                        doc = Document(
                            page_content=page_content,
                            metadata={
                                'source': str(file_path.relative_to(data_path)), # Путь относительно 'raw'
                                'filename': file_path.name,
                                'page': i + 1, # Номер страницы, начиная с 1
                            }
                        )
                        all_documents.append(doc)
                
                # print(f"Обработан файл: {file_path.name} ({len(reader.pages)} страниц)")
                
            except Exception as e:
                print(f"Ошибка при загрузке файла {file_path}: {e}")
                
        print(f"Всего загружено {len(all_documents)} страниц.")
        return all_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Разбивает список документов (страниц) на текстовые чанки.

        Аргументы:
            documents: Список объектов LangChain Document (страниц).

        Возвращает:
            List[Document]: Список текстовых чанков (фрагментов).
        """
        print(f"Начало разбиения {len(documents)} страниц на чанки...")
        
        # Разбиение, сохраняющее метаданные страниц
        chunks = self.splitter.split_documents(documents)
        
        print(f"Разбиение завершено. Создано {len(chunks)} чанков.")
        return chunks

# if __name__ == "__main__":
    
#     # проверка, что папка data/raw/ существует и содержит PDF-файлы
#     if not RAW_DATA_PATH.exists() or not list(RAW_DATA_PATH.rglob("*.pdf")):
#         print(f"Ошибка извлечения данных из папки: {RAW_DATA_PATH}")
#     else:
#         # 1. Инициализация
#         splitter = TextSplitter()
        
#         # 2. Загрузка
#         loaded_pages = splitter.load_documents(RAW_DATA_PATH)
        
#         if loaded_pages:
#             # 3. Разбиение
#             text_chunks = splitter.split_documents(loaded_pages)
#             print(text_chunks[0])