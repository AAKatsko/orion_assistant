import os
import requests
import zipfile
import shutil
import json
from pathlib import Path
from urllib.parse import urlencode

from src.core.config import (
    RAW_DATA_PATH, DATA_PATH, YANDEX_DISK_PUBLIC_KEY,
    YANDEX_DISK_BASE_URL, YAD_ZIP_FILENAME, YAD_EXTRACTED_FOLDER,
    PDF_ZIP_EXTRACTED_FOLDER, FOLDER_STRUCTURE_FILE
)

class DataLoader:
    """
    Класс для загрузки, распаковки и подготовки корпоративной документации.
    """
    def __init__(self):
        print("Инициализация Downloader")

    def _cleanup_data_folder(self, data_path: Path):
        """
        Очищает папку 'data/', удаляя все, кроме папки с PDF-файлами,
        которая будет переименована в 'raw'.
        """
        try:
            if not data_path.exists():
                print(f"Папка {data_path} не существует. Создаем.")
                data_path.mkdir(parents=True, exist_ok=True)
                return

            print(f"Начало очистки папки: {data_path}")

            for item in data_path.iterdir():
                # Сохраняем папку с PDF, которая еще не переименована в 'raw'
                if item.name == PDF_ZIP_EXTRACTED_FOLDER:
                    print(f"Временная папка с PDF сохранена: {item.name}")
                    continue
                # Сохраняем папку 'raw', если она уже существует
                if item.name == RAW_DATA_PATH.name and item.is_dir():
                    print(f"Текущая папка 'raw' сохранена: {item.name}")
                    continue

                # Удаляем все остальные файлы/папки
                try:
                    if item.is_file():
                        item.unlink()
                        print(f"Удален файл: {item.name}")
                    elif item.is_dir():
                        shutil.rmtree(item)
                        print(f"Удалена папка: {item.name}")
                except Exception as e:
                    print(f"Ошибка при удалении {item}: {e}")

            print(f"Очистка папки {data_path} завершена.")

        except Exception as e:
            print(f"Общая ошибка при очистке папки: {e}")

    def download_and_prepare_data(self):
        """
        Основной метод: скачивает, распаковывает и подготавливает данные.
        """
        print("Запуск загрузки и подготовки данных")
        
        # 1. Формируем URL для скачивания
        yad_download_url = YANDEX_DISK_BASE_URL + urlencode(dict(public_key=YANDEX_DISK_PUBLIC_KEY))
        zip_path = DATA_PATH / YAD_ZIP_FILENAME
        
        DATA_PATH.mkdir(parents=True, exist_ok=True)

        try:
            # 2. Скачиваем ZIP-архив
            print(f"Скачивание {YAD_ZIP_FILENAME}")
            response = requests.get(yad_download_url)
            response.raise_for_status()
            
            download_url = response.json()['href']
            download_response = requests.get(download_url)
            download_response.raise_for_status()

            with open(zip_path, 'wb') as f:
                f.write(download_response.content)
            print(f"Архив успешно скачан: {zip_path}")

            # 3. Распаковываем внешний архив (AI_Boostcamp.zip)
            print(f'Распаковка архива Я.Диска')
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_PATH)
            zip_path.unlink()
            print(f'Архив ЯД распакован. Исходный ZIP удален.')

            # Путь к ZIP-файлу с PDF внутри
            pdf_zip_path = DATA_PATH / YAD_EXTRACTED_FOLDER / "All_PDFs_merged_1.zip"
            if not pdf_zip_path.exists():
                raise FileNotFoundError(f"Не найден файл: {pdf_zip_path}")

            # 4. Распаковываем архив с PDF-файлами
            print(f'Распаковка архива с PDF-файлами...')
            with zipfile.ZipFile(pdf_zip_path, 'r') as zip_ref:
                filtered_files = [f for f in zip_ref.namelist() if not f.startswith('__MACOSX/')]
                for file in filtered_files:
                    zip_ref.extract(file, DATA_PATH)
            print(f'Архив PDF распакован.')
            
            # 5. Очистка и переименование
            self._cleanup_data_folder(DATA_PATH)

            # Переименовываем папку с PDF в 'raw'
            source_folder = DATA_PATH / PDF_ZIP_EXTRACTED_FOLDER
            if source_folder.exists():
                # Удаляем старую папку RAW_DATA_PATH, если она существует
                if RAW_DATA_PATH.exists():
                    shutil.rmtree(RAW_DATA_PATH)
                    print(f"Старая папка {RAW_DATA_PATH.name} удалена.")

                source_folder.rename(RAW_DATA_PATH)
                print(f"Папка с документацией переименована: {source_folder.name} -> {RAW_DATA_PATH.name}")
            else:
                raise FileNotFoundError(f"Исходная папка PDF {source_folder} не найдена после распаковки.")

            # 6. Создание метаданных
            metadata = self._get_folder_structure(RAW_DATA_PATH)
            output_path = DATA_PATH / FOLDER_STRUCTURE_FILE
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"Структура папок сохранена в: {output_path}")

            print("Загрузка данных завершена успешно")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Ошибка HTTP-запроса при скачивании: {e}")
            return False
        except Exception as e:
            print(f"Непредвиденная ошибка при загрузке: {e}")
            return False

    def _get_folder_structure(self, base_path: Path) -> dict:
        """
        Сохраняет метаданные о структуре папок с PDF-файлами.
        """
        folder_stats = {}
        total_pdfs = 0

        for root, dirs, files in os.walk(base_path):
            pdf_count = sum(1 for f in files if f.lower().endswith('.pdf'))
            
            if pdf_count > 0:
                rel_path = os.path.relpath(root, base_path)
                if rel_path == '.':
                    rel_path = '/'
                
                folder_stats[rel_path] = pdf_count
                total_pdfs += pdf_count

        metadata = {
            "total_pdfs": total_pdfs,
            "total_folders": len(folder_stats),
            "folders": folder_stats,
            # Отображаем путь относительно корня проекта для удобства
            "data_path": str(RAW_DATA_PATH.relative_to(RAW_DATA_PATH.parent.parent)) 
        }
        
        return metadata

# if __name__ == "__main__":
#     downloader = DataLoader()
#     downloader.download_and_prepare_data()