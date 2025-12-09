from typing import List
import requests
import json
from langchain.schema.document import Document

from src.core.config import (LLM_API_URL, LLM_TOKEN, LLM_MODEL_NAME, LLM_MAX_TOKENS)

class LLMClient:
    def __init__(self):
        self.api_url = LLM_API_URL
        self.token = LLM_TOKEN
        self.model_name = LLM_MODEL_NAME
        self.headers = {'Authorization': f"Bearer {LLM_API_TOKEN}",
        'Content-Type': "application/json"}
        print("LLMClient инициализирован.")

    def generate_response(self, query: str, context: List[Document]) -> str:
        """
        Отправляет запрос на генерацию ответа в LLM.

        Аргументы:
            query: Вопрос пользователя.
            context: Список релевантных чанков.
            
        Возвращает:
            str: Сгенерированный ответ LLM или сообщение об ошибке.
        """
        payload = {
            "model": self.model_name,
            "max_tokens": self.max_tokens
            }
        
        print(f"Отправка запроса к {self.model_name}.")
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, verify=False)
            response.raise_for_status()

            response_data = response.json()[0]['generated_text']
            
            # Извлечение сгенерированного текста
            if response_data:
                return response_data
            else:
                return f"LLM не вернула ответ. Детали: {response_data}"

        except requests.exceptions.RequestException as e:
            return f"Ошибка HTTP-запроса к LLM: {e}"
        except json.JSONDecodeError:
            return "Ошибка декодирования JSON-ответа от LLM."
        except Exception as e:
            return f"Ошибка при генерации ответа: {e}"