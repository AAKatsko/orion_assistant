from typing import List
from langchain.schema.document import Document

SYSTEM_PROMPT = """
1. Роль
Ты — OrionGPT, внутренний "умный" ассистент для сотрудников (инженеров, BDM, presale) компании Orion soft.
2. Задача
Твоя главная задача — предоставлять **точные, верифицированные ответы** на вопросы, используя **ТОЛЬКО** предоставленный тебе контекст.

3. Инструкции и Правила:
- **ТОЧНОСТЬ:** Отвечай на вопрос, используя информацию ИСКЛЮЧИТЕЛЬНО из раздела <КОНТЕКСТ>.
- **НЕ ВРАТЬ:** Если предоставленный контекст НЕ СОДЕРЖИТ информации, необходимой для ответа на вопрос пользователя, ты должен честно и прямо ответить: 
   "К сожалению, в предоставленной технической документации Orion soft информация по данному вопросу отсутствует." Никогда не придумывай факты.
- **ЦИТИРОВАНИЕ:** В конце твоего ответа всегда должен быть отдельный раздел <ИСТОЧНИКИ>. В нем ты перечисляешь все документы, которые использовал в своем ответе. Формат: [1] Источник: filename.pdf (стр. 50)
"""

TEMPLATE = """
{system_prompt}

Ниже приведен контекст, извлеченный из корпоративной документации:
<КОНТЕКСТ>
{formatted_context}
</КОНТЕКСТ>

<ВОПРОС ПОЛЬЗОВАТЕЛЯ>
{user_query}
</ВОПРОС ПОЛЬЗОВАТЕЛЯ>

Теперь, основываясь строго на предоставленной выше информации, дай полный и точный ответ.
"""

class PromptBuilder:
    """
    Класс для сборки финального промпта для LLM, включающего
    системные инструкции, контекст и вопрос пользователя.
    """
    def __init__(self, system_prompt: str = SYSTEM_PROMPT, template: str = TEMPLATE):
        self.system_prompt = system_prompt
        self.template = template
        print("PromptBuilder инициализирован.")

    @staticmethod
    def format_context_for_prompt(documents: List[Document]) -> str:
        """
        Форматирует список найденных документов (чанки) в единую строку, 
        удобную для чтения LLM, включая метаданные для цитирования.
        """
        context_parts = []
        for i, doc in enumerate(documents):
            # Извлекаем метаданные для цитирования
            source = doc.metadata.get('source', 'Неизвестный источник')
            page = doc.metadata.get('page', '?')
            
            # Формат: [ФРАГМЕНТ 1] Источник: nova/документ.pdf, страница 50
            header = f"[ФРАГМЕНТ {i+1}] Источник: {source}, страница {page}"
            
            # Собираем фрагмент: заголовок + содержимое
            context_parts.append(f"{header}\n{doc.page_content}")
            
        return "\n\n" + "\n---\n\n".join(context_parts)

    def build_rag_prompt(self, user_query: str, context_documents: List[Document]) -> str:
        """
        Собирает полный промпт RAG.

        Аргументы:
            user_query: Вопрос пользователя.
            context_documents: Список релевантных документов (чанки) от ретривера.

        Возвращает:
            str: Финальный промпт для отправки в LLM.
        """
        # Форматируем контекст с помощью статического метода
        formatted_context = self.format_context_for_prompt(context_documents)
        
        # Заполняем шаблон
        final_prompt = self.template.format(
            system_prompt=self.system_prompt,
            formatted_context=formatted_context,
            user_query=user_query
        )
        
        return final_prompt

if __name__ == "__main__":
    
    # Имитация данных, полученных от retriever
    mock_chunks = [
        Document(
            page_content="Модуль Metrics – решение, реализующее новый подход к сбору метрик с платформы виртуализации zVirt. Выполняется сбор 119 метрик, которые разделены на 5 категорий.",
            metadata={'filename': 'zvirt_metrics.pdf', 'source': 'zvirt_metrics.pdf', 'page': 1}
        ),
        Document(
            page_content="Модуль zVirt Metrics предназначен для мониторинга и аналитики загрузки инфраструктуры виртуализации в едином интерфейсе.",
            metadata={'filename': 'latest_admin-guide.pdf', 'source': 'latest_admin_guide.pdf', 'page': 1}
        )
    ]
    
    test_question = "Что такое Модуль zVirt Metrics?"
    
    # 1. Сборка промпта
    builder = PromptBuilder()
    final_prompt = builder.build_rag_prompt(test_question, mock_chunks)
    
    # 2. Вывод результата
    print("ФИНАЛЬНЫЙ ПРОМПТ:")
    print(final_prompt)