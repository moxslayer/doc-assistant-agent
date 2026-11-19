import os
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool

# Импортируем наши существующие функции
from src.rag import rag_answer
from src.classifier import classify_text

# 1. Инициализация локальной модели Ollama для агента
llm = ChatOllama(
    model="gemma2:2b",
    base_url=os.getenv("OLLAMA_HOST", "http://ollama:11434"),
    temperature=0.1
)

# 2. Определение инструментов (Tools)
@tool
def rag_search(query: str) -> str:
    """Ищет информацию в базе отзывов о лекарствах. 
    Используй этот инструмент, когда пользователь спрашивает о побочных эффектах, 
    опыте применения или отзывах о конкретном препарате."""
    # Здесь нужен доступ к index_state. В реальном приложении его можно передать через контекст.
    # Для простоты, мы предполагаем, что index_state доступен глобально или передаётся.
    # В вашем текущем app.py index_state инициализируется при старте.
    # Это упрощённый пример.
    # В идеале, нужно передавать index_state в агента.
    # Пока что, сделаем заглушку, которая будет работать, если index_state доступен.
    # В следующем шаге мы покажем, как интегрировать это в app.py.
    return "Функция RAG-поиска должна быть вызвана с доступом к индексу."

@tool
def classify_sentiment(text: str) -> str:
    """Классифицирует текст отзыва как положительный или отрицательный.
    Используй этот инструмент, когда пользователь просит оценить тональность 
    или отношение в тексте."""
    result = classify_text(text)
    return f"Тональность: {result['label']} (уверенность: {result['confidence']})"

# 3. Создание агента
agent = create_agent(
    model=llm,
    tools=[rag_search, classify_sentiment],
    system_prompt="Ты — полезный ассистент-фармацевт. Используй инструменты для ответа на вопросы."
)