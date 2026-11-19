from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import sys

# Добавляем путь, чтобы импорты работали корректно
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import build_index, rag_answer
from src.classifier import init_classifier, classify_text

# --- Инициализация ИИ-Агента ---
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage

index_state = None
ai_agent = None

# Объявляем инструменты для агента
@tool
def rag_search(query: str) -> str:
    """Ищет информацию в базе отзывов о лекарствах. 
    Используй этот инструмент, когда пользователь спрашивает о побочных эффектах, 
    опыте применения или отзывах о конкретном препарате."""
    if not index_state:
        return "Ошибка: База данных еще не загружена."
    
    answer, sources = rag_answer(query, index_state, k=3)
    context = "\n".join([f"Файл: {s['filename']}\nТекст: {s['preview'][:200]}..." for s in sources])
    return f"Ответ поиска: {answer}\n\nНайденные источники:\n{context}"

@tool
def classify_sentiment(text: str) -> str:
    """Классифицирует текст отзыва как положительный (Positive) или отрицательный (Negative).
    Используй этот инструмент, когда пользователь просит оценить тональность текста."""
    result = classify_text(text)
    return f"Тональность: {result['label']} (уверенность: {result['confidence']})"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global index_state, ai_agent
    
    print("Инициализация RAG, классификатора и Агента...")
    # 1. Строим индекс RAG
    index_state = build_index()
    
    # 2. Инициализируем классификатор
    init_classifier()
    
    # 3. Создаем "мозг" агента на базе Ollama
    llm = ChatOllama(
        model="qwen2.5:1.5b",
        base_url=os.getenv("OLLAMA_HOST", "http://ollama:11434"),
        temperature=0.1
    )
    
    # 4. Собираем агента из модели и инструментов
    ai_agent = create_agent(
        model=llm,
        tools=[rag_search, classify_sentiment],
        system_prompt="Ты — полезный ассистент-фармацевт. Используй предоставленные инструменты для ответа на вопросы. Если нужно найти отзывы, используй rag_search. Если нужно понять эмоции текста, используй classify_sentiment."
    )
    
    print("Система полностью готова к работе!")
    yield
    print("Выключение сервера...")


app = FastAPI(title="Doc Assistant API", lifespan=lifespan)

class Query(BaseModel):
    question: str
    k: int = 3

class TextToClassify(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(q: Query):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Пустой вопрос")
    answer, sources = rag_answer(q.question, index_state, k=q.k)
    return {"answer": answer, "sources": sources}

@app.post("/classify")
def classify(req: TextToClassify):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Пустой текст")
    return classify_text(req.text)

@app.post("/agent_ask")
async def agent_ask(q: Query):
    """Эндпоинт, использующий ИИ-агента для автономного выбора инструментов."""
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Пустой вопрос")
    
    try:
        # Агент сам решает, что делать
        response = await ai_agent.ainvoke({
            "messages": [HumanMessage(content=q.question)]
        })
        
        # Извлекаем финальный ответ
        final_answer = response["messages"][-1].content
        
        # Пытаемся найти источники, если агент использовал RAG (для красивого вывода)
        sources = []
        # Простая эвристика: если в ответе есть ответ RAG, найдем его в индексе еще раз
        # (В идеале агент должен возвращать структурированный ответ, но для простоты UI сделаем так)
        _, raw_sources = rag_answer(q.question, index_state, k=q.k)
        
        # Если ответ агента похож на "не найдено", то источники не нужны
        if "не найдено" not in final_answer.lower() and "нет информации" not in final_answer.lower():
             sources = raw_sources

        return {"answer": final_answer, "sources": sources}
        
    except Exception as e:
        return {"answer": f"Ошибка агента: {e}", "sources": []}


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
        <head>
            <title>Doc Assistant (Agent)</title>
            <style>
                body { font-family: Arial; max-width: 800px; margin: 40px auto; background: #f4f4f9; }
                .chat-box { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
                input[type="text"] { width: 75%; padding: 10px; font-size: 16px; border: 1px solid #ccc; border-radius: 5px; }
                button { padding: 10px 20px; font-size: 16px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
                button:hover { background: #0056b3; }
                #status { margin-top: 15px; color: #666; font-style: italic; font-size: 14px; min-height: 20px; }
                #answer { margin-top: 15px; white-space: pre-wrap; font-size: 16px; line-height: 1.5; }
                .source-block { background: #f8f9fa; padding: 10px; margin-top: 10px; border-left: 4px solid #007bff; font-size: 14px; }
            </style>
        </head>
        <body>
            <div class="chat-box">
                <h2> ИИ-Агент по отзывам</h2>
                <p style="color: #666; font-size: 14px;">Спросите про побочки лекарства, или попросите проанализировать тональность текста.</p>
                <input type="text" id="questionInput" placeholder="Например: Какие побочки у Ингавирина?">
                <button onclick="askAgent()">Спросить</button>
                
                <!-- Индикатор работы агента -->
                <div id="status"></div>
                <div id="answer"></div>
            </div>

            <script>
                async function askAgent() {
                    const q = document.getElementById('questionInput').value;
                    const statusDiv = document.getElementById('status');
                    const answerDiv = document.getElementById('answer');
                    if (!q) return;
                    
                    // Показываем, что агент "думает"
                    statusDiv.innerHTML = "⏳ <b>Агент анализирует запрос и выбирает инструменты...</b>";
                    answerDiv.innerHTML = "";
                    
                    try {
                        const response = await fetch('/agent_ask', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ question: q })
                        });
                        
                        const data = await response.json();
                        
                        statusDiv.innerHTML = "✅ <b>Агент принял решение.</b>";
                        answerDiv.innerHTML = "<b>Ответ:</b><br>" + data.answer;
                        
                        if (data.sources && data.sources.length > 0) {
                            let sourcesHtml = data.sources.map(s => 
                                `<div class="source-block"><b>📄 ${s.filename}</b><br>${s.preview}</div>`
                            ).join("");
                            answerDiv.innerHTML += "<br><br><b>Использованные источники:</b>" + sourcesHtml;
                        }
                    } catch (e) {
                        statusDiv.innerHTML = "❌ Ошибка при запросе.";
                        answerDiv.innerHTML = "Не удалось получить ответ от сервера.";
                    }
                }
            </script>
        </body>
    </html>
    """