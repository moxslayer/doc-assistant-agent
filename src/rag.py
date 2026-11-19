import os
import faiss
import pandas as pd
import numpy as np
import re
import time
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import ollama

# Модель для эмбеддингов (поиска)
EMB_MODEL_NAME = "intfloat/multilingual-e5-small" 
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Клиент Ollama
ollama_client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://ollama:11434"))

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    if len(text) <= size: return [text]
    step = size - overlap
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text): break
        start += step
    return chunks

def tokenize_ru(text):
    text = text.lower()
    text = re.sub(r'[^a-zа-яё0-9\s]', '', text) 
    return text.split()

def build_index():
    print("Загрузка данных...")
    df = pd.read_csv("data/dataset_clean.csv", encoding="utf-8-sig")
    df["text"] = df["text"].astype(str).str.strip()
    
    full_texts = dict(zip(df["filename"], df["text"]))
    
    all_chunks = []
    for _, row in df.iterrows():
        for i, ch in enumerate(chunk_text(row["text"])):
            all_chunks.append({"filename": row["filename"], "chunk_id": i, "text": ch})
            
    texts_for_emb = ["passage: " + c["text"] for c in all_chunks]
    
    print("Построение эмбеддингов (E5)...")
    model = SentenceTransformer(EMB_MODEL_NAME)
    emb = model.encode(
        texts_for_emb, batch_size=32, show_progress_bar=False, 
        convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    
    print("Построение BM25 индекса...")
    tokenized_corpus = [tokenize_ru(doc) for doc in [c["text"] for c in all_chunks]]
    bm25 = BM25Okapi(tokenized_corpus)
    
    print("Индекс готов!")
    return {"index": index, "chunks": all_chunks, "model": model, "bm25": bm25, "full_texts": full_texts}

def rag_answer(question: str, index_state: dict, k: int = 3):
    model = index_state["model"]
    index = index_state["index"]
    all_chunks = index_state["chunks"]
    bm25 = index_state["bm25"]
    full_texts = index_state["full_texts"]
    
    # 1. Поиск кандидатов
    qv = model.encode(["query: " + question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    dist, idx = index.search(qv, k * 5)
    vector_indices = [int(i) for i in idx[0] if i != -1]
    
    tokenized_query = tokenize_ru(question)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:k * 5].tolist()
    
    # 2. Объединение (RRF)
    rrf_scores = {}
    rrf_constant = 60
    for rank, doc_idx in enumerate(vector_indices):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (rrf_constant + rank + 1)
    for rank, doc_idx in enumerate(bm25_indices):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (rrf_constant + rank + 1)
        
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 3. УМНЫЙ ФИЛЬТР
    STOP_WORDS = {
        'какие', 'какой', 'какова', 'каковы', 'побочки', 'побочек', 'побочных', 'побочные', 
        'эффекты', 'эффектов', 'эффект', 'действие', 'действия', 'применение', 'применения', 
        'лекарство', 'лекарства', 'препарат', 'препарата', 'средство', 'средства', 'можно', 
        'нужно', 'есть', 'были', 'было', 'отзывы', 'отзыв', 'помогает', 'помогло', 'пить', 
        'принимать', 'принимал', 'принимала', 'врач', 'врача', 'мнение', 'как', 'чем', 'что',
        'если', 'это', 'также', 'очень', 'было', 'были', 'быть', 'него', 'только', 'есть'
    }
    
    keywords = [w for w in tokenized_query if len(w) > 4 and w not in STOP_WORDS]
    
    final_filenames = []
    
    if keywords:
        for doc_idx, score in sorted_docs:
            chunk_text_lower = all_chunks[doc_idx]["text"].lower()
            if any(kw in chunk_text_lower for kw in keywords):
                final_filenames.append(all_chunks[doc_idx]["filename"])
            if len(final_filenames) >= k:
                break
                
        if not final_filenames:
            return "По вашему запросу ничего не найдено. В базе нет отзывов об этом препарате.", []
    else:
        final_filenames = [all_chunks[doc_idx]["filename"] for doc_idx, _ in sorted_docs[:k]]

    unique_filenames = list(dict.fromkeys(final_filenames))
    
    context = ""
    for fname in unique_filenames:
        context += f"[Источник: {fname}]\n{full_texts[fname]}\n\n"
        
    prompt = f"""Ниже приведены полные тексты отзывов. Ответь на вопрос, используя ТОЛЬКО эту информацию.
Если информации нет, напиши ровно одну фразу: «В предоставленных отзывах нет информации об этом».

Фрагменты:
{context}

Вопрос: {question}
Ответ:"""
    
    # 4. Вызов Ollama (Используем qwen2.5:1.5b!)
    answer = ""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = ollama_client.chat(
                model='qwen2.5:1.5b',  # <--- ЗАМЕНЕНО НА QWEN
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.1}
            )
            answer = response['message']['content'].strip()
            break 
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Ошибка подключения к Ollama: {e}. Ждем 2 секунды...")
                time.sleep(2)
            else:
                answer = f"Ошибка LLM после {max_retries} попыток: {e}"
        
    sources = [{"filename": fname, "preview": full_texts[fname]} for fname in unique_filenames]
    return answer, sources