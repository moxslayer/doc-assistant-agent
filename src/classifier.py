from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Инициализируем и "обучаем" на ходу (для демо)
# В реальности модель обучают на отдельном датасете и сохраняют через pickle
classifier_model = LogisticRegression()
vectorizer = TfidfVectorizer()

def init_classifier():
    texts = [
        "хорошее средство помогло быстро вылечился рекомендую", 
        "плохо стало тошнить побочка ужас не покупайте",
        "отличный препарат эффективно снимает боль",
        "выпил таблетку и стало еще хуже, отравился"
    ]
    labels = [1, 0, 1, 0] # 1 = Positive, 0 = Negative
    X = vectorizer.fit_transform(texts)
    classifier_model.fit(X, labels)

def classify_text(text: str):
    X = vectorizer.transform([text])
    pred = classifier_model.predict(X)[0]
    proba = np.max(classifier_model.predict_proba(X))
    return {"label": "Positive" if pred == 1 else "Negative", "confidence": round(float(proba), 2)}