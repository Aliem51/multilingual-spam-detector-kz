
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostClassifier
import pandas as pd

# Инициализируем API
app = FastAPI(
    title="Spam Detector Pro API",
    description="API для проверки сообщений на спам с помощью модели CatBoost",
    version="1.0.0"
)

# Загружаем модель при старте сервера
try:
    model = CatBoostClassifier()
    model.load_model('catboost_spam_model.cbm')
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model = None

# Структура входящего запроса (JSON)
class MessageRequest(BaseModel):
    text: str
    threshold: float = 0.5 # Порог по умолчанию (от 0.0 до 1.0)

# Ендпоинт для проверки текста
@app.post("/predict")
async def check_spam(req: MessageRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена на сервере.")
    
    if len(req.text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Текст слишком короткий.")

    # Упаковываем текст для CatBoost
    df = pd.DataFrame({'text': [req.text]})
    
    # Получаем вероятность
    proba = model.predict_proba(df)[0]
    spam_prob = float(proba[1])
    
    # Выносим вердикт на основе порога
    is_spam = bool(spam_prob >= req.threshold)
    
    # Возвращаем ответ в формате JSON
    return {
        "is_spam": is_spam,
        "spam_probability": round(spam_prob * 100, 2),
        "verdict": "SPAM DETECTED" if is_spam else "SAFE",
        "threshold_used": req.threshold
    }