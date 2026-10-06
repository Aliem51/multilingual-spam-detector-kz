# ==========================================
# СКРИПТ ДЛЯ ЛОКАЛЬНОГО ЗАПУСКА (VS CODE)
# ==========================================
import os
import pandas as pd
import re
import joblib
import nltk
import gc
import time
from tqdm import tqdm  # Прогресс-бар
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Включаем tqdm для pandas
tqdm.pandas()

# Проверяем и качаем стоп-слова
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ==========================================
# 1. ЗАГРУЗКА
# ==========================================
# Убедись, что файл лежит РЯДОМ со скриптом или укажи полный путь, например: "C:\\Users\\...\\final_bilingual_dataset.csv"
filename = 'dataset_with_kz_context.csv'
 
print(f"📂 Загружаю {filename}...")
start_time = time.time()

# engine='c' быстрее, но 'python' надежнее. Попробуем 'c' для скорости, если упадет - меняй на 'python'
try:
    df = pd.read_csv(filename, on_bad_lines='skip', engine='c') 
except:
    df = pd.read_csv(filename, on_bad_lines='skip', engine='python')

print(f"⏱ Время загрузки: {time.time() - start_time:.2f} сек.")

# Стандартная чистка колонок
if 'message' in df.columns:
    df.rename(columns={'message': 'text'}, inplace=True)
df.dropna(subset=['text', 'label'], inplace=True)
df['label'] = pd.to_numeric(df['label'], errors='coerce')
df.dropna(subset=['label'], inplace=True)
df['label'] = df['label'].astype(int)

# ==========================================
# 🛑 НАСТРОЙКА ОБЪЕМА 🛑
# ==========================================
# Для 16 ГБ ОЗУ - 1 миллион это безопасный максимум.
# Хочешь рискнуть? Поставь 1.5 млн (1500000).
sample_size = 1694322  

if len(df) > sample_size:
    print(f"🔪 Обрезаю датасет до {sample_size} строк...")
    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

print(f"✅ Готовы к работе. Строк: {len(df)}")

# ==========================================
# 2. ОЧИСТКА
# ==========================================
stop_words_ru = set(stopwords.words('russian'))
stop_words_en = set(stopwords.words('english'))
combined_stop_words = stop_words_ru.union(stop_words_en)

def clean_text_bilingual(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    # Фильтрацию можно ускорить, не создавая новый список каждый раз, но оставим так для надежности
    return " ".join([word for word in words if word not in combined_stop_words])

print("⏳ Чищу текст (смотри на прогресс-бар)...")
# progress_apply покажет полоску загрузки
df['clean_text'] = df['text'].progress_apply(clean_text_bilingual)

# Чистим память
del df['text']
gc.collect()
print("✅ ОЗУ освобождено.")

# ==========================================
# 3. ВЕКТОРИЗАЦИЯ И ОБУЧЕНИЕ
# ==========================================
print("⏳ Векторизация (самый тяжелый этап)...")
# max_features=15000 - оптимально для 16 ГБ ОЗУ
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

print("🚀 Обучаю модель...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1) # n_jobs=-1 задействует ВСЕ ядра Ryzen
model.fit(X_train, y_train)

# ==========================================
# 4. РЕЗУЛЬТАТЫ
# ==========================================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Точность: {acc:.4f}")
print(classification_report(y_test, y_pred))

# ==========================================
# 5. СОХРАНЕНИЕ
# ==========================================
print("💾 Сохраняю файлы...")

# Получаем папку, где лежит ИМЕННО ЭТОТ скрипт
current_dir = os.path.dirname(os.path.abspath(__file__))

# Создаем полные пути к файлам
model_path = os.path.join(current_dir, 'spam_model.pkl')
vec_path = os.path.join(current_dir, 'vectorizer.pkl')

joblib.dump(model, model_path)
joblib.dump(vectorizer, vec_path)

print(f"🎉 Готово! Файлы сохранены точно сюда:\n{model_path}\n{vec_path}")