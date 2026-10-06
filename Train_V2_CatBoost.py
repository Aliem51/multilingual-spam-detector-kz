# ==========================================
# СКРИПТ С ИСПОЛЬЗОВАНИЕМ CATBOOST (SOTA для CPU)
# ==========================================
import os
import pandas as pd
import gc
import time
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

filename = 'perfect_kz_dataset.csv'
print(f"📂 Загружаю {filename}...")
start_time = time.time()

# Загрузка
try:
    df = pd.read_csv(filename, on_bad_lines='skip', engine='c') 
except:
    df = pd.read_csv(filename, on_bad_lines='skip', engine='python')

print(f"⏱ Время загрузки: {time.time() - start_time:.2f} сек.")

# Подготовка колонок
if 'message' in df.columns:
    df.rename(columns={'message': 'text'}, inplace=True)
df.dropna(subset=['text', 'label'], inplace=True)
df['label'] = pd.to_numeric(df['label'], errors='coerce')
df.dropna(subset=['label'], inplace=True)
df['label'] = df['label'].astype(int)

# Приводим все к строке (защита от ошибок)
df['text'] = df['text'].astype(str)

sample_size = 1600000 
if len(df) > sample_size:
    print(f"🔪 Обрезаю датасет до {sample_size} строк...")
    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

print(f"✅ Готовы к работе. Строк: {len(df)}")

# ==========================================
# ВАЖНО: Мы больше НЕ удаляем ссылки и цифры!
# Оставляем текст как есть (сырым), CatBoost сам разберется.
# ==========================================

X = df[['text']] # Передаем DataFrame, а не Series
y = df['label']

print("🚀 Разделение данных...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

del df
gc.collect()

# Создаем пулы данных (формат, который любит CatBoost)
# text_features=['text'] говорит модели, что эта колонка содержит текст
train_pool = Pool(data=X_train, label=y_train, text_features=['text'])
test_pool = Pool(data=X_test, label=y_test, text_features=['text'])

print("🧠 Обучаю CatBoost (встроенный TF-IDF + Dictionary)...")
# Настройки для Ryzen (CPU)
model = CatBoostClassifier(
    iterations=800,              # Количество деревьев
    learning_rate=0.1,           # Шаг обучения
    depth=6,                     # Глубина дерева
    eval_metric='F1',            # Смотрим на качество (F1 лучше для спама, чем Accuracy)
    auto_class_weights='Balanced', # Решает проблему перекоса классов
    task_type='CPU',
    thread_count=-1,             # Использовать все ядра процессора
    verbose=100,                 # Выводить прогресс каждые 100 шагов
    
    # ИСПРАВЛЕННЫЙ БЛОК: Безопасный синтаксис для биграмм
    text_processing=[
        "NaiveBayes+Word,BiGram|BoW+Word,BiGram"
    ]
)

model.fit(train_pool, eval_set=test_pool)

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
print("💾 Сохраняю модель...")
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'catboost_spam_model.cbm')

# CatBoost сохраняет все в ОДИН файл (и векторизатор, и саму модель)
model.save_model(model_path)

print(f"🎉 Готово! Модель сохранена сюда:\n{model_path}")