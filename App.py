import streamlit as st
import pandas as pd
import time
from catboost import CatBoostClassifier
import io

# 1. НАСТРОЙКА СТРАНИЦЫ

st.set_page_config(
    page_title="Onir",
    page_icon="🛡️",
    layout="centered"
)

# CSS 
st.markdown("""
    <style>
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: #FF4B4B;
        text-align: center;
        font-size: 3rem;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF914D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 50px;
        font-weight: bold;
        font-size: 18px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .api-code {
        background-color: #272822;
        color: #f8f8f2;
        padding: 15px;
        border-radius: 8px;
        font-family: monospace;
    }
    </style>
""", unsafe_allow_html=True)

# 2. ПЕРЕВОД

translations = {
    "RU": {
        "settings": "⚙️ Настройки Core Engine",
        "model_choice": "Выбор алгоритма классификации:",
        "model_catboost": "1. CatBoost (Текущий, SOTA)",
        "model_logreg": "2. Logistic Regression (Базовый)",
        "model_warn": "⚠️ Logistic Regression находится в стадии интеграции. Временно используется движок CatBoost.",
        "threshold_title": "**Управление чувствительностью**",
        "threshold": "Порог уверенности для Спама (%)",
        "threshold_help": "Если вероятность спама выше этого значения, сообщение будет помечено как СПАМ. Повысьте порог, чтобы избежать ложных срабатываний.",
        "diploma": f"Дипломный проект {time.strftime('%Y')}\nМультиязычный анализ: RU, EN, KZ",
        "title": "🛡️ Spam Detector",
        "tab1": "💬 Одиночная",
        "tab2": "📁 Пакетная (CSV)",
        "tab3": "📊 Метрики",
        "tab4": "💻 API Интеграция",
        "t1_desc": "Введите текст сообщения (SMS или Email) для проверки на спам.",
        "t1_label": "Текст",
        "t1_placeholder": "Вставьте текст сюда...",
        "t1_btn": "🚀 ЗАПУСТИТЬ АНАЛИЗ",
        "t1_err_model": "❌ Ошибка: Не найден файл модели.",
        "t1_warn_short": "⚠️ Текст слишком короткий.",
        "t1_spam": "**ОБНАРУЖЕН СПАМ / SPAM DETECTED**",
        "t1_safe": "**БЕЗОПАСНОЕ СООБЩЕНИЕ**",
        "t1_prob_spam": "Вероятность спама",
        "t1_prob_safe": "Уверенность в безопасности",
        "t1_thresh_info": "Порог:",
        "t1_thresh_below": "Ниже порога",
        "t2_title": "Пакетная проверка массива сообщений",
        "t2_info": "Загрузите CSV или TXT файл. Убедитесь, что первая строка содержит заголовок **text** или **message**.",
        "t2_upload": "Выберите CSV файл",
        "t2_loaded": "✅ Файл загружен. Найдено сообщений:",
        "t2_err_col": "❌ В файле не найдена колонка с текстом. Убедитесь, что самое первое слово в файле — это 'text'.",
        "t2_btn": "⚙️ Начать пакетную обработку",
        "t2_success": "🎉 Обработка завершена!",
        "t2_download": "📥 Скачать результаты (.csv)",
        "t2_err_unexp": "Произошла непредвиденная ошибка:",
        "t3_title": "Показатели качества модели (CatBoost SOTA)",
        "t3_desc": "Результаты тестирования на валидационной выборке (60 000 сообщений: RU, EN, KZ).",
        "t3_acc": "Точность (Accuracy)",
        "t3_acc_desc": "Общий процент правильных ответов.",
        "t3_prec": "Precision (Спам)",
        "t3_prec_desc": "Доля реального спама среди того, что модель назвала спамом.",
        "t3_rec": "Полнота (Recall)",
        "t3_rec_desc": "Какую часть легитимных писем модель правильно пропустила.",
        "t3_arch": "**Детали архитектуры:**",
        "t4_title": "Интеграция по API (Microservice Architecture)"
    },
    "EN": {
        "settings": "⚙️ Core Engine Settings",
        "model_choice": "Classification Algorithm:",
        "model_catboost": "1. CatBoost (Current, SOTA)",
        "model_logreg": "2. Logistic Regression (Baseline)",
        "model_warn": "⚠️ Logistic Regression is currently being integrated. Defaulting to CatBoost.",
        "threshold_title": "**Sensitivity Control**",
        "threshold": "Spam Confidence Threshold (%)",
        "threshold_help": "If spam probability exceeds this value, the message is marked as SPAM. Increase threshold to avoid false positives.",
        "diploma": f"Diploma Project {time.strftime('%Y')}\nMultilingual analysis: RU, EN, KZ",
        "title": "🛡️ Spam Detector",
        "tab1": "💬 Single Check",
        "tab2": "📁 Batch (CSV)",
        "tab3": "📊 Metrics",
        "tab4": "💻 API Integration",
        "t1_desc": "Enter message text (SMS or Email) to check for spam.",
        "t1_label": "Text",
        "t1_placeholder": "Paste text here...",
        "t1_btn": "🚀 RUN ANALYSIS",
        "t1_err_model": "❌ Error: Model file not found.",
        "t1_warn_short": "⚠️ Text is too short.",
        "t1_spam": "**SPAM DETECTED**",
        "t1_safe": "**SAFE MESSAGE**",
        "t1_prob_spam": "Spam Probability",
        "t1_prob_safe": "Safe Confidence",
        "t1_thresh_info": "Threshold:",
        "t1_thresh_below": "Below threshold",
        "t2_title": "Batch Message Checking",
        "t2_info": "Upload a CSV or TXT file. Ensure the first row contains a **text** or **message** header.",
        "t2_upload": "Choose CSV file",
        "t2_loaded": "✅ File loaded. Messages found:",
        "t2_err_col": "❌ Text column not found. Ensure the very first word in the file is 'text'.",
        "t2_btn": "⚙️ Start Batch Processing",
        "t2_success": "🎉 Processing complete!",
        "t2_download": "📥 Download results (.csv)",
        "t2_err_unexp": "Unexpected error occurred:",
        "t3_title": "Model Quality Metrics (CatBoost SOTA)",
        "t3_desc": "Validation set testing results (60,000 messages: RU, EN, KZ).",
        "t3_acc": "Accuracy",
        "t3_acc_desc": "Overall percentage of correct predictions.",
        "t3_prec": "Precision (Spam)",
        "t3_prec_desc": "Proportion of real spam among what the model labeled as spam.",
        "t3_rec": "Recall",
        "t3_rec_desc": "Proportion of legitimate emails the model correctly passed.",
        "t3_arch": "**Architecture Details:**",
        "t4_title": "API Integration (Microservice Architecture)"
    },
    "KZ": {
        "settings": "⚙️ Баптаулар (Core Engine)",
        "model_choice": "Классификация алгоритмін таңдау:",
        "model_catboost": "1. CatBoost (Ағымдағы, SOTA)",
        "model_logreg": "2. Logistic Regression (Базалық)",
        "model_warn": "⚠️ Logistic Regression интеграциялануда. Уақытша CatBoost қолданылады.",
        "threshold_title": "**Сезімталдықты басқару**",
        "threshold": "Спамды анықтау шегі (%)",
        "threshold_help": "Егер спам ықтималдығы осы мәннен жоғары болса, хабарлама СПАМ ретінде белгіленеді.",
        "diploma": f"Дипломдық жоба {time.strftime('%Y')}\nКөптілді талдау: RU, EN, KZ",
        "title": "🛡️ Spam Detector",
        "tab1": "💬 Жеке тексеру",
        "tab2": "📁 Топтама (CSV)",
        "tab3": "📊 Метрикалар",
        "tab4": "💻 API Интеграция",
        "t1_desc": "Спамға тексеру үшін хабарлама мәтінін (SMS немесе Email) енгізіңіз.",
        "t1_label": "Мәтін",
        "t1_placeholder": "Мәтінді осында қойыңыз...",
        "t1_btn": "🚀 ТАЛДАУДЫ БАСТАУ",
        "t1_err_model": "❌ Қате: Модель файлы табылмады.",
        "t1_warn_short": "⚠️ Мәтін тым қысқа.",
        "t1_spam": "**СПАМ АНЫҚТАЛДЫ / SPAM DETECTED**",
        "t1_safe": "**ҚАУІПСІЗ ХАБАРЛАМА**",
        "t1_prob_spam": "Спам ықтималдығы",
        "t1_prob_safe": "Қауіпсіздікке сенімділік",
        "t1_thresh_info": "Шек:",
        "t1_thresh_below": "Шектен төмен",
        "t2_title": "Хабарламалар массивін топтамалық тексеру",
        "t2_info": "CSV немесе TXT файлын жүктеңіз. Бірінші жолда **text** немесе **message** тақырыбы болуы керек.",
        "t2_upload": "CSV файлын таңдаңыз",
        "t2_loaded": "✅ Файл жүктелді. Табылған хабарламалар:",
        "t2_err_col": "❌ Мәтін бағаны табылмады. Файлдағы ең бірінші сөз 'text' екеніне көз жеткізіңіз.",
        "t2_btn": "⚙️ Топтамалық өңдеуді бастау",
        "t2_success": "🎉 Өңдеу аяқталды!",
        "t2_download": "📥 Нәтижелерді жүктеп алу (.csv)",
        "t2_err_unexp": "Күтпеген қате орын алды:",
        "t3_title": "Модель сапасының көрсеткіштері (CatBoost SOTA)",
        "t3_desc": "Валидациялық іріктеудегі тестілеу нәтижелері (60 000 хабарлама: RU, EN, KZ).",
        "t3_acc": "Дәлдік (Accuracy)",
        "t3_acc_desc": "Дұрыс жауаптардың жалпы пайызы.",
        "t3_prec": "Precision (Спам)",
        "t3_prec_desc": "Модель спам деп тапқандардың ішіндегі нақты спамның үлесі.",
        "t3_rec": "Толықтық (Recall)",
        "t3_rec_desc": "Модель дұрыс өткізген заңды хаттардың үлесі.",
        "t3_arch": "**Архитектура мәліметтері:**",
        "t4_title": "API арқылы интеграция (Microservice Architecture)"
    }
}

# 3. ЗАГРУЗКА МОДЕЛИ

@st.cache_resource
def load_model():
    try:
        model = CatBoostClassifier()
        model.load_model('catboost_spam_model.cbm')
        return model
    except Exception as e:
        return None

model = load_model()


# 4. БОКОВАЯ ПАНЕЛЬ (НАСТРОЙКИ & ЯЗЫК)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=100)
    
    # ПЕРЕКЛЮЧАТЕЛЬ ЯЗЫКА
    lang = st.selectbox("🌐 Язык / Language / Тіл:", ["RU", "EN", "KZ"])
    t = translations[lang] # Загружаем нужный словарь в переменную 't'
    
    st.title(t["settings"])
    
    model_choice = st.selectbox(
        t["model_choice"],
        [t["model_catboost"], t["model_logreg"]]
    )
    if "Logistic" in model_choice or "Базалық" in model_choice or "Baseline" in model_choice:
        st.warning(t["model_warn"])
    
    st.markdown("---")
    
    st.write(t["threshold_title"])
    threshold_pct = st.slider(
        t["threshold"], 
        min_value=50, 
        max_value=90, 
        value=50, 
        step=1,
        help=t["threshold_help"]
    )
    threshold = threshold_pct / 100.0

    st.markdown("---")
    st.caption(t["diploma"])


# 5. ОСНОВНОЙ ЭКРАН (ВКЛАДКИ)

st.markdown(f'<h1 class="main-header">{t["title"]}</h1>', unsafe_allow_html=True)

# Вкладки с динамическими названиями
tab1, tab2, tab3, tab4 = st.tabs([t["tab1"], t["tab2"], t["tab3"], t["tab4"]])


# ВКЛАДКА 1: ОДИНОЧНАЯ ПРОВЕРКА

with tab1:
    st.write(t["t1_desc"])

    if 'u_input' not in st.session_state: st.session_state['u_input'] = ""
    user_text = st.text_area(
        label=t["t1_label"], 
        value=st.session_state['u_input'], 
        height=150, 
        placeholder=t["t1_placeholder"],
        label_visibility="collapsed"
    )

    if st.button(t["t1_btn"], type="primary", key="btn_single"):
        if model is None:
            st.error(t["t1_err_model"])
        elif user_text.strip():
            with st.spinner('...'):
                time.sleep(0.5)
                
                raw_text = str(user_text)
                if len(raw_text.strip()) < 3:
                     st.warning(t["t1_warn_short"])
                else:
                    df_single = pd.DataFrame({'text': [raw_text]})
                    
                    proba_result = model.predict_proba(df_single)
                    proba = proba_result[0]
                    
                    spam_prob = float(proba[1])
                    ham_prob = float(proba[0])
                    
                    is_spam = True if spam_prob >= threshold else False

                    st.markdown("---")
                    c1, c2 = st.columns([1, 2])
                    
                    with c1:
                        if is_spam:
                            st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=120)
                        else:
                            st.image("https://cdn-icons-png.flaticon.com/512/148/148767.png", width=120)
                    
                    with c2:
                        if is_spam:
                            st.error(f"{t['t1_spam']} ({t['t1_thresh_info']} {threshold_pct}%)")
                            st.metric(t["t1_prob_spam"], f"{spam_prob*100:.1f}%")
                            st.progress(spam_prob)
                        else:
                            st.success(f"{t['t1_safe']} ({t['t1_thresh_below']} {threshold_pct}%)")
                            st.metric(t["t1_prob_safe"], f"{ham_prob*100:.1f}%")
                            st.progress(ham_prob)


# ВКЛАДКА 2: ПАКЕТНАЯ ОБРАБОТКА (CSV)

with tab2:
    st.subheader(t["t2_title"])
    st.info(t["t2_info"])
    
    uploaded_file = st.file_uploader(t["t2_upload"], type=['csv', 'txt'])
    
    if uploaded_file is not None:
        try:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
            except pd.errors.ParserError:
                uploaded_file.seek(0)
                content = uploaded_file.getvalue().decode('utf-8').splitlines()
                lines = [line.strip() for line in content if line.strip()]
                
                if len(lines) > 1:
                    header = lines[0]
                    data = lines[1:]
                    df = pd.DataFrame({header: data})
                else:
                    df = pd.DataFrame()

            st.write(f"{t['t2_loaded']} **{len(df)}**")
            
            text_col = None
            for col in ['text', 'message', 'Текст', 'Message', 'email_text']:
                if col in df.columns:
                    text_col = col
                    break
            
            if not text_col:
                st.error(t["t2_err_col"])
            else:
                if st.button(t["t2_btn"], type="primary"):
                    with st.spinner('...'):
                        
                        df_predict = pd.DataFrame({'text': df[text_col].astype(str)})
                        probas = model.predict_proba(df_predict)
                        
                        spam_probs = probas[:, 1] 
                        verdicts = ["Спам" if p >= threshold else "Не спам" for p in spam_probs]
                        
                        df['Вероятность спама (%)'] = (spam_probs * 100).round(2)
                        df['Вердикт'] = verdicts
                        
                        st.success(t["t2_success"])
                        st.dataframe(df.head(50))
                        
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=t["t2_download"],
                            data=csv,
                            file_name="spam_detection_results.csv",
                            mime="text/csv",
                        )
        except Exception as e:
            st.error(f"{t['t2_err_unexp']} {e}")


# ВКЛАДКА 3: ДАШБОРД МЕТРИК

with tab3:
    st.subheader(t["t3_title"])
    st.write(t["t3_desc"])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #4CAF50;">98.25%</h2>
            <p style="font-weight: bold; color: gray;">{t["t3_acc"]}</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption(t["t3_acc_desc"])

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #2196F3;">98.00%</h2>
            <p style="font-weight: bold; color: gray;">{t["t3_prec"]}</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption(t["t3_prec_desc"])

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #FF9800;">99.00%</h2>
            <p style="font-weight: bold; color: gray;">{t["t3_rec"]}</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption(t["t3_rec_desc"])
        
    st.markdown("---")
    st.write(t["t3_arch"])
    st.write("✔️ **Движок:** CatBoost (Gradient Boosting on Decision Trees)")
    st.write("✔️ **Лингвистика:** Встроенная токенизация и анализ n-грамм (BiGrams)")
    st.write("✔️ **Локализация:** Специальная защита от ложных срабатываний на брендах (Kaspi, Egov, Яндекс, Netflix)")

# ВКЛАДКА 4: API ДОКУМЕНТАЦИЯ
with tab4:
    st.subheader(t["t4_title"])
    
    st.markdown("### 📡 Эндпоинт: `POST /predict`")
    st.write("**Request JSON:**")
    st.code('''{
  "text": "Внимание! Ваш аккаунт заблокирован. Перейдите по ссылке: http://scam.com",
  "threshold": 0.7
}''', language='json')

    st.write("**Response JSON:**")
    st.code('''{
  "is_spam": true,
  "spam_probability": 99.85,
  "verdict": "SPAM DETECTED",
  "threshold_used": 0.7
}''', language='json')

    st.markdown("### 💻 Интеграция (Python):")
    st.code('''import requests

url = "http://localhost:8000/predict"
data = {
    "text": "Сәлем, қалайсың? Ертең жұмысқа барасың ба?",
    "threshold": 0.8
}

response = requests.post(url, json=data)
print(response.json())
''', language='python')