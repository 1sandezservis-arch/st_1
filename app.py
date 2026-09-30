import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# 1. Налаштування сторінки
st.set_page_config(
    page_title="ШІ Примірка Шпалер",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Стилізація інтерфейсу для мобільних пристроїв
st.markdown("""
    <style>
    .main { padding: 1rem; }
    .stButton>button { 
        width: 100%; 
        border-radius: 12px; 
        height: 3.2em; 
        background: linear-gradient(90deg, #4F46E5, #06B6D4); 
        color: white; 
        font-weight: bold; 
        font-size: 16px;
        border: none;
    }
    .stSelectbox, .stSlider { font-weight: 500; }
    </style>
""", unsafe_allow_html=True)

st.title("🖼️ ШІ-Примірка Шпалер")
st.caption("Автоматичне розпізнавання стін, обтікання меблів та збереження тіней")

# --- Алгоритми обробки (OpenCV) ---

def generate_wallpaper_pattern(style, width, height, scale=1.0):
    """Створення текстури шпалер"""
    size = int(120 * scale)
    tile = np.zeros((size, size, 3), dtype=np.uint8)

    if style == "Смарагд + Золото":
        tile[:] = (43, 61, 16)  # BGR
        cv2.circle(tile, (size//2, size//2), int(size*0.35), (89, 175, 212), 3)
        cv2.circle(tile, (0, 0), int(size*0.2), (89, 175, 212), 2)
    elif style == "Лофт Цегла":
        tile[:] = (46, 59, 122)
        cv2.rectangle(tile, (4, 4), (size-4, size//2-4), (27, 35, 74), -1)
        cv2.rectangle(tile, (4, size//2+4), (size-4, size-4), (27, 35, 74), -1)
    elif style == "Текстурний Льон":
        tile[:] = (210, 215, 218)
        for i in range(0, size, 6):
            cv2.line(tile, (i, 0), (i, size), (185, 190, 195), 1)
            cv2.line(tile, (0, i), (size, i), (185, 190, 195), 1)
    else:  # Тропічний бамбук
        tile[:] = (50, 90, 40)
        for i in range(10, size, 30):
            cv2.line(tile, (i, 0), (i+15, size), (80, 140, 60), 4)

    rx = int(np.ceil(width / size)) + 1
    ry = int(np.ceil(height / size)) + 1
    pattern = np.tile(tile, (ry, rx, 1))
    return pattern[:height, :width]


def apply_ai_wallpaper(room_bgr, texture_bgr, sensitivity, opacity):
    """ШІ-обтікання об'єктів та збереження природних тіней"""
    h, w, _ = room_bgr.shape
    texture_resized = cv2.resize(texture_bgr, (w, h))

    # 1. Детекція контурів меблів, вазонів, вікон (Canny)
    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, sensitivity, sensitivity * 2)
    
    # Розширення меж для безпечної зони навколо дрібного листя/ніжок
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # 2. Створення маски стіни
    wall_mask = np.ones((h, w), dtype=np.uint8) * 255
    wall_mask[dilated_edges > 0] = 0

    # 3. Витягнення освітлення та тіней з оригіналу (L-канал LAB)
    lab = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    shadow_map = l_channel.astype(float) / 255.0
    shadow_map = np.clip(shadow_map * 1.15, 0, 1)

    # 4. Накладання тіней на нову текстуру (Multiply Blend)
    textured_wall = (texture_resized.astype(float) * shadow_map[:, :, np.newaxis]).astype(np.uint8)

    # 5. М'яке змішування маски
    smooth_mask = cv2.GaussianBlur(wall_mask, (7, 7), 0)[:, :, np.newaxis] / 255.0
    alpha = opacity * smooth_mask

    output_bgr = (room_bgr * (1 - alpha) + textured_wall * alpha).astype(np.uint8)
    return output_bgr


# --- КРОК 1: Отримання фото ---
st.subheader("Крок 1: Оберіть або зробіть фото")
source_mode = st.radio("Джерело фото:", ["📷 Зробити знімок з камери", "📁 Завантажити з галереї"], horizontal=True)

img_file = None
if "Зробити знімок" in source_mode:
    img_file = st.camera_input("Наведіть камеру на стіну")
else:
    img_file = st.file_uploader("Оберіть файл зображення", type=["jpg", "png", "jpeg"])


# --- КРОК 2 та 3: Обробка при наявності фото ---
if img_file is not None:
    room_pil = Image.open(img_file)
    room_bgr = cv2.cvtColor(np.array(room_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
    h, w, _ = room_bgr.shape

    st.subheader("Крок 2: Оберіть шпалери")
    wp_choice = st.selectbox(
        "Каталог шпалер:",
        ["Смарагд + Золото", "Лофт Цегла", "Текстурний Льон", "Тропічний бамбук", "Завантажити власний візерунок"]
    )

    custom_texture_bgr = None
    if wp_choice == "Завантажити власний візерунок":
        custom_file = st.file_uploader("Завантажте файл вашої текстури", type=["jpg", "png"])
        if custom_file:
            c_pil = Image.open(custom_file)
            custom_texture_bgr = cv2.cvtColor(np.array(c_pil.convert('RGB')), cv2.COLOR_RGB2BGR)

    # Налаштування у розгортці (щоб не захаращувати екран)
    with st.expander("⚙️ Точна настройка (Обтікання та Тіні)", expanded=False):
        sens = st.slider("Точність обтікання меблів", 10, 80, 35, help="Зменшіть, якщо шпалери заходять на меблі")
        opac = st.slider("Яскравість / Насиченість", 0.2, 1.0, 0.8)
        scale = st.slider("Масштаб малюнка", 0.5, 2.5, 1.0)

    # Підготовка текстури
    if custom_texture_bgr is not None:
        texture_bgr = custom_texture_bgr
    else:
        texture_bgr = generate_wallpaper_pattern(wp_choice, w, h, scale)

    # Генерація результату
    with st.spinner("AI накладає шпалери та огинає об'єкти..."):
        res_bgr = apply_ai_wallpaper(room_bgr, texture_bgr, sens, opac)
        res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)

    # КРОК 3: Відображення результату
    st.subheader("✨ Готовий результат:")
    st.image(res_rgb, use_container_width=True)

    # Кнопка завантаження
    result_pil = Image.fromarray(res_rgb)
    buf = io.BytesIO()
    result_pil.save(buf, format="JPEG", quality=95)
    
    st.download_button(
        label="💾 Зберегти готове фото",
        data=buf.getvalue(),
        file_name="wallpaper_result.jpg",
        mime="image/jpeg"
    )
