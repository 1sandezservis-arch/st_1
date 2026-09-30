import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Налаштування мобільного інтерфейсу
st.set_page_config(
    page_title="Real-Time AR Wallpaper",
    page_icon="📹",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("📹 AR-Шпалери в реальному часі")
st.caption("Наведіть камеру на стіну — шпалери накладаються на живому відео.")

# Конфігурація WebRTC з безкоштовними STUN-серверами Google для стабільного підключення з мобільних
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}
    ]}
)

# --- Налаштування у випадаючому меню ---
with st.expander("⚙️ Налаштування шпалер та ШІ", expanded=True):
    pattern_style = st.selectbox(
        "Оберіть стиль шпалер:",
        ["Смарагд + Золото", "Лофт Цегла", "Текстурний Льон", "Тропічний бамбук"]
    )
    sensitivity = st.slider("Обтікання меблів та рослин", 10, 80, 35)
    opacity = st.slider("Прозорість / Насиченість", 0.1, 1.0, 0.75)
    scale = st.slider("Масштаб малюнка", 0.5, 2.5, 1.0)


# --- Функція генерації текстури ---
def get_pattern(style, h, w, scale_factor=1.0):
    size = max(20, int(120 * scale_factor))
    tile = np.zeros((size, size, 3), dtype=np.uint8)

    if style == "Смарагд + Золото":
        tile[:] = (43, 61, 16) # BGR
        cv2.circle(tile, (size//2, size//2), int(size*0.35), (89, 175, 212), 3)
    elif style == "Лофт Цегла":
        tile[:] = (46, 59, 122)
        cv2.rectangle(tile, (4, 4), (size-4, size//2-4), (27, 35, 74), -1)
        cv2.rectangle(tile, (4, size//2+4), (size-4, size-4), (27, 35, 74), -1)
    elif style == "Текстурний Льон":
        tile[:] = (210, 215, 218)
        for i in range(0, size, 6):
            cv2.line(tile, (i, 0), (i, size), (185, 190, 195), 1)
    else:  # Тропічний бамбук
        tile[:] = (50, 90, 40)
        for i in range(10, size, 30):
            cv2.line(tile, (i, 0), (i+15, size), (80, 140, 60), 4)

    rx = int(np.ceil(w / size)) + 1
    ry = int(np.ceil(h / size)) + 1
    tiled = np.tile(tile, (ry, rx, 1))
    return tiled[:h, :w]


# --- Обробка кожного кадру відеопотоку на льоту ---
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # 1. Отримання кадру з камери
    img = frame.to_ndarray(format="bgr24")
    h, w, _ = img.shape

    # 2. Генерація текстури шпалер
    texture = get_pattern(pattern_style, h, w, scale)

    # 3. Виявлення меж об'єктів (меблі, рослини, картини)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, sensitivity, sensitivity * 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # 4. Формування маски стіни
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask[dilated_edges > 0] = 0

    # 5. Збереження природного освітлення та тіней
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, _, _ = cv2.split(lab)
    shadow_map = l_chan.astype(float) / 255.0
    shadow_map = np.clip(shadow_map * 1.15, 0, 1)

    # 6. Реалістичне накладання
    textured_wall = (texture.astype(float) * shadow_map[:, :, np.newaxis]).astype(np.uint8)
    smooth_mask = cv2.GaussianBlur(mask, (7, 7), 0)[:, :, np.newaxis] / 255.0
    alpha = opacity * smooth_mask

    result = (img * (1 - alpha) + textured_wall * alpha).astype(np.uint8)

    return av.VideoFrame.from_ndarray(result, format="bgr24")


# --- Запуск WebRTC стриму з захистом від системного плеєра Android ---
webrtc_streamer(
    key="wallpaper-realtime-ar",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {"facingMode": "environment"},  # Включає задню камеру смартфона
        "audio": False
    },
    video_html_attrs={
        "autoPlay": True,
        "controls": False,        # Приховує кнопка паузи/плеєра
        "style": {"width": "100%", "border-radius": "12px"},
        "playsinline": True,      # Обов'язково: грати на сторінці, а не у плеєрі Android
        "muted": True             # Вимикає звук для автоматичного запуску
    },
    async_processing=True,
)
