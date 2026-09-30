import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    VideoProcessorBase
)

# Налаштування сторінки
st.set_page_config(
    page_title="Real-Time AR Wallpaper",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🖼️ AR-Шпалери у реальному часі")
st.caption("Натисніть **START** нижче, надайте дозвіл камери та оберіть потрібну камеру.")

# Стилізація для приховування зайвого технічного мусору WebRTC
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    div[data-testid="stExpander"] { border: 1px solid #2e384d; border-radius: 12px; }
    </style>
""", unsafe_allow_html=True)


# --- Клас обробки відеопотоку у реальному часі ---
class AIWallpaperProcessor(VideoProcessorBase):
    def __init__(self):
        self.pattern_style = "Смарагд + Золото"
        self.sensitivity = 35
        self.opacity = 0.75
        self.scale = 1.0

    def generate_pattern(self, style, h, w, scale_factor):
        size = max(20, int(120 * scale_factor))
        tile = np.zeros((size, size, 3), dtype=np.uint8)

        if style == "Смарагд + Золото":
            tile[:] = (43, 61, 16)  # BGR
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

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        try:
            # 1. Текстура
            texture = self.generate_pattern(self.pattern_style, h, w, self.scale)

            # 2. Виявлення меж (меблі, рослини)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, self.sensitivity, self.sensitivity * 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(edges, kernel, iterations=1)

            # 3. Маска стіни
            mask = np.ones((h, w), dtype=np.uint8) * 255
            mask[dilated > 0] = 0

            # 4. Тіні та освітлення
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_chan, _, _ = cv2.split(lab)
            shadow_map = l_chan.astype(float) / 255.0
            shadow_map = np.clip(shadow_map * 1.15, 0, 1)

            # 5. Накладання
            textured_wall = (texture.astype(float) * shadow_map[:, :, np.newaxis]).astype(np.uint8)
            smooth_mask = cv2.GaussianBlur(mask, (7, 7), 0)[:, :, np.newaxis] / 255.0
            alpha = self.opacity * smooth_mask

            res = (img * (1 - alpha) + textured_wall * alpha).astype(np.uint8)
            return av.VideoFrame.from_ndarray(res, format="bgr24")

        except Exception:
            # Якщо виникла помилка під час обробки кадру — повертаємо оригінал, щоб відео не зависало
            return frame


# --- Налаштування у випадаючому меню ---
with st.expander("⚙️ Налаштування шпалер та ШІ", expanded=True):
    pattern_style = st.selectbox(
        "Оберіть стиль шпалер:",
        ["Смарагд + Золото", "Лофт Цегла", "Текстурний Льон", "Тропічний бамбук"]
    )
    sensitivity = st.slider("Обтікання меблів та рослин", 10, 80, 35)
    opacity = st.slider("Прозорість / Насиченість", 0.1, 1.0, 0.75)
    scale = st.slider("Масштаб малюнка", 0.5, 2.5, 1.0)


# --- Налаштування WebRTC ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_ctx = webrtc_streamer(
    key="ai-wallpaper-stream",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=AIWallpaperProcessor,
    media_stream_constraints={"video": True, "audio": False},
    video_html_attrs={
        "autoPlay": True,
        "controls": False,
        "style": {"width": "100%", "border-radius": "12px"},
        "playsinline": True,
        "muted": True
    },
    async_processing=True,
)

# Передача параметрів з меню прямо в обробник відеокадрів
if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.pattern_style = pattern_style
    webrtc_ctx.video_processor.sensitivity = sensitivity
    webrtc_ctx.video_processor.opacity = opacity
    webrtc_ctx.video_processor.scale = scale
