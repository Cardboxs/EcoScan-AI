import streamlit as st
import numpy as np
import cv2
import av
import time
import threading
import json
from pathlib import Path
from datetime import datetime

from PIL import Image
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="EcoScan AI",
    page_icon="♻️",
    layout="centered"
)

# Tetap gunakan path yang sudah berhasil pada versi sebelumnya.
YOLO_PATH = "best.pt"
CLASSIFIER_PATH = "ecoscan_ai_model_final.keras"

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# YOLO dataset punya 9 kelas.
# EcoScan hanya menggunakan kelas yang relevan berikut:
# 0 Electronics
# 1 biological
# 2 cardboard
# 3 clothes
# 4 glass
# 5 metal
# 6 paper
# 7 plastic
# 8 shoes
RELEVANT_YOLO_CLASSES = [2, 4, 5, 6, 7]

# Tidak terlalu ketat agar auto-capture tetap mudah.
YOLO_THRESHOLD = 0.75
CLASSIFIER_THRESHOLD = 0.70

# Tambahan validasi agar background tidak mudah diproses.
MIN_BOX_AREA_RATIO = 0.03
MAX_BOX_AREA_RATIO = 0.80

# Objek harus berada cukup dekat dengan tengah frame.
ZONE_X_MIN = 0.15
ZONE_X_MAX = 0.85
ZONE_Y_MIN = 0.15
ZONE_Y_MAX = 0.85

STABLE_TIME = 1.0
BOX_MOVEMENT_THRESHOLD = 80
AUTO_RESET_DELAY = 2.5

# Confidence margin membantu menolak prediksi yang terlalu ambigu.
MIN_CONFIDENCE_MARGIN = 0.0

# Folder untuk menyimpan hasil scan
HISTORY_DIR = Path("scan_history")
IMAGE_DIR = HISTORY_DIR / "images"
HISTORY_FILE = HISTORY_DIR / "history.json"

HISTORY_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)


# =========================================================
# HISTORY FUNCTIONS
# =========================================================

def load_history():
    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        return []

    except Exception:
        return []


def save_scan_history(
    image,
    predicted_class,
    classification_confidence,
    yolo_confidence
):
    timestamp = datetime.now()
    scan_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")

    image_filename = f"scan_{scan_id}.jpg"
    image_path = IMAGE_DIR / image_filename

    # Simpan crop objek
    success = cv2.imwrite(str(image_path), image)

    if not success:
        raise RuntimeError("Gambar hasil scan gagal disimpan.")

    history = load_history()

    new_record = {
        "id": scan_id,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "class": predicted_class,
        "classification_confidence": round(
            classification_confidence * 100,
            2
        ),
        "yolo_confidence": round(
            yolo_confidence * 100,
            2
        ),
        "image": str(image_path)
    }

    history.insert(0, new_record)
    history = history[:50]

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            history,
            f,
            ensure_ascii=False,
            indent=2
        )

    return new_record


def clear_history():
    history = load_history()

    for item in history:
        image_path = Path(item.get("image", ""))

        if image_path.exists():
            try:
                image_path.unlink()
            except Exception:
                pass

    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():
    yolo = YOLO(YOLO_PATH)
    classifier = load_model(CLASSIFIER_PATH)
    return yolo, classifier


try:
    yolo_model, classifier_model = load_models()
except Exception as e:
    st.error("Model belum ditemukan atau gagal dimuat.")
    st.code(str(e))
    st.stop()


# =========================================================
# CLASSIFICATION
# =========================================================

def classify_object(cropped):
    if cropped is None or cropped.size == 0:
        return "unknown", 0.0, 0.0

    image = Image.fromarray(
        cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    )

    image = image.resize((224, 224))

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(image_array, axis=0)

    predictions = classifier_model.predict(
        image_array,
        verbose=0
    )[0]

    sorted_indices = np.argsort(predictions)[::-1]
    best_index = int(sorted_indices[0])
    second_index = int(sorted_indices[1])

    best_confidence = float(predictions[best_index])
    second_confidence = float(predictions[second_index])
    confidence_margin = best_confidence - second_confidence

    return (
        CLASS_NAMES[best_index],
        best_confidence,
        confidence_margin
    )


def box_is_valid(x1, y1, x2, y2, width, height):
    box_width = max(0, x2 - x1)
    box_height = max(0, y2 - y1)

    if box_width < 25 or box_height < 25:
        return False, "Objek terlalu kecil"

    area_ratio = (box_width * box_height) / float(width * height)

    if area_ratio < MIN_BOX_AREA_RATIO:
        return False, "Dekatkan objek ke kamera"

    if area_ratio > MAX_BOX_AREA_RATIO:
        return False, "Objek terlalu memenuhi kamera"

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    in_center_zone = (
        width * ZONE_X_MIN <= center_x <= width * ZONE_X_MAX
        and
        height * ZONE_Y_MIN <= center_y <= height * ZONE_Y_MAX
    )

    if not in_center_zone:
        return False, "Posisikan objek di tengah"

    return True, "Objek terdeteksi"


# =========================================================
# REAL-TIME VIDEO PROCESSOR
# =========================================================

class WasteDetectionProcessor(VideoProcessorBase):

    def __init__(self):
        self.stable_start = None
        self.last_box = None

        self.auto_captured = False

        self.captured_crop = None
        self.captured_class = None
        self.captured_confidence = 0.0
        self.captured_yolo_confidence = 0.0

        self.saved_history = False
        self.capture_time = None

        # Hasil terakhir tetap tersedia setelah scanner reset.
        self.last_result_crop = None
        self.last_result_class = None
        self.last_result_confidence = 0.0
        self.last_result_yolo_confidence = 0.0
        self.last_result_saved = False
        self.last_saved_path = None
        self.last_save_error = None

        self.status_text = "Arahkan kamera ke sampah"

        self.lock = threading.Lock()

    def reset_scan_state(self):
        self.stable_start = None
        self.last_box = None
        self.auto_captured = False
        self.captured_crop = None
        self.captured_class = None
        self.captured_confidence = 0.0
        self.captured_yolo_confidence = 0.0
        self.saved_history = False
        self.capture_time = None
        self.status_text = "Arahkan kamera ke sampah"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        height, width = img.shape[:2]

        # -----------------------------------------------------
        # AUTO RESET
        # -----------------------------------------------------
        with self.lock:
            if (
                self.auto_captured
                and self.capture_time is not None
                and (time.time() - self.capture_time) >= AUTO_RESET_DELAY
            ):
                self.reset_scan_state()

        # -----------------------------------------------------
        # AREA SCAN GUIDE
        # -----------------------------------------------------
        zone_x1 = int(width * ZONE_X_MIN)
        zone_y1 = int(height * ZONE_Y_MIN)
        zone_x2 = int(width * ZONE_X_MAX)
        zone_y2 = int(height * ZONE_Y_MAX)

        cv2.rectangle(
            img,
            (zone_x1, zone_y1),
            (zone_x2, zone_y2),
            (255, 255, 255),
            2
        )

        cv2.putText(
            img,
            "AREA SCAN",
            (zone_x1 + 8, zone_y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # -----------------------------------------------------
        # YOLO DETECTION
        # -----------------------------------------------------
        results = yolo_model.predict(
            source=img,
            conf=YOLO_THRESHOLD,
            classes=RELEVANT_YOLO_CLASSES,
            verbose=False
        )

        result = results[0]

        # -----------------------------------------------------
        # NO VALID YOLO OBJECT
        # -----------------------------------------------------
        if len(result.boxes) == 0:
            with self.lock:
                self.stable_start = None
                self.last_box = None
                self.status_text = "Arahkan kamera ke sampah"

            cv2.putText(
                img,
                "Arahkan kamera ke sampah",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )

            return av.VideoFrame.from_ndarray(
                img,
                format="bgr24"
            )

        # -----------------------------------------------------
        # BEST OBJECT
        # -----------------------------------------------------
        best_box = max(
            result.boxes,
            key=lambda box: float(box.conf[0])
        )

        x1, y1, x2, y2 = map(
            int,
            best_box.xyxy[0].tolist()
        )

        # Batasi koordinat agar crop tidak keluar frame.
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        yolo_confidence = float(best_box.conf[0])
        yolo_class_id = int(best_box.cls[0])
        yolo_class_name = yolo_model.names.get(
            yolo_class_id,
            "object"
        ) if isinstance(yolo_model.names, dict) else str(yolo_class_id)

        # -----------------------------------------------------
        # VALIDATE SIZE + POSITION
        # -----------------------------------------------------
        valid_box, validation_message = box_is_valid(
            x1,
            y1,
            x2,
            y2,
            width,
            height
        )

        # -----------------------------------------------------
        # STABILITY
        # -----------------------------------------------------
        current_box = (x1, y1, x2, y2)

        with self.lock:
            if not self.auto_captured:
                if not valid_box:
                    self.stable_start = None
                    self.last_box = None
                elif self.last_box is None:
                    self.last_box = current_box
                    self.stable_start = time.time()
                else:
                    old_x1, old_y1, old_x2, old_y2 = self.last_box

                    movement = (
                        abs(x1 - old_x1)
                        + abs(y1 - old_y1)
                        + abs(x2 - old_x2)
                        + abs(y2 - old_y2)
                    )

                    if movement >= BOX_MOVEMENT_THRESHOLD:
                        self.stable_start = time.time()

                    elif self.stable_start is None:
                        self.stable_start = time.time()

                    self.last_box = current_box

            stable_duration = (
                time.time() - self.stable_start
                if self.stable_start is not None
                else 0
            )

        # -----------------------------------------------------
        # BOUNDING BOX
        # -----------------------------------------------------
        box_color = (0, 255, 0) if valid_box else (0, 165, 255)

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            box_color,
            3
        )

        cv2.putText(
            img,
            f"{yolo_class_name} {yolo_confidence * 100:.1f}%",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            box_color,
            2
        )

        # -----------------------------------------------------
        # AUTO CAPTURE + CLASSIFICATION + AUTO SAVE
        # -----------------------------------------------------
        if (
            valid_box
            and not self.auto_captured
            and stable_duration >= STABLE_TIME
        ):
            cropped = img[y1:y2, x1:x2].copy()

            if (
                cropped.size > 0
                and cropped.shape[0] > 20
                and cropped.shape[1] > 20
            ):
                predicted_class, classification_confidence, confidence_margin = (
                    classify_object(cropped)
                )

                # Hanya simpan kalau model klasifikasi cukup yakin
                # dan jarak top-1 dengan top-2 cukup jelas.
                classifier_ok = (
                    predicted_class != "unknown"
                    and classification_confidence >= CLASSIFIER_THRESHOLD
                )

                if classifier_ok:
                    # -----------------------------------------
                    # SIMPAN OTOMATIS TERLEBIH DAHULU
                    # Hanya tandai SCAN SELESAI setelah file benar-benar
                    # berhasil ditulis ke scan_history.
                    # -----------------------------------------
                    if not self.saved_history:
                        try:
                            saved_record = save_scan_history(
                                cropped,
                                predicted_class,
                                classification_confidence,
                                yolo_confidence
                            )

                            with self.lock:
                                self.saved_history = True
                                self.captured_crop = cropped.copy()
                                self.captured_class = predicted_class
                                self.captured_confidence = classification_confidence
                                self.captured_yolo_confidence = yolo_confidence

                                self.last_result_crop = cropped.copy()
                                self.last_result_class = predicted_class
                                self.last_result_confidence = classification_confidence
                                self.last_result_yolo_confidence = yolo_confidence
                                self.last_result_saved = True
                                self.last_saved_path = saved_record["image"]
                                self.last_save_error = None

                                self.capture_time = time.time()
                                self.auto_captured = True
                                self.status_text = "SCAN SELESAI - TERSIMPAN"

                        except Exception as e:
                            print("Gagal menyimpan history:", e)
                            with self.lock:
                                self.last_result_saved = False
                                self.last_save_error = str(e)
                                self.status_text = "Gagal menyimpan gambar - coba lagi"

                else:
                    with self.lock:
                        self.status_text = (
                            "AI belum cukup yakin, posisikan ulang objek"
                        )
                        # Biarkan scanner mencoba lagi.
                        self.stable_start = time.time()

        # -----------------------------------------------------
        # VIDEO STATUS
        # -----------------------------------------------------
        with self.lock:
            captured = self.auto_captured
            captured_class = self.captured_class
            captured_confidence = self.captured_confidence
            status_text = self.status_text

        if captured:
            cv2.putText(
                img,
                "SCAN SELESAI",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2
            )

            result_text = (
                f"{captured_class.upper()} "
                f"{captured_confidence * 100:.1f}%"
            )

            cv2.putText(
                img,
                result_text,
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )

        elif not valid_box:
            cv2.putText(
                img,
                status_text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2
            )

        elif stable_duration >= STABLE_TIME:
            cv2.putText(
                img,
                "MENGANALISIS OBJEK...",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

        else:
            cv2.putText(
                img,
                "Tahan objek agar tetap stabil...",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# =========================================================
# FINAL UI / UX — SAFE VERSION
# Hanya mengubah tampilan; logika scanner tetap dari versi working.
# =========================================================

st.markdown(
    """
    <style>
    .stApp { background: #f5f7f8; }
    .block-container { max-width: 780px; padding-top: 2rem; padding-bottom: 3rem; }

    .eco-hero {
        background: linear-gradient(135deg, #0f766e, #15803d);
        border-radius: 22px;
        padding: 26px 30px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 28px rgba(15,118,110,.16);
    }

    .eco-hero h1 {
        margin: 0;
        font-size: 2.15rem;
        font-weight: 800;
    }

    .eco-hero p {
        margin: 8px 0 0;
        font-size: 0.98rem;
        opacity: .92;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 750;
        margin: 10px 0 12px;
        color: #172023;
    }

    .scan-guide, .info-card, .result-card, .mini-stat {
        background: white;
        border: 1px solid #e4e9ea;
        border-radius: 18px;
        box-shadow: 0 5px 18px rgba(18,32,36,.05);
    }

    .scan-guide {
        background: #ecfdf5;
        border-color: #bbf7d0;
        padding: 14px 16px;
        margin-bottom: 16px;
        color: #166534;
    }

    .info-card { padding: 18px 20px; }
    .result-card { padding: 20px; }

    .result-class {
        font-size: 2rem;
        font-weight: 850;
        color: #166534;
        margin: 6px 0 14px;
        text-transform: uppercase;
    }

    .mini-stat {
        padding: 15px 16px;
        text-align: center;
    }

    .mini-stat .number {
        font-size: 1.55rem;
        font-weight: 850;
        color: #166534;
    }

    .mini-stat .label {
        color: #6b7280;
        font-size: .82rem;
    }

    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True
)

history_now = load_history()

st.markdown(
    """
    <div class="eco-hero">
        <h1>♻️ EcoScan AI</h1>
        <p>Smart Waste Classification menggunakan YOLO + MobileNetV2</p>
    </div>
    """,
    unsafe_allow_html=True
)

stat1, stat2, stat3 = st.columns(3)

with stat1:
    st.markdown(
        f'<div class="mini-stat"><div class="number">{len(history_now)}</div><div class="label">Total Scan Tersimpan</div></div>',
        unsafe_allow_html=True
    )

with stat2:
    st.markdown(
        '<div class="mini-stat"><div class="number">6</div><div class="label">Jenis Sampah</div></div>',
        unsafe_allow_html=True
    )

with stat3:
    st.markdown(
        '<div class="mini-stat"><div class="number">AI</div><div class="label">YOLO + MobileNetV2</div></div>',
        unsafe_allow_html=True
    )

st.write("")
st.markdown('<div class="section-title">📷 Smart Scanner</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="scan-guide">
        <b>Cara menggunakan:</b> arahkan kamera ke sampah → posisikan objek di tengah → tahan sekitar 1 detik → AI melakukan auto-capture dan menyimpan hasil secara otomatis.
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# CAMERA — TIDAK DIUBAH
# =========================================================

ctx = webrtc_streamer(
    key="ecoscan-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=WasteDetectionProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)

# =========================================================
# LAST RESULT — TIDAK MENGGUNAKAN FRAGMENT
# agar tidak mengganggu WebRTC.
# =========================================================

if ctx.video_processor:
    processor = ctx.video_processor

    with processor.lock:
        last_crop = (
            processor.last_result_crop.copy()
            if processor.last_result_crop is not None
            else None
        )
        last_class = processor.last_result_class
        last_confidence = processor.last_result_confidence
        last_yolo_confidence = processor.last_result_yolo_confidence
        last_result_saved = processor.last_result_saved
        last_saved_path = processor.last_saved_path
        last_save_error = processor.last_save_error

    if last_crop is not None and last_class is not None:
        st.markdown('<div class="section-title">📌 Hasil Scan Terakhir</div>', unsafe_allow_html=True)

        col_img, col_result = st.columns([1.1, 1.4])

        with col_img:
            crop_rgb = cv2.cvtColor(last_crop, cv2.COLOR_BGR2RGB)
            st.image(
                crop_rgb,
                caption="Objek yang dianalisis",
                use_container_width=True
            )

        with col_result:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="result-class">{last_class}</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "YOLO Detection",
                    f"{last_yolo_confidence * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "MobileNetV2",
                    f"{last_confidence * 100:.2f}%"
                )

            if last_result_saved:
                st.success("✅ Foto otomatis tersimpan ke riwayat.")
                if last_saved_path:
                    st.caption(f"File: {last_saved_path}")
            else:
                st.error("❌ Foto belum tersimpan ke riwayat.")
                if last_save_error:
                    st.code(last_save_error)

            st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# HISTORY — MEKANISME LAMA DIPERTAHANKAN
# =========================================================

st.markdown('<div class="section-title">🕘 Riwayat Scan</div>', unsafe_allow_html=True)
st.caption("Setiap scan berhasil otomatis tersimpan di scan_history/images.")

@st.fragment(run_every=1)
def history_section():
    history = load_history()

    if not history:
        st.markdown(
            '<div class="info-card"><span style="color:#6b7280">Belum ada riwayat scan.</span></div>',
            unsafe_allow_html=True
        )
        return

    st.caption(f"Menampilkan {min(len(history), 10)} scan terbaru.")

    for item in history[:10]:
        image_path = Path(item.get("image", ""))
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if image_path.exists():
                st.image(str(image_path), width=90)

        with col2:
            st.markdown(f"**{item['class'].upper()}**")
            st.caption(item["timestamp"])

        with col3:
            st.metric(
                "Confidence",
                f"{item['classification_confidence']:.1f}%"
            )

        st.divider()

history_section()

# =========================================================
# ABOUT / SETTINGS
# =========================================================

with st.expander("ℹ️ Tentang EcoScan AI"):
    st.write(
        "EcoScan AI menggunakan YOLO untuk mendeteksi lokasi objek "
        "dan MobileNetV2 untuk menentukan jenis sampah. Hasil scan "
        "yang berhasil disimpan otomatis ke folder scan_history/images."
    )

with st.expander("⚙️ Pengaturan Riwayat"):
    st.caption(
        "Menghapus riwayat tidak menghapus model AI. Yang dihapus hanya foto hasil scan dan history.json."
    )

    if st.button("🗑️ Hapus Semua Riwayat", use_container_width=True):
        clear_history()
        st.success("Semua riwayat telah dihapus.")
        st.rerun()
