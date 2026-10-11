import streamlit as st
from PIL import Image
from collections import Counter
import io
import datetime
from ultralytics import YOLO

st.set_page_config(page_title="AccessLens 2(AN)K AI", page_icon="👁️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.7rem; max-width: 1200px;}
.hero {padding: 1.4rem; border-radius: 18px; background: linear-gradient(120deg,#101827,#243b55); color: white;}
.small-note {font-size: .88rem; color: #6b7280;}
.stat-box {padding: 0.6rem; border-radius: 10px; background: #eef2ff; text-align: center; margin-bottom: 0.5rem; border-left: 4px solid #4f46e5;}
.stat-num {font-size: 1.4rem; font-weight: bold; color: #4f46e5;}
.stat-label {font-size: 0.75rem; color: #6b7280;}
.alert-high {padding: 1.2rem; border-radius: 12px; background: #fee2e2; border-left: 6px solid #dc2626; margin-bottom: 1rem;}
.alert-high h3 {color: #7f1d1d !important; margin: 0 0 0.5rem 0; font-size: 1.4rem; font-weight: 700;}
.alert-high p {color: #991b1b !important; margin: 0; font-size: 1.05rem; font-weight: 500;}
.alert-med {padding: 1.2rem; border-radius: 12px; background: linear-gradient(90deg, #fef3c7 0%, #fde68a 100%); border-left: 6px solid #f59e0b; margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);}
.alert-med h3 {color: #78350f !important; margin: 0 0 0.5rem 0; font-size: 1.4rem; font-weight: 700;}
.alert-med p {color: #92400e !important; margin: 0; font-size: 1.05rem; font-weight: 500;}
.alert-low {padding: 1.2rem; border-radius: 12px; background: #d1fae5; border-left: 6px solid #10b981; margin-bottom: 1rem;}
.alert-low h3 {color: #064e3b !important; margin: 0 0 0.5rem 0; font-size: 1.4rem; font-weight: 700;}
.alert-low p {color: #065f46 !important; margin: 0; font-size: 1.05rem; font-weight: 500;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>👁️ AccessLens 2(AN)K.AI</h1><p>Visual scene understanding for more accessible digital experiences.</p></div>', unsafe_allow_html=True)
st.write("Upload a photo to receive AI-generated object detections with accessibility insights and safety alerts.")

# ---------- Session State Init ----------
if "history" not in st.session_state:
    st.session_state.history = []
if "total_objects" not in st.session_state:
    st.session_state.total_objects = 0

# ---------- Urdu Translation Dictionary ----------
URDU_DICT = {
    "person": "شخص", "bicycle": "سائیکل", "car": "گاڑی", "motorcycle": "موٹر سائیکل",
    "airplane": "ہوائی جہاز", "bus": "بس", "train": "ٹرین", "truck": "ٹرک",
    "boat": "کشتی", "traffic light": "ٹریفک لائٹ", "fire hydrant": "فائر ہائیڈرنٹ",
    "stop sign": "اسٹاپ سائن", "parking meter": "پارکنگ میٹر", "bench": "بینچ",
    "bird": "پرندہ", "cat": "بلی", "dog": "کتا", "horse": "گھوڑا", "sheep": "بھیڑ",
    "cow": "گائے", "elephant": "ہاتھی", "bear": "ریچھ", "zebra": "زیبرا",
    "giraffe": "زرافہ", "backpack": "بیگ", "umbrella": "چھتری", "handbag": "ہینڈ بیگ",
    "tie": "ٹائی", "suitcase": "سوٹ کیس", "frisbee": "فرسبی", "skis": "سکی",
    "snowboard": "سنو بورڈ", "sports ball": "گیند", "kite": "پتنگ",
    "baseball bat": "بیس بال بیٹ", "baseball glove": "بیس بال گلوز",
    "skateboard": "اسکیٹ بورڈ", "surfboard": "سرف بورڈ", "tennis racket": "ٹینس ریکٹ",
    "bottle": "بوتل", "wine glass": "وائن گلاس", "cup": "کپ", "fork": "کانٹا",
    "knife": "چاقو", "spoon": "چمچ", "bowl": "پیالہ", "banana": "کیلا",
    "apple": "سیب", "sandwich": "سینڈوچ", "orange": "سنترہ", "broccoli": "بروکلی",
    "carrot": "گاجر", "hot dog": "ہاٹ ڈاگ", "pizza": "پیزا", "donut": "ڈونٹ",
    "cake": "کیک", "chair": "کرسی", "couch": "صوفہ", "potted plant": "گملے کا پودا",
    "bed": "بستر", "dining table": "کھانے کی میز", "toilet": "ٹوائلٹ",
    "tv": "ٹی وی", "laptop": "لیپ ٹاپ", "mouse": "ماؤس", "remote": "ریموٹ",
    "keyboard": "کی بورڈ", "cell phone": "موبائل فون", "microwave": "مائیکروویو",
    "oven": "اوون", "toaster": "ٹوسٹر", "sink": "سنک", "refrigerator": "فریج",
    "book": "کتاب", "clock": "گھڑی", "vase": "گلدان", "scissors": "قینچی",
    "teddy bear": "ٹیڈی بیئر", "hair drier": "ہیئر ڈرائر", "toothbrush": "ٹوتھ برش",
}

def translate_label(label, lang):
    if lang == "English":
        return label
    return URDU_DICT.get(label.lower(), label)

def build_summary(detections, counts, lang):
    if not detections:
        return "No objects detected in the image."
    parts = []
    for name, count in counts.items():
        translated = translate_label(name, lang)
        if lang == "English":
            parts.append(f"{count} {translated}" + ("s" if count > 1 else ""))
        else:
            parts.append(f"{count} {translated}")
    if lang == "English":
        return "This scene contains " + ", ".join(parts) + "."
    else:
        return "اس منظر میں " + "، ".join(parts) + " موجود ہیں۔"

# ---------- 🚨 NEW: Obstacle Detection Logic ----------
HIGH_RISK_OBSTACLES = {
    "car", "truck", "bus", "motorcycle", "bicycle", "train",
    "fire hydrant", "stop sign", "traffic light", "parking meter"
}
MEDIUM_RISK_OBSTACLES = {
    "person", "chair", "bench", "dining table", "couch", "bed", "toilet"
}

def analyze_obstacles(detections, image_size):
    alerts = []
    total_area = image_size * image_size
    for d in detections:
        raw = d["Object_raw"]
        box_area = d.get("Box_Area", 0)
        area_ratio = box_area / total_area if total_area > 0 else 0
        if raw in HIGH_RISK_OBSTACLES:
            risk = "high" if area_ratio > 0.05 else "medium"
            alerts.append({"object": raw, "risk": risk, "area": area_ratio, "translated": d["Object"]})
        elif raw in MEDIUM_RISK_OBSTACLES:
            risk = "medium" if area_ratio > 0.08 else "low"
            alerts.append({"object": raw, "risk": risk, "area": area_ratio, "translated": d["Object"]})
    risk_order = {"high": 3, "medium": 2, "low": 1}
    alerts.sort(key=lambda x: (risk_order[x["risk"]], x["area"]), reverse=True)
    return alerts

def get_safety_level(alerts):
    if any(a["risk"] == "high" for a in alerts):
        return "HIGH RISK", "🔴", "alert-high"
    elif any(a["risk"] == "medium" for a in alerts):
        return "CAUTION", "🟡", "alert-med"
    else:
        return "SAFE", "🟢", "alert-low"

def build_alert_message(alerts, safety_level, lang):
    if not alerts:
        if lang == "English":
            return "Path appears clear. No major obstacles detected."
        else:
            return "راستہ صاف نظر آ رہا ہے۔ کوئی بڑی رکاوٹ نہیں ملی۔"
    top = alerts[0]
    if lang == "English":
        if safety_level == "HIGH RISK":
            return f"Warning! {top['object'].capitalize()} detected in your path. Please be careful."
        elif safety_level == "CAUTION":
            return f"Caution. {top['object'].capitalize()} detected nearby. Stay alert."
        else:
            return f"{top['object'].capitalize()} detected but at safe distance."
    else:
        if safety_level == "HIGH RISK":
            return f"خبردار! آپ کے راستے میں {top['translated']} موجود ہے۔ احتیاط کریں۔"
        elif safety_level == "CAUTION":
            return f"احتیاط! قریب {top['translated']} موجود ہے۔"
        else:
            return f"{top['translated']} نظر آیا لیکن محفوظ فاصلے پر ہے۔"

# ---------- Sidebar Settings ----------
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider("Detection Confidence", 0.1, 0.9, 0.25, 0.05)
    voice_output = st.checkbox("🔊 Enable Voice Output", value=False,
                               help="Requires internet + gtts library")
    language = st.selectbox("🌐 Language", ["English", "اردو (Urdu)"])

    st.divider()

    # ---------- Detection Presets ----------
    st.header("🎯 Detection Preset")
    preset = st.radio(
        "Choose scene type:",
        ["🌍 General", "🚦 Road/Street", "🏠 Indoor", "🍔 Food", "👤 People"],
        index=0,
        help="Presets tune detection for better results"
    )
    preset_map = {
        "🌍 General": 0.25,
        "🚦 Road/Street": 0.30,
        "🏠 Indoor": 0.25,
        "🍔 Food": 0.35,
        "👤 People": 0.40,
    }
    if preset != "🌍 General":
        confidence = preset_map[preset]

    st.divider()

    # ---------- Live Stats Dashboard ----------
    st.header("📊 Session Stats")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-num">{len(st.session_state.history)}</div>
            <div class="stat-label">Images</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-num">{st.session_state.total_objects}</div>
            <div class="stat-label">Objects</div>
        </div>
        """, unsafe_allow_html=True)

    # ---------- Session History ----------
    if st.session_state.history:
        with st.expander(f"📜 History ({len(st.session_state.history)})", expanded=False):
            for i, item in enumerate(reversed(st.session_state.history[-5:]), 1):
                st.caption(f"**{i}.** {item['name']} — {item['objects']} objects @ {item['time']}")
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.session_state.total_objects = 0
                st.rerun()

    st.divider()
    st.caption("🔖 **AccessLens 2(AN)K AI**")
    st.caption("🏆 Built for Banana Hacks 2026")

# ---------- Model Loader ----------
@st.cache_resource(show_spinner="Loading YOLO object detector...")
def load_yolo():
    return YOLO("yolov8n.pt")

# ---------- Helper Functions ----------
def get_accessibility_score(detections):
    if not detections:
        return 0, "No objects detected", "🔴"
    obstacle_labels = {"car", "truck", "bus", "motorcycle", "bicycle", "chair",
                       "bench", "fire hydrant", "stop sign"}
    obstacles = sum(1 for d in detections if d["Object_raw"] in obstacle_labels)
    diversity = len(set(d["Object_raw"] for d in detections))
    score = min(100, diversity * 10 + 30 - obstacles * 5)
    score = max(0, score)
    if score >= 75: level, emoji = "Excellent", "🟢"
    elif score >= 50: level, emoji = "Good", "🟡"
    elif score >= 25: level, emoji = "Moderate", "🟠"
    else: level, emoji = "Low", "🔴"
    return score, level, emoji

def get_category(label):
    categories = {
        "person": "👤 Person", "bicycle": "🚗 Vehicle", "car": "🚗 Vehicle",
        "motorcycle": "🚗 Vehicle", "bus": "🚗 Vehicle", "truck": "🚗 Vehicle",
        "chair": "🪑 Furniture", "couch": "🪑 Furniture", "bed": "🪑 Furniture",
        "dining table": "🪑 Furniture", "toilet": "🪑 Furniture",
        "bottle": "🍽️ Food/Drink", "cup": "🍽️ Food/Drink", "wine glass": "🍽️ Food/Drink",
        "bowl": "🍽️ Food/Drink", "banana": "🍽️ Food/Drink", "apple": "🍽️ Food/Drink",
        "sandwich": "🍽️ Food/Drink", "orange": "🍽️ Food/Drink", "broccoli": "🍽️ Food/Drink",
        "laptop": "📱 Electronic", "tv": "📱 Electronic", "cell phone": "📱 Electronic",
        "keyboard": "📱 Electronic", "mouse": "📱 Electronic",
        "dog": "🐾 Animal", "cat": "🐾 Animal", "bird": "🐾 Animal", "horse": "🐾 Animal",
    }
    return categories.get(label, "📦 Other")

# ---------- Main App ----------
uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("📷 Your image")
        st.image(image, use_container_width=True)
        analyze = st.button("🔍 Analyze Image", type="primary", use_container_width=True)

    if analyze:
        try:
            progress = st.progress(0, text="Starting analysis...")

            progress.progress(30, text="🔍 Detecting objects with YOLO...")
            detector = load_yolo()
            results = detector.predict(image, conf=confidence, verbose=False)
            result = results[0]
            annotated = result.plot()
            import cv2
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            progress.progress(70, text="✅ Objects detected")

            names = result.names
            labels_raw = [names[int(box.cls[0])] for box in result.boxes]
            counts = Counter(labels_raw)
            detections = []
            img_w, img_h = image.size

            for box in result.boxes:
                raw = names[int(box.cls[0])]
                xyxy = box.xyxy[0].tolist()
                box_w = xyxy[2] - xyxy[0]
                box_h = xyxy[3] - xyxy[1]
                box_area = box_w * box_h
                detections.append({
                    "Object": translate_label(raw, language),
                    "Object_raw": raw,
                    "Category": get_category(raw),
                    "Confidence": f"{float(box.conf[0]):.0%}",
                    "Conf_Raw": float(box.conf[0]),
                    "Box_Area": box_area,
                })

            score, level, emoji = get_accessibility_score(detections)
            summary = build_summary(detections, counts, language)

            # 🚨 Obstacle Analysis
            progress.progress(85, text="🚨 Checking for obstacles...")
            alerts = analyze_obstacles(detections, max(img_w, img_h))
            safety_level, safety_emoji, safety_class = get_safety_level(alerts)
            alert_msg = build_alert_message(alerts, safety_level, language)

            st.session_state.history.append({
                "name": uploaded.name,
                "objects": len(detections),
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
            })
            st.session_state.total_objects += len(detections)

            progress.progress(100, text="✅ Analysis complete!")
            progress.empty()

            with right:
                st.subheader("🤖 AI Visual Analysis")
                st.image(annotated_rgb, caption="Detected objects (boxes are model predictions)", use_container_width=True)

                # 🚨 OBSTACLE ALERT BOX
                st.markdown(f"""
                <div class="{safety_class}">
                    <h3 style="margin: 0;">{safety_emoji} Obstacle Alert: {safety_level}</h3>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.05rem;">{alert_msg}</p>
                </div>
                """, unsafe_allow_html=True)

                if alerts:
                    st.markdown("##### 🚨 Detected Obstacles")
                    for a in alerts[:5]:
                        risk_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[a["risk"]]
                        risk_text = {"high": "Close / High Risk", "medium": "Nearby / Caution", "low": "Safe Distance"}[a["risk"]]
                        st.caption(f"{risk_icon} **{a['translated']}** — {risk_text}")

                # Voice output (alert + summary)
                if voice_output and detections:
                    try:
                        from gtts import gTTS
                        voice_text = alert_msg + " " + summary
                        tts = gTTS(text=voice_text, lang='en' if language == "English" else 'ur')
                        audio_bytes = io.BytesIO()
                        tts.write_to_fp(audio_bytes)
                        audio_bytes.seek(0)
                        st.audio(audio_bytes, format='audio/mp3')
                    except Exception as e:
                        st.caption(f"⚠️ Voice failed: {str(e)[:60]}. Run: pip install gtts")

                st.markdown("#### ♿ Accessibility Score")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Score", f"{score}/100")
                with col2:
                    st.metric("Level", f"{emoji} {level}")
                st.progress(score / 100)

                st.markdown("#### 📝 Scene Summary")
                st.write(summary)

                st.markdown("#### 🎯 Detected Objects")
                if detections:
                    all_cats = sorted(set(d["Category"] for d in detections))
                    selected_cats = st.multiselect("Filter by category", all_cats, default=all_cats)
                    filtered = [d for d in detections if d["Category"] in selected_cats]

                    st.dataframe(
                        [{k: v for k, v in d.items() if k not in ("Conf_Raw", "Object_raw", "Box_Area")} for d in filtered],
                        use_container_width=True, hide_index=True
                    )
                    st.caption("Object counts: " + ", ".join(
                        f"{translate_label(name, language)}: {count}" for name, count in counts.items()
                    ))

                    st.markdown("#### 📊 Confidence Distribution")
                    import pandas as pd
                    chart_data = pd.DataFrame(
                        {"Object": [d["Object"] for d in detections],
                         "Confidence": [d["Conf_Raw"] for d in detections]}
                    )
                    st.bar_chart(chart_data, x="Object", y="Confidence")
                else:
                    st.info("No objects were detected above the selected confidence threshold.")

                st.markdown("#### 📥 Download Report")
                report = f"""AccessLens 2(AN)K AI - Analysis Report
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
File: {uploaded.name}
Language: {language}
Preset: {preset}

========================================
OBSTACLE ALERT: {safety_level}
{alert_msg}
========================================

SCENE SUMMARY:
{summary}

ACCESSIBILITY SCORE: {score}/100 ({level})

DETECTED OBJECTS ({len(detections)} total):
"""
                for d in detections:
                    report += f"- {d['Object']} ({d['Category']}): {d['Confidence']}\n"

                if alerts:
                    report += "\nOBSTACLE DETAILS:\n"
                    for a in alerts:
                        report += f"- {a['translated']} [{a['risk'].upper()} RISK]\n"

                st.download_button(
                    "⬇️ Download Report (.txt)",
                    data=report,
                    file_name=f"accesslens2_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            st.divider()
            st.subheader("💡 What this means")
            st.write("The detected objects, summary, and obstacle alerts are based on YOLO model predictions. Check the image yourself: models can miss objects or mislabel items.")
            st.info("🔬 **AccessLens 2(AN)K AI** is an experimental accessibility prototype designed for hackathon demonstration.")

        except Exception as exc:
            st.error("The analysis could not be completed.")
            st.exception(exc)

else:
    st.info("👆 Upload a clear photo to begin. For a reliable demo, try a scene with several everyday objects and good lighting.")

    st.markdown("### 🚨 Smart Obstacle Alert System")
    st.markdown("""
    AccessLens 2(AN)K AI now **automatically detects potential obstacles** using bounding box analysis:
    - 🔴 **HIGH RISK** — Vehicles or large objects very close
    - 🟡 **CAUTION** — Objects nearby, stay alert
    - 🟢 **SAFE** — Path appears clear
    """)

    st.markdown("### ✨ Key Features")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**🚨 Obstacle Alerts**\n\nReal-time path warnings")
    c2.markdown("**🔍 Object Detection**\n\nYOLOv8 identifies 80+ types")
    c3.markdown("**🌐 Multi-language**\n\nEnglish + اردو support")
    c4.markdown("**🔊 Voice Output**\n\nListen to alerts & summary")

    st.markdown("### 🎯 Detection Presets")
    st.markdown("Choose from **General**, **Road/Street**, **Indoor**, **Food**, or **People** modes in the sidebar for optimized results.")

    st.markdown("### 🎯 Designed for")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Visual understanding**\n\nTurn a photo into a scene summary.")
    c2.markdown("**Visible evidence**\n\nSee object labels and confidence scores.")
    c3.markdown("**Safe navigation**\n\nObstacle alerts for accessibility.")