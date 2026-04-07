import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Food Detection and Calorie Estimation",
    page_icon="🍽️",
    layout="wide"
)

# -----------------------------
# Custom Styling (YOUR ORIGINAL)
# -----------------------------
st.markdown("""
<style>
.title {
font-size:42px;
font-weight:800;
text-align:center;
background: linear-gradient(90deg,#ff416c,#ff4b2b,#ffb347);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
margin-bottom:5px;
}
.subtitle {
text-align:center;
font-size:18px;
color:#8aa0b4;
margin-bottom:5px;
}
.banner {
background:linear-gradient(135deg,#1f4037,#99f2c8);
padding:12px;
border-radius:12px;
text-align:center;
font-size:16px;
font-weight:600;
color:black;
margin-top:10px;
margin-bottom:20px;
}
.card{
background:linear-gradient(135deg,#667eea,#764ba2);
padding:8px;
border-radius:10px;
text-align:center;
color:white;
margin-bottom:8px;
font-size:14px;
}
.sidebar-title{
font-size:22px;
font-weight:700;
text-align:center;
margin-bottom:10px;
color:#ff7a18;
}
.sidebar-box{
background:linear-gradient(135deg,#1c1c1c,#2c3e50);
padding:10px;
border-radius:10px;
margin-bottom:10px;
color:white;
text-align:center;
}
.support-text{
font-size:13px;
color:#9ca3af;
margin-top:-8px;
text-align:center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="title">🍽️ FOOD DETECTION AND CALORIE ESTIMATION</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a food image and detect items using your trained YOLO model</div>', unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown('<div class="sidebar-title">⚙ Detection Settings</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-box">Adjust confidence level for detection</div>', unsafe_allow_html=True)

confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)

st.sidebar.markdown("### 📷 Upload Food Image")

uploaded_file = st.sidebar.file_uploader("Upload", type=["jpg","jpeg","png"])

st.sidebar.markdown(
    '<div class="support-text">Supported formats: JPG • JPEG • PNG</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Banner
# -----------------------------
if uploaded_file is None:
    st.markdown('<div class="banner">📷 Upload a food image using the sidebar to start food detection</div>', unsafe_allow_html=True)

# -----------------------------
# Calories
# -----------------------------
calorie_dict = {
    "apple":95,
    "banana":105,
    "orange":62,
    "pizza":285,
    "burger":354
}

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best_new.pt")

# -----------------------------
# Main Logic
# -----------------------------
if uploaded_file is not None:

    st.subheader("🔍 Food Detection Results")

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        temp_path = temp.name

    with st.spinner("⏳ Loading model..."):
        model = load_model()

    results = model(temp_path, conf=confidence)

    annotated = results[0].plot()

    with col2:
        st.subheader("🎯 Detection Output")
        st.image(annotated, channels="BGR", use_container_width=True)

    detections = []
    names = model.names

    for r in results:
        if r.boxes is None:
            continue

        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        for c, conf_score in zip(cls_ids, confs):
            detections.append({
                "Food Item": names[c],
                "Confidence": round(float(conf_score), 3)
            })

    if len(detections) > 0:

        df = pd.DataFrame(detections)

        st.subheader("🍎 Detected Food Items")

        counts = df["Food Item"].value_counts()

        # ✅ FIXED: smaller + aligned grid
        cols = st.columns(3)
        i = 0

        for food, count in counts.items():
            avg_conf = df[df["Food Item"] == food]["Confidence"].mean() * 100

            with cols[i % 3]:
                st.markdown(f"""
                <div class="card">
                <b>{food.capitalize()}</b><br>
                {count} item(s)<br>
                {avg_conf:.1f}% confidence
                </div>
                """, unsafe_allow_html=True)
            i += 1

        # Nutrition Table
        st.subheader("📊 Nutrition Table")

        nutrition = []
        total_calories = 0

        for food, count in counts.items():
            calories = calorie_dict.get(food, 50)
            total = calories * count
            total_calories += total

            nutrition.append({
                "Food Item": food,
                "Count": count,
                "Calories": total
            })

        nutrition_df = pd.DataFrame(nutrition)
        st.dataframe(nutrition_df, use_container_width=True)

        st.markdown(
        f"<h2 style='color:#ff4b2b'>🔥 Total Estimated Calories: {total_calories} kcal</h2>",
        unsafe_allow_html=True
        )

        # -----------------------------
        # 🔥 UPDATED UNIQUE DONUT CHART
        # -----------------------------
        st.subheader("🥧 Calorie Distribution")

        fig, ax = plt.subplots()

        colors = plt.cm.Set3.colors

        wedges, texts, autotexts = ax.pie(
            nutrition_df["Calories"],
            labels=nutrition_df["Food Item"],
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops=dict(width=0.4)
        )

        # center circle
        centre_circle = plt.Circle((0,0),0.60,fc='white')
        fig.gca().add_artist(centre_circle)

        # center text
        ax.text(0, 0, f"{total_calories}\nkcal",
                ha='center', va='center', fontsize=12, fontweight='bold')

        ax.axis("equal")

        st.pyplot(fig)

    else:
        st.warning("No food items detected.")

    os.remove(temp_path)