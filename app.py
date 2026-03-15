import streamlit as st
import keras
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import os

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="WasteAI - Smart Waste Segregation",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.60

CLASS_COLORS = {
    "cardboard": "#FF6B6B",
    "glass":     "#4ECDC4",
    "metal":     "#FFE66D",
    "paper":     "#96E6A1",
    "plastic":   "#A29BFE",
    "trash":     "#B2BEC3",
}

CLASS_ICONS = {
    "cardboard": "📦",
    "glass":     "🥛",
    "metal":     "🥫",
    "paper":     "📄",
    "plastic":   "🧴",
    "trash":     "🗑️",
}

DISPOSAL_INFO = {
    "cardboard": {
        "bin": "Blue Bin ♻️",
        "recyclable": True,
        "tip": "Flatten boxes and remove any tape or staples. Keep it dry!",
        "decompose": "2–3 months",
        "co2_saved": 0.21,
        "water_saved": 26,
        "energy_saved": 0.5,
        "steps": [
            "Flatten all cardboard boxes",
            "Remove tape, staples, and labels",
            "Keep it dry — wet cardboard goes to trash",
            "Place in the blue recycling bin",
        ],
        "fact": "♻️ Recycling 1 ton of cardboard saves 46 gallons of oil!",
    },
    "glass": {
        "bin": "Green Bin ♻️",
        "recyclable": True,
        "tip": "Rinse the container and remove caps. Do NOT break the glass.",
        "decompose": "1 million years",
        "co2_saved": 0.31,
        "water_saved": 1.5,
        "energy_saved": 0.27,
        "steps": [
            "Rinse the glass container thoroughly",
            "Remove caps and lids (recycle separately)",
            "Do NOT break the glass",
            "Place in the glass recycling bin",
        ],
        "fact": "♻️ Glass can be recycled INFINITELY without losing quality!",
    },
    "metal": {
        "bin": "Yellow Bin ♻️",
        "recyclable": True,
        "tip": "Rinse cans and crush them to save space.",
        "decompose": "200–500 years",
        "co2_saved": 0.90,
        "water_saved": 7.6,
        "energy_saved": 3.7,
        "steps": [
            "Rinse food residue from cans",
            "Crush cans to save space",
            "Remove paper labels if possible",
            "Place in the metal recycling bin",
        ],
        "fact": "♻️ Recycling aluminum saves 95% of the energy to make new aluminum!",
    },
    "paper": {
        "bin": "Blue Bin ♻️",
        "recyclable": True,
        "tip": "Keep paper clean and dry. Shred confidential documents first.",
        "decompose": "2–6 weeks",
        "co2_saved": 0.06,
        "water_saved": 10,
        "energy_saved": 0.2,
        "steps": [
            "Keep paper clean and dry",
            "Remove plastic windows from envelopes",
            "Shred confidential documents first",
            "Place in the paper recycling bin",
        ],
        "fact": "♻️ Paper can be recycled 5–7 times before fibres become too short!",
    },
    "plastic": {
        "bin": "Yellow Bin ♻️ (check type)",
        "recyclable": True,
        "tip": "Check the recycling number (1–7) on the bottom. Types 1 & 2 are most recyclable.",
        "decompose": "450–1000 years",
        "co2_saved": 0.15,
        "water_saved": 3.8,
        "energy_saved": 0.58,
        "steps": [
            "Check the recycling number (♻️ 1–7) on the bottom",
            "Rinse containers to remove food",
            "Remove caps (often a different plastic type)",
            "Types 1 (PET) and 2 (HDPE) are most recyclable",
        ],
        "fact": "⚠️ Only 9% of all plastic ever produced has been recycled!",
    },
    "trash": {
        "bin": "Black Bin 🗑️",
        "recyclable": False,
        "tip": "This item cannot be recycled. Place in the general waste bin.",
        "decompose": "Varies (very long)",
        "co2_saved": 0,
        "water_saved": 0,
        "energy_saved": 0,
        "steps": [
            "This item cannot be recycled",
            "Place in the general waste / black bin",
            "Check if any component can be separated for recycling",
            "Try to reduce use of non-recyclable items",
        ],
        "fact": "🌍 The average person generates 4.4 lbs of trash per day!",
    },
}

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ── */
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    .main { background-color: #f0f4f0; }

    /* ── Header banner ── */
    .header-banner {
        background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 50%, #40916c 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .header-banner h1 { font-size: 2.6rem; margin: 0; letter-spacing: -0.5px; }
    .header-banner p  { font-size: 1.1rem; opacity: 0.9; margin: 0.4rem 0 0; }

    /* ── Prediction card ── */
    .pred-card {
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.10);
        margin-bottom: 1rem;
    }
    .pred-label  { font-size: 2.2rem; font-weight: 800; margin: 0.3rem 0; }
    .pred-icon   { font-size: 3.5rem; }
    .pred-conf   { font-size: 1.3rem; opacity: 0.85; }

    /* ── Info cards ── */
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .info-card h4 { margin: 0 0 0.4rem; font-size: 1rem; color: #555; text-transform: uppercase; letter-spacing: 0.5px; }
    .info-card p  { margin: 0; font-size: 1.1rem; font-weight: 600; color: #222; }

    /* ── Metric pill ── */
    .metric-pill {
        display: inline-block;
        background: #e9f5e9;
        color: #2d6a4f;
        border-radius: 30px;
        padding: 0.35rem 1rem;
        font-size: 0.95rem;
        font-weight: 600;
        margin: 0.25rem 0.2rem;
    }

    /* ── Steps list ── */
    .step-item {
        display: flex;
        align-items: flex-start;
        gap: 0.7rem;
        padding: 0.5rem 0;
        font-size: 1rem;
    }
    .step-num {
        background: #2d6a4f;
        color: white;
        border-radius: 50%;
        width: 26px; height: 26px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.85rem; font-weight: 700; flex-shrink: 0;
    }

    /* ── Rejected banner ── */
    .rejected-banner {
        background: #fff3cd;
        border-left: 5px solid #f0ad4e;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        color: #856404;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a472a 0%, #2d6a4f 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stMarkdown a { color: #a8d5ba !important; }

    /* ── Upload zone ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed #2d6a4f !important;
        border-radius: 12px !important;
        background: #f7fdf7 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #2d6a4f, #40916c);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.6rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.9; }

    /* ── Footer ── */
    .footer-note {
        text-align: center;
        color: #888;
        font-size: 0.85rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  MODEL LOADER  (cached so it loads once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model…")
def load_model():
    """
    Try several common save locations for the .h5 model file.
    Edit MODEL_PATHS to add your own path.
    """
    MODEL_PATHS = [
        "waste_model.keras",
        "model.keras",
        "model.h5",
        "waste_model.h5",
        "best_model.h5",
        "waste_best_model_MobileNetV2.h5",
        "waste_best_model_ResNet50.h5",
        "waste_best_model_VGG16.h5",
        "waste_best_model_InceptionV3.h5",
        "waste_finetuned_MobileNetV2.h5",
    ]
    last_error = None
    for path in MODEL_PATHS:
        if os.path.exists(path):
            try:
                # compile=False avoids deserializing optimizer/loss objects that often
                # break when training and serving environments use different versions.
                return keras.models.load_model(path, compile=False), path, None
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                continue
    return None, None, last_error


# ─────────────────────────────────────────────
#  INFERENCE HELPER
# ─────────────────────────────────────────────
def predict(model, pil_image):
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]) * 100, probs


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ♻️ WasteAI")
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown(
        "1. Upload a waste image  \n"
        "2. AI classifies it instantly  \n"
        "3. Follow the disposal guide  \n"
        "4. Help the planet 🌍"
    )
    st.markdown("---")
    st.markdown("### Classes detected")
    for name in CLASS_NAMES:
        st.markdown(f"{CLASS_ICONS[name]} **{name.capitalize()}**")
    st.markdown("---")
    st.markdown("### Model info")
    st.markdown(
        "- Framework: TensorFlow / Keras  \n"
        "- Technique: Transfer Learning  \n"
        "- Input size: 224 × 224  \n"
        "- Confidence threshold: 60%"
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.8rem;opacity:0.7;'>Built with Streamlit · "
        "BTech CSE Project</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="header-banner">
        <h1>♻️ WasteAI — Smart Waste Segregation</h1>
        <p>Upload an image of waste and AI will classify it instantly with disposal instructions</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load model
model, model_path, model_error = load_model()

if model is None:
    if model_error:
        st.error("**Model file found, but failed to load.**")
        st.warning("Model file was found, but failed to load due to version mismatch.")
        st.code(model_error)
    else:
        st.error(
            "**Model file not found.**  \n\n"
            "Place your downloaded `.h5` model file in the same folder as `app.py` "
            "and rename it to **`model.h5`** (or any name listed in `MODEL_PATHS` inside `app.py`).  \n\n"
            "Common names the app will auto-detect:  \n"
            "`model.h5` · `waste_model.h5` · `best_model.h5` · "
            "`waste_best_model_MobileNetV2.h5` · `waste_finetuned_MobileNetV2.h5`"
        )
    st.stop()
else:
    st.success(f"Model loaded: `{model_path}`", icon="✅")

st.markdown("---")

# ── Tabs ──────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Classify Waste", "📊 Model Insights", "🌍 Environmental Impact"])

# ═══════════════════════════════════════════════
#  TAB 1 — CLASSIFY
# ═══════════════════════════════════════════════
with tab1:
    upload_col, result_col = st.columns([1, 1.3], gap="large")

    with upload_col:
        st.markdown("### 📤 Upload Waste Image")
        uploaded = st.file_uploader(
            "Drag & drop or click to browse",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded image", width="stretch")
            classify_btn = st.button("🔍 Classify Now", width="stretch")
        else:
            st.info("Upload a JPG / PNG image of waste to get started.")
            classify_btn = False

    with result_col:
        st.markdown("### 🤖 AI Prediction")

        if uploaded and classify_btn:
            with st.spinner("Analysing…"):
                label, confidence, all_probs = predict(model, img)

            info = DISPOSAL_INFO[label]
            color = CLASS_COLORS[label]
            icon = CLASS_ICONS[label]
            is_accepted = confidence >= CONFIDENCE_THRESHOLD * 100

            # ── Prediction card ──────────────────────
            bg_light = color + "22"
            st.markdown(
                f"""
                <div class="pred-card" style="background:{bg_light}; border: 2px solid {color};">
                    <div class="pred-icon">{icon}</div>
                    <div class="pred-label" style="color:{color};">{label.upper()}</div>
                    <div class="pred-conf">Confidence: <strong>{confidence:.1f}%</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if not is_accepted:
                st.markdown(
                    """
                    <div class="rejected-banner">
                        ⚠️ <strong>Low confidence prediction.</strong>
                        The model is uncertain — try a clearer, well-lit photo with the waste centred.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ── Confidence bar chart ─────────────────
            sorted_idx = np.argsort(all_probs)[::-1]
            fig = go.Figure(go.Bar(
                x=[all_probs[i] * 100 for i in sorted_idx],
                y=[CLASS_NAMES[i].capitalize() for i in sorted_idx],
                orientation="h",
                marker_color=[CLASS_COLORS[CLASS_NAMES[i]] for i in sorted_idx],
                text=[f"{all_probs[i]*100:.1f}%" for i in sorted_idx],
                textposition="outside",
            ))
            fig.add_vline(x=60, line_dash="dash", line_color="red",
                          annotation_text="60% threshold", annotation_position="top left")
            fig.update_layout(
                margin=dict(l=0, r=20, t=10, b=10),
                height=240,
                xaxis=dict(range=[0, 115], showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=13),
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

            # ── Disposal info ────────────────────────
            st.markdown("---")
            st.markdown("### 🗑️ Disposal Guide")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(
                    f'<div class="info-card"><h4>Bin</h4><p>{info["bin"]}</p></div>',
                    unsafe_allow_html=True,
                )
            with col_b:
                recyclable_text = "✅ Yes" if info["recyclable"] else "❌ No"
                st.markdown(
                    f'<div class="info-card"><h4>Recyclable</h4><p>{recyclable_text}</p></div>',
                    unsafe_allow_html=True,
                )
            with col_c:
                st.markdown(
                    f'<div class="info-card"><h4>Decomposes in</h4><p>{info["decompose"]}</p></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("**Steps to dispose properly:**")
            steps_html = ""
            for i, step in enumerate(info["steps"], 1):
                steps_html += (
                    f'<div class="step-item">'
                    f'<div class="step-num">{i}</div>'
                    f'<div>{step}</div>'
                    f'</div>'
                )
            st.markdown(steps_html, unsafe_allow_html=True)

            st.markdown(
                f'<div class="metric-pill">{info["fact"]}</div>',
                unsafe_allow_html=True,
            )

        elif not uploaded:
            st.markdown(
                '<div style="color:#aaa; text-align:center; padding:4rem 0; font-size:1.1rem;">'
                "Upload an image on the left to see results here."
                "</div>",
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════
#  TAB 2 — MODEL INSIGHTS
# ═══════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Model Architecture & Comparison")

    arch_data = {
        "Model": ["MobileNetV2", "ResNet50", "VGG16", "InceptionV3"],
        "Year": [2018, 2015, 2014, 2015],
        "Parameters (M)": [3.4, 25.6, 138.0, 23.8],
        "Size (MB)": [14, 98, 528, 92],
        "Key Innovation": [
            "Depthwise separable convolutions",
            "Residual / skip connections",
            "Uniform 3×3 conv stacking",
            "Parallel multi-scale convolutions",
        ],
    }

    fig2 = px.bar(
        arch_data,
        x="Model",
        y="Parameters (M)",
        color="Model",
        color_discrete_sequence=["#4ECDC4", "#FF6B6B", "#FFE66D", "#A29BFE"],
        title="Number of Parameters per Model",
        text="Parameters (M)",
    )
    fig2.update_traces(texttemplate="%{text}M", textposition="outside")
    fig2.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10),
        height=320,
    )
    st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

    st.markdown("#### Custom Classification Head")
    st.code(
        """BaseModel (frozen ImageNet weights)
    → GlobalAveragePooling2D
    → BatchNormalization
    → Dense(256, activation='relu')
    → Dropout(0.5)
    → Dense(128, activation='relu')
    → Dropout(0.3)
    → Dense(6, activation='softmax')   ← 6 waste classes""",
        language="text",
    )

    st.markdown("#### Training Configuration")
    cfg_col1, cfg_col2 = st.columns(2)
    cfg = {
        "Image Size": "224 × 224",
        "Batch Size": "32",
        "Base Epochs": "10",
        "Fine-tune Epochs": "15",
        "Optimizer": "Adam (lr = 0.001)",
        "Fine-tune LR": "0.0001 (10× smaller)",
        "Loss": "Categorical Cross-Entropy",
        "Train / Val Split": "80% / 20%",
    }
    items = list(cfg.items())
    for k, v in items[:4]:
        cfg_col1.metric(k, v)
    for k, v in items[4:]:
        cfg_col2.metric(k, v)

# ═══════════════════════════════════════════════
#  TAB 3 — ENVIRONMENTAL IMPACT
# ═══════════════════════════════════════════════
with tab3:
    st.markdown("### 🌍 Environmental Impact of Recycling")
    st.markdown(
        "See how much CO₂, water, and energy you save by recycling each waste type correctly."
    )

    classes_for_chart = [c for c in CLASS_NAMES if c != "trash"]
    co2   = [DISPOSAL_INFO[c]["co2_saved"] for c in classes_for_chart]
    water = [DISPOSAL_INFO[c]["water_saved"] for c in classes_for_chart]
    enrg  = [DISPOSAL_INFO[c]["energy_saved"] for c in classes_for_chart]

    env_col1, env_col2, env_col3 = st.columns(3)

    with env_col1:
        fig_co2 = go.Figure(go.Bar(
            x=classes_for_chart,
            y=co2,
            marker_color=[CLASS_COLORS[c] for c in classes_for_chart],
            text=[f"{v} kg" for v in co2],
            textposition="outside",
        ))
        fig_co2.update_layout(
            title="CO₂ Saved per item (kg)",
            height=300,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_co2, width="stretch", config={"displayModeBar": False})

    with env_col2:
        fig_water = go.Figure(go.Bar(
            x=classes_for_chart,
            y=water,
            marker_color=[CLASS_COLORS[c] for c in classes_for_chart],
            text=[f"{v} L" for v in water],
            textposition="outside",
        ))
        fig_water.update_layout(
            title="Water Saved per item (L)",
            height=300,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_water, width="stretch", config={"displayModeBar": False})

    with env_col3:
        fig_enrg = go.Figure(go.Bar(
            x=classes_for_chart,
            y=enrg,
            marker_color=[CLASS_COLORS[c] for c in classes_for_chart],
            text=[f"{v} kWh" for v in enrg],
            textposition="outside",
        ))
        fig_enrg.update_layout(
            title="Energy Saved per item (kWh)",
            height=300,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_enrg, width="stretch", config={"displayModeBar": False})

    st.markdown("---")
    st.markdown("#### Decomposition Timeline")
    decomp = {
        "Cardboard": 0.25,
        "Paper":     0.08,
        "Glass":     1_000_000,
        "Metal":     350,
        "Plastic":   700,
        "Trash":     500,
    }
    colors_d = ["#96E6A1", "#96E6A1", "#FF6B6B", "#FFE66D", "#A29BFE", "#B2BEC3"]

    fig_d = go.Figure(go.Bar(
        y=list(decomp.keys()),
        x=list(decomp.values()),
        orientation="h",
        marker_color=colors_d,
        text=["~3 months", "~1 month", "1 million yrs", "~350 yrs", "~700 yrs", "~500 yrs"],
        textposition="outside",
    ))
    fig_d.update_layout(
        xaxis_type="log",
        xaxis_title="Years (log scale)",
        height=280,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, r=120),
        showlegend=False,
    )
    st.plotly_chart(fig_d, width="stretch", config={"displayModeBar": False})

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown(
    '<div class="footer-note">WasteAI · Built with TensorFlow & Streamlit · '
    'BTech CSE Deep Learning Project · 2026</div>',
    unsafe_allow_html=True,
)
