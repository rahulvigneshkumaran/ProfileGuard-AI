import streamlit as st
import numpy as np
from PIL import Image
from predict import predict_profile

st.set_page_config(
    page_title="ProfileGuard AI",
    page_icon="🕵️",
    layout="centered"
)

st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(59,130,246,0.22), transparent 28%),
            radial-gradient(circle at top right, rgba(168,85,247,0.18), transparent 25%),
            radial-gradient(circle at bottom left, rgba(16,185,129,0.14), transparent 22%),
            linear-gradient(135deg, #020617, #0f172a, #111827);
        color: #f8fafc;
    }

    .main-container {
        max-width: 900px;
        margin: auto;
        padding: 16px;
    }

    .glass-card {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.10);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 24px;
        padding: 28px;
        margin-bottom: 18px;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        font-size: 1.2rem;
        font-weight: 600;
        color: #bfdbfe;
        margin-bottom: 12px;
    }

    .hero-desc {
        font-size: 1rem;
        line-height: 1.75;
        color: #cbd5e1;
    }

    .badge-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 18px;
    }

    .badge {
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.08);
        color: #e2e8f0;
        font-size: 0.9rem;
    }

    .section-heading {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 10px;
    }

    .prediction-box {
        border-radius: 18px;
        padding: 16px 18px;
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 24px rgba(0,0,0,0.22);
    }

    .prediction-fake {
        background: linear-gradient(135deg, rgba(127,29,29,0.75), rgba(153,27,27,0.55));
        color: #fecaca;
    }

    .prediction-real {
        background: linear-gradient(135deg, rgba(20,83,45,0.75), rgba(22,101,52,0.55));
        color: #bbf7d0;
    }

    .metric-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 18px;
        text-align: center;
        min-height: 110px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.5rem;
        font-weight: 800;
    }

    .reason-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(96,165,250,0.16);
        border-left: 4px solid #60a5fa;
        border-radius: 14px;
        padding: 13px 15px;
        color: #e2e8f0;
        margin-bottom: 10px;
        box-shadow: 0 6px 14px rgba(0,0,0,0.15);
    }

    .footer-text {
        text-align: center;
        color: #94a3b8;
        font-size: 0.92rem;
        margin-top: 14px;
        margin-bottom: 12px;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.05);
        border: 1px dashed rgba(96,165,250,0.45);
        border-radius: 18px;
        padding: 12px;
    }

    div[data-testid="stTextArea"] textarea,
    div[data-testid="stNumberInput"] input {
        background: rgba(15, 23, 42, 0.78) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(148,163,184,0.22) !important;
        border-radius: 14px !important;
    }

    div[data-testid="stTextArea"] textarea::placeholder,
    div[data-testid="stNumberInput"] input::placeholder {
        color: #64748b !important;
    }

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        font-size: 1rem;
        font-weight: 800;
        color: white;
        background: linear-gradient(135deg, rgba(37,99,235,0.95), rgba(147,51,234,0.9));
        box-shadow: 0 10px 28px rgba(59,130,246,0.28);
        transition: 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        background: linear-gradient(135deg, rgba(29,78,216,1), rgba(126,34,206,0.95));
        color: white;
    }

    label, .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
    }

    div[data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
    <div class="hero-title">🕵️ ProfileGuard AI</div>
    <div class="hero-subtitle">Premium Multimodal Fake Profile Detection System</div>
    <div class="hero-desc">
        Analyze a social media profile using image patterns, bio text, and account activity.
        This system combines computer vision, NLP, and structured data analysis to estimate
        whether the account appears genuine or suspicious.
    </div>
    <div class="badge-row">
        <div class="badge">CNN Image Analysis</div>
        <div class="badge">LSTM Bio Understanding</div>
        <div class="badge">Behavior Pattern Detection</div>
        <div class="badge">Explainable Prediction</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Profile Inputs</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Profile Image",
    type=["jpg", "jpeg", "png", "webp"]
)

bio_text = st.text_area("Bio Text", placeholder="Enter profile bio here...")

col1, col2 = st.columns(2)
with col1:
    followers = st.number_input("Followers", min_value=0, step=1)
    posts = st.number_input("Posts", min_value=0, step=1)
    engagement_rate = st.number_input(
        "Engagement Rate", min_value=0.0, step=0.0001, format="%.4f"
    )

with col2:
    following = st.number_input("Following", min_value=0, step=1)
    account_age = st.number_input("Account Age (days)", min_value=0, step=1)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Profile Image", use_container_width=True)

check = st.button("Analyze Profile")
st.markdown('</div>', unsafe_allow_html=True)

if check:
    if uploaded_file is None:
        st.error("Please upload a profile image.")
    elif bio_text.strip() == "":
        st.error("Please enter bio text.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)

        label, confidence, raw_score, reasons = predict_profile(
            image_array=image_array,
            bio_text=bio_text,
            followers=followers,
            following=following,
            posts=posts,
            account_age=account_age,
            engagement_rate=engagement_rate
        )

        fake_probability = raw_score * 100
        pred_class = "prediction-fake" if label.lower() == "fake" else "prediction-real"

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Analysis Result</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="prediction-box {pred_class}">Prediction: {label}</div>',
            unsafe_allow_html=True
        )

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{confidence:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Fake Probability</div>
                    <div class="metric-value">{fake_probability:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("### Probability Meter")
        st.progress(min(max(int(fake_probability), 0), 100))

        st.markdown("### Reasons")
        for reason in reasons:
            st.markdown(f'<div class="reason-card">• {reason}</div>', unsafe_allow_html=True)

        st.markdown("### Input Summary")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Followers</div>
                    <div class="metric-value">{followers}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Posts</div>
                    <div class="metric-value">{posts}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Engagement Rate</div>
                    <div class="metric-value">{engagement_rate:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with s2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Following</div>
                    <div class="metric-value">{following}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Account Age</div>
                    <div class="metric-value">{account_age} days</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer-text">Built with Streamlit, TensorFlow, CNN, LSTM, and structured profile features.</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)