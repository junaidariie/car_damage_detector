import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Car Damage Detector",
    page_icon="🚗",
    layout="centered"
)

# Custom CSS for clean, compact styling
st.markdown("""
    <style>
    .main {
        max-width: 800px;
        padding: 1rem;
        margin: 0 auto;
    }
    .block-container {
        max-width: 800px;
        padding-left: 2rem;
        padding-right: 2rem;
        margin: 0 auto;
    }
    h1 {
        color: #2c3e50;
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.3rem !important;
    }
    .subtitle {
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 8px;
        border: 2px dashed #d5d8dc;
        text-align: center;
        margin: 1.5rem 0;
    }
    .result-card {
        background: #fff3cd;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
        border-left: 4px solid #f39c12;
        text-align: center;
    }
    .result-label {
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .result-value {
        color: #f39c12;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .damage-detected {
        background: #f8d7da;
        border-left: 4px solid #e74c3c;
    }
    .damage-value {
        color: #e74c3c;
    }
    .no-damage {
        background: #d4edda;
        border-left: 4px solid #27ae60;
    }
    .no-damage-value {
        color: #27ae60;
    }
    .info-box {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        border-left: 3px solid #3498db;
    }
    .stButton>button {
        width: 100%;
        background: #f39c12;
        color: white;
        font-weight: 600;
        padding: 0.6rem;
        border: none;
        border-radius: 6px;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background: #e67e22;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🚗 Car Damage Detector")
st.markdown('<p class="subtitle">AI-powered vehicle damage assessment</p>', unsafe_allow_html=True)

# Info box
st.markdown("""
<div class="info-box">
    <strong>📋 How it works:</strong><br>
    Upload a clear image of your car and our AI will detect and classify any damage.
</div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "Upload a car image",
    type=["png", "jpg", "jpeg"],
    help="Supported formats: PNG, JPG, JPEG"
)

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Analyze button
    if st.button("🔍 Analyze Damage"):
        API_URL = st.secrets["API_URL"]
        
        with st.spinner("Analyzing image..."):
            try:
                # Convert image to base64 for API transmission
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Prepare payload
                payload = {
                    "image": img_str
                }
                
                # Make API request
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get('prediction', 'Unknown')
                    
                    # Determine card style based on prediction
                    is_damage = 'Normal' not in prediction
                    card_class = 'damage-detected' if is_damage else 'no-damage'
                    value_class = 'damage-value' if is_damage else 'no-damage-value'
                    
                    # Display result
                    st.markdown(f"""
                    <div class="result-card {card_class}">
                        <div class="result-label">Detection Result</div>
                        <div class="result-value {value_class}">{prediction}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional information based on result
                    if is_damage:
                        st.warning("⚠️ Damage detected! Please consult a professional for assessment.")
                    else:
                        st.success("✅ No damage detected. Vehicle appears to be in good condition.")
                    
                else:
                    st.error(f"❌ API Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")

else:
    # Show placeholder when no image is uploaded
    st.markdown("""
    <div class="upload-section">
        <h3 style="color: #7f8c8d;">👆 Upload an image to get started</h3>
        <p style="color: #95a5a6;">Supported damage types:</p>
        <p style="color: #7f8c8d; font-size: 0.9rem;">
            Front Breakage • Front Crushed • Rear Breakage • Rear Crushed • Normal Condition
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #999; padding: 1rem 0; font-size: 0.85rem;">
    <p>🚗 Powered by Advanced Computer Vision AI</p>
</div>
""", unsafe_allow_html=True)
