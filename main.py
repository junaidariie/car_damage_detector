import streamlit as st
import requests

# ✅ Your FastAPI endpoint on Render
API_URL = "https://damage-detection-ojmo.onrender.com/predict"

st.title("🛠️ Damage Detection App")
st.write("Upload an image to detect damage using the deployed API.")
st.markdown(
    '<p class="subtitle" style="color:blue; font-weight:bold;">The app could take up to <b>1 minute</b> when using for the first time due to API latency.</p>',
    unsafe_allow_html=True
)


# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect Damage"):
        with st.spinner("Analyzing image..."):
            # Send file to API
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                try:
                    result = response.json()
                    st.success("✅ Damage detection completed!")
                    st.json(result)  # Display full API response as JSON
                except Exception:
                    st.error("⚠️ Could not parse JSON response.")
            else:
                st.error(f"❌ API returned status code {response.status_code}")
                st.text(response.text)
