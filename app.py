
import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = "http://127.0.0.1:8000/predict"
API_KEY = os.getenv("API_KEY", "")

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Speech Emotion Recognition")
st.write(
    "Upload a WAV audio recording to predict the emotion "
    "detected by the trained AI model."
)

st.info(
    "Supported emotions: neutral, calm, happy, sad, angry, fearful."
)

audio = st.file_uploader(
    "Choose an audio file",
    type=["wav"]
)

if st.button("Analyze Emotion", type="primary"):
    if not audio:
        st.warning("Please upload a WAV audio file first.")
    elif not API_KEY:
        st.error("API key is missing. Check your .env file.")
    else:
        with st.spinner("Analyzing audio..."):
            try:
                response = requests.post(
                    API_URL,
                    headers={"x-api-key": API_KEY},
                    files={
                        "file": (
                            audio.name,
                            audio.getvalue(),
                            "audio/wav"
                        )
                    },
                    timeout=120
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success("Analysis complete!")
                    st.subheader(
                        f"Predicted emotion: {result['emotion'].title()}"
                    )
                    if "confidence" in result:
                        confidence = float(result["confidence"])
                        st.metric(
                            "Model confidence",
                            f"{confidence:.1%}"
                        )
                        st.progress(
                            max(0.0, min(1.0, confidence))
                        )
                    st.caption(
                        "Confidence is the model's score, "
                        "not a guarantee of real-world accuracy."
                    )
                else:
                    st.error(
                        f"API error ({response.status_code}): "
                        f"{response.text}"
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to the API. Start api.py's "
                    "Uvicorn server in another terminal."
                )
            except requests.exceptions.RequestException as error:
                st.error(f"Request failed: {error}")
