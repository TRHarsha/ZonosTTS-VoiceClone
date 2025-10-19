import json
import streamlit as st
from deepgram import Deepgram, DeepgramClient, LiveTranscriptionEvents, LiveOptions, DeepgramClientOptions
import os
import logging
from dotenv import load_dotenv
import socketio

# Load environment variables
load_dotenv()

API_KEY = os.getenv("DEEPGRAM_API_KEY")

deepgram = Deepgram(API_KEY)

dg_connection = None

st.title("Audio Transcription with Deepgram")

# File upload
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
url = st.text_input("Or provide a Cloudinary URL")

# Feature selection
features = st.text_area("Deepgram Features (JSON format)", "{}")
model = st.selectbox("Select Model", ["nova", "whisper"], index=0)
version = st.text_input("Version (Optional)")
tier = st.selectbox("Tier", ["base", "enhanced"], index=0) if model == "whisper" else None

# Set up client configuration
config = DeepgramClientOptions(
    verbose=logging.WARN,  # Change to logging.INFO or logging.DEBUG for more verbose output
    options={"keepalive": "true"}
)

deepgram_client = DeepgramClient(API_KEY, config)

def initialize_deepgram_connection():
    global dg_connection
    dg_connection = deepgram_client.listen.live.v("1")

    def on_open(self, open, **kwargs):
        st.write(f"\n\n{open}\n\n")

    def on_message(self, result, **kwargs):
        transcript = result.channel.alternatives[0].transcript
        if len(transcript) > 0:
            st.write(result.channel.alternatives[0].transcript)
            
    def on_close(self, close, **kwargs):
        st.write(f"\n\n{close}\n\n")

    def on_error(self, error, **kwargs):
        st.write(f"\n\n{error}\n\n")

    dg_connection.on(LiveTranscriptionEvents.Open, on_open)
    dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
    dg_connection.on(LiveTranscriptionEvents.Close, on_close)
    dg_connection.on(LiveTranscriptionEvents.Error, on_error)

    options = LiveOptions(model="nova-3", language="en-US")

    if dg_connection.start(options) is False:
        st.error("Failed to start connection")
        return

if st.button("Start Live Transcription"):
    initialize_deepgram_connection()

if st.button("Transcribe File"):
    try:
        dgRequest = None
        
        if url and url.startswith("https://res.cloudinary.com/deepgram"):
            dgRequest = {"url": url}
        
        if uploaded_file is not None:
            dgRequest = {"mimetype": uploaded_file.type, "buffer": uploaded_file.read()}
        
        dgFeatures = json.loads(features)
        dgFeatures["model"] = model
        
        if version:
            dgFeatures["version"] = version
        
        if model == "whisper" and tier:
            dgFeatures["tier"] = tier
        
        if not dgRequest:
            st.error("You need to choose a file or provide a valid URL.")
        else:
            transcription = deepgram.transcription.prerecorded(dgRequest, dgFeatures)
            st.json(transcription)
    except Exception as error:
        st.error(f"Error: {str(error)}")
