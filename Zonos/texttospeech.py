import streamlit as st
import torch
import torchaudio
import sys
import os
from pathlib import Path
import tempfile
import logging
from phonemizer.backend.espeak.wrapper import EspeakWrapper

# Suppress Torch inductor errors and fall back to eager mode
torch._dynamo.config.suppress_errors = True  # Added to avoid cl.exe dependency

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Adjust path to import Zonos from your cloned repo
ZONOS_PATH = Path(r"C:\Harsha\Work\perigo\try5\Zonos")
if ZONOS_PATH.exists():
    sys.path.append(str(ZONOS_PATH))
else:
    st.error(f"Zonos path {ZONOS_PATH} does not exist. Please check the directory.")
    st.stop()

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

# Set eSpeak environment variable and configure EspeakWrapper
ESPEAK_PATH = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
ESPEAK_LIB = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
if os.path.exists(ESPEAK_PATH):
    os.environ["PHONEMIZER_ESPEAK_PATH"] = ESPEAK_PATH
    if os.path.exists(ESPEAK_LIB):
        EspeakWrapper.set_library(ESPEAK_LIB)
    else:
        st.warning(f"eSpeak-NG library not found at {ESPEAK_LIB}. Falling back to default.")
else:
    st.error(f"eSpeak-NG executable not found at {ESPEAK_PATH}. Please install it or update the path.")
    st.stop()

# Load model from local repo with manual progress feedback
@st.cache_resource
def load_model():
    with st.spinner("⏳ Loading Zonos model..."):
        try:
            model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            logger.error(f"Model loading error: {e}", exc_info=True)
            raise

# Initialize model early using session state
if "model" not in st.session_state:
    st.session_state.model = load_model()
model = st.session_state.model

# Streamlit UI
st.title("🗣️ Zonos Text-to-Speech (TTS)")
st.markdown("""
    Generate high-quality speech from text using the Zonos model from your local repository.
    Powered by eSpeak-NG for phoneme generation.
""")

# Text input with placeholder and validation
text = st.text_area(
    "Enter text to synthesize:",
    value="Hello, world!",
    placeholder="Type your text here...",
    height=100,
)
if not text.strip():
    st.warning("Please enter some text to synthesize.")

# File uploader for speaker embedding with format validation
uploaded_file = st.file_uploader(
    "Upload an audio file for speaker embedding:",
    type=["wav", "mp3"],
    help="Upload a WAV or MP3 file (max 10 seconds recommended for faster processing)."
)

# Language selection with expanded options
LANGUAGES = {
    "English (US)": "en-us",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
}
language = st.selectbox("Select language:", list(LANGUAGES.keys()), index=0)
language_code = LANGUAGES[language]

# Generate speech button with progress bar
if st.button("Generate Speech", disabled=not (text.strip() and uploaded_file)):
    try:
        with st.spinner("Processing..."):
            # Use temporary file for audio upload
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                temp_audio.write(uploaded_file.read())
                temp_audio_path = temp_audio.name

            # Load audio file
            st.write("🔊 Loading audio file...")
            wav, sampling_rate = torchaudio.load(temp_audio_path)
            if wav.shape[-1] / sampling_rate > 10:
                st.warning("Audio is longer than 10 seconds; processing may be slow.")

            # Generate speaker embedding
            st.write("🔄 Extracting speaker embedding...")
            speaker = model.make_speaker_embedding(wav, sampling_rate)

            # Prepare conditioning
            st.write("🛠️ Preparing conditioning...")
            cond_dict = make_cond_dict(text=text, speaker=speaker, language=language_code)
            conditioning = model.prepare_conditioning(cond_dict)

            # Generate speech codes
            st.write("🎙️ Generating speech...")
            codes = model.generate(conditioning)

            # Decode audio with error handling
            st.write("🎵 Decoding audio...")
            wavs = model.autoencoder.decode(codes).cpu()
            output_file = "generated_speech.wav"
            torchaudio.save(output_file, wavs[0], model.autoencoder.sampling_rate)

            # Clean up temporary file
            os.unlink(temp_audio_path)

            # Play audio
            st.audio(output_file, format="audio/wav")
            st.success("✅ Speech synthesis complete!")

            # Download button
            with open(output_file, "rb") as file:
                st.download_button(
                    label="⬇️ Download Audio",
                    data=file,
                    file_name="generated_speech.wav",
                    mime="audio/wav",
                )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Synthesis error: {e}", exc_info=True)

# Footer with debug info (optional)
if st.checkbox("Show debug info"):
    st.write(f"Device: {device}")
    st.write(f"eSpeak path: {os.environ.get('PHONEMIZER_ESPEAK_PATH')}")
    st.write(f"eSpeak library: {ESPEAK_LIB}")
    st.write(f"Zonos path: {ZONOS_PATH}")