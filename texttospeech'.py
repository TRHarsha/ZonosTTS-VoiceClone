import streamlit as st
import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

# Load the Zonos model
@st.cache_resource
def load_model():
    return Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

model = load_model()

# Streamlit UI
st.title("🗣️ Zonos TTS - Text-to-Speech Generator")

st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload reference audio (.wav, .mp3)", type=["wav", "mp3"])
text_input = st.text_area("Enter text to synthesize", "Hello, world!")
language = st.selectbox("Select Language", ["en-us", "es-es", "fr-fr"], index=0)

if uploaded_file is not None:
    # Load uploaded audio
    wav, sampling_rate = torchaudio.load(uploaded_file)
    st.sidebar.audio(uploaded_file, format="audio/mp3")

    # Generate speaker embedding
    speaker = model.make_speaker_embedding(wav, sampling_rate)

    if st.button("Generate Speech"):
        with st.spinner("Generating speech..."):
            # Prepare conditioning
            cond_dict = make_cond_dict(text=text_input, speaker=speaker, language=language)
            conditioning = model.prepare_conditioning(cond_dict)

            # Generate audio
            codes = model.generate(conditioning)
            wavs = model.autoencoder.decode(codes).cpu()
            
            # Save output
            output_path = "output.wav"
            torchaudio.save(output_path, wavs[0], model.autoencoder.sampling_rate)

            # Display audio output
            st.audio(output_path, format="audio/wav")

            # Provide download link
            with open(output_path, "rb") as f:
                st.download_button("Download Audio", f, file_name="generated_speech.wav", mime="audio/wav")

