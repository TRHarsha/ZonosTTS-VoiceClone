import streamlit as st
from deepgram import DeepgramClient, PrerecordedOptions
import json
import httpx
import os

def transcribe_audio(file_path, api_key):
    try:
        deepgram = DeepgramClient(api_key)
        options = PrerecordedOptions(
            model="nova-2",
            language="hi",
            smart_format=True,
            punctuate=True,
            diarize=True,
            sentiment=True,
        )
        with open(file_path, "rb") as audio_file:
            source = {"buffer": audio_file.read()}
            
        # Increase the timeout to 300 seconds (5 minutes)
        response = deepgram.listen.prerecorded.v("1").transcribe_file(
            source, options, timeout=(10.0, 300.0)  # (connect timeout, read timeout)
        )
        json_response = response.to_json(indent=4)
        print("Full API Response:", json_response)  # Debugging: Print the full response
        
        # Check the duration field in the response
        if "metadata" in json.loads(json_response):
            duration = json.loads(json_response)["metadata"]["duration"]
            print(f"Processed Duration: {duration} seconds")
        
        return json_response
    
    except Exception as e:
        return f"Exception: {e}"

def extract_transcript(transcription_json):
    """Extract the transcript text from the Deepgram JSON response."""
    if not transcription_json or transcription_json.strip() == "":
        return "Error: Empty response from Deepgram API."
    
    try:
        data = json.loads(transcription_json)
        if "results" in data and "channels" in data["results"]:
            words = data["results"]["channels"][0]["alternatives"][0]["words"]
            transcript = " ".join(word["word"] for word in words)
            return transcript.strip()
        else:
            return "Error: Invalid JSON structure in API response."
    except json.JSONDecodeError:
        return f"Error: Invalid JSON response. Raw response: {transcription_json}"
    except Exception as e:
        return f"Error extracting transcript: {e}"

def main():
    st.title("Deepgram Audio Transcription")
    
    api_key = "f552c865a1223d9848d86988bb63ffc6c01426fb"
    
    if not api_key:
        st.warning("Please enter your Deepgram API key to proceed.")
        return
    
    # Step 2: Upload Audio File
    st.subheader("Upload an Audio File")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])
    
    if uploaded_file is not None:
        # Validate file size (e.g., limit to 100MB)
        if uploaded_file.size > 100 * 1024 * 1024:  
            st.error("File size too large. Please upload a file smaller than 100MB.")
            return
        
        # Save the uploaded file to a temporary file
        with open("temp_audio", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Step 3: Transcribe the Audio File
        st.subheader("Transcribe the Audio File")
        if st.button("Transcribe"):
            with st.spinner("Transcribing audio..."):
                transcription_json = transcribe_audio("temp_audio", api_key)
                transcription_text = extract_transcript(transcription_json)
                
                # Display the transcription
                st.subheader("Transcription:")
                st.text_area("Transcribed Text", transcription_text, height=300)
                
                # Provide a download button for the transcription
                st.download_button(
                    label="Download Transcription",
                    data=transcription_text,
                    file_name="transcription.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()