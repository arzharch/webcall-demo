import os
from google.cloud import texttospeech

# Point to your downloaded credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\arshc\\Desktop\\demo\\backend\\synthion-demo-call-80b547a5bbbd.json"



def synthesize_text(text: str, output_file: str = "output.mp3"):
    client = texttospeech.TextToSpeechClient()

    # Input text
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Voice selection 
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Neural2-F"
    )

    # Audio configuration
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        pitch=0.0,
        speaking_rate=1.0
    )

    response = client.synthesize_speech(
        input=synthesis_input, 
        voice=voice, 
        audio_config=audio_config
    )

    with open(output_file, "wb") as out:
        out.write(response.audio_content)
        print(f'Audio content written to "{output_file}"')

# Usage with your LLM response
llm_response = "Hello Arsh, I've processed your request."
synthesize_text(llm_response)