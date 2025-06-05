import sounddevice as sd
import numpy as np
import whisper
import os
import tempfile
import wave
import openai

class AudioRecorder:
    """
    A class to handle audio recording functionality.
    """

    def __init__(self, duration=5, sample_rate=44100):
        """
        Initialize the AudioRecorder with specified duration and sample rate.

        :param duration: The duration of the recording in seconds.
        :param sample_rate: The sample rate of the audio recording.
        """
        self.duration = duration
        self.sample_rate = sample_rate

    def record(self):
        """
        Record audio for the specified duration.

        :return: The recorded audio data as a numpy array.
        """
        print(f"Recording for {self.duration} seconds...")
        audio_data = sd.rec(int(self.duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype=np.float32)
        sd.wait()
        return audio_data

class Transcriber:
    """
    A class to handle audio transcription using the Whisper model.
    """

    def __init__(self, model_name="base"):
        """
        Initialize the Transcriber with a specified Whisper model.

        :param model_name: The name of the Whisper model to use for transcription.
        """
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_data):
        """
        Transcribe the provided audio data using the Whisper model.

        :param audio_data: The audio data to transcribe.
        :return: The transcribed text.
        """
        print("Transcribing audio...")
        audio_data = (audio_data * 32767).astype(np.int16)

        # Save audio data to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 2 bytes per sample
                wf.setframerate(44100)  # Sample rate
                wf.writeframes(audio_data.tobytes())

        # Transcribe the audio file
        result = self.model.transcribe(temp_file.name)
        os.unlink(temp_file.name)  # Delete the temporary file
        print("Transcription complete.")
        return result["text"]

class ChatGPTProcessor:
    """
    A class to handle interactions with the ChatGPT API.
    """

    def __init__(self, api_key):
        """
        Initialize the ChatGPTProcessor with an API key.

        :param api_key: The OpenAI API key for authentication.
        """
        openai.api_key = api_key

    def process(self, transcript):
        """
        Process the transcript using the ChatGPT API.

        :param transcript: The text to be processed by ChatGPT.
        :return: The response from ChatGPT.
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": transcript}
            ]
        )
        return response.choices[0].message['content']

def main():
    """
    Main function to orchestrate the recording, transcription, and processing of audio.
    """
    # Initialize components
    recorder = AudioRecorder()
    transcriber = Transcriber()
    chatgpt_processor = ChatGPTProcessor('your-api-key')

    # Record audio
    input("Press Enter to start recording...")
    audio_data = recorder.record()

    # Transcribe audio
    transcribed_text = transcriber.transcribe(audio_data)
    print("Transcribed text:", transcribed_text)

    # Save the transcribed text to a file
    with open("transcripts.txt", "a") as file:
        file.write(transcribed_text + "\n")
    print("Transcription saved to transcripts.txt")

    # Process with ChatGPT
    with open("transcripts.txt", "r") as file:
        transcript = file.read()
    chatgpt_response = chatgpt_processor.process(transcript)
    print("ChatGPT Response:", chatgpt_response)

if __name__ == "__main__":
    main() 