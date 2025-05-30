import sounddevice as sd
import numpy as np
import whisper
import os
import tempfile
import wave
import openai

class AudioRecorder:
    """
    Handles audio recording functionality.
    """
    def __init__(self, sample_rate=16000, channels=1):
        """
        Initialize the AudioRecorder with recording parameters.
        
        Args:
            sample_rate (int): Audio sample rate in Hz
            channels (int): Number of audio channels
        """
        self.sample_rate = sample_rate
        self.channels = channels

    def record_audio(self, duration=5):
        """
        Record audio for a specified duration.
        
        Args:
            duration (int): Recording duration in seconds
            
        Returns:
            numpy.ndarray: Recorded audio data
        """
        print(f"Recording for {duration} seconds...")
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels
        )
        sd.wait()
        print("Recording finished!")
        return recording

class Transcriber:
    """
    Handles audio transcription using the Whisper model.
    """
    def __init__(self, model_name="base"):
        """
        Initialize the Transcriber with a Whisper model.
        
        Args:
            model_name (str): Name of the Whisper model to use
        """
        self.model = whisper.load_model(model_name)

    def transcribe_audio(self, audio_data, sample_rate=16000):
        """
        Transcribe audio data to text.
        
        Args:
            audio_data (numpy.ndarray): Audio data to transcribe
            sample_rate (int): Sample rate of the audio data
            
        Returns:
            str: Transcribed text
        """
        # Save audio data to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 2 bytes per sample
                wf.setframerate(sample_rate)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
            
            # Transcribe the audio file
            result = self.model.transcribe(temp_file.name)
            
            # Clean up the temporary file
            os.unlink(temp_file.name)
            
            return result["text"]

class ChatGPTProcessor:
    """
    Handles processing of transcribed text using the ChatGPT API.
    """
    def __init__(self, api_key):
        """
        Initialize the ChatGPTProcessor with an API key.
        
        Args:
            api_key (str): OpenAI API key
        """
        self.api_key = api_key
        openai.api_key = api_key

    def process_text(self, text):
        """
        Process text using the ChatGPT API.
        
        Args:
            text (str): Text to process
            
        Returns:
            str: ChatGPT's response
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error processing text with ChatGPT: {e}")
            return None

def main():
    """
    Main function to orchestrate the application flow.
    """
    # Initialize components
    recorder = AudioRecorder()
    transcriber = Transcriber()
    
    # Set your OpenAI API key here
    chatgpt = ChatGPTProcessor("your-api-key-here")
    
    # Record audio
    audio_data = recorder.record_audio()
    
    # Transcribe audio
    transcribed_text = transcriber.transcribe_audio(audio_data)
    print("\nTranscribed text:", transcribed_text)
    
    # Save transcription to file
    with open("transcripts.txt", "a") as f:
        f.write(transcribed_text + "\n")
    
    # Process with ChatGPT
    response = chatgpt.process_text(transcribed_text)
    if response:
        print("\nChatGPT response:", response)

if __name__ == "__main__":
    main() 