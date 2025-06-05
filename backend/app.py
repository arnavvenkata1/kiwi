# import sounddevice as sd
# import numpy as np
# import whisper
# import os
# import tempfile
# import wave
# from openai import OpenAI
# from dotenv import load_dotenv
# from TTS.api import TTS
# import asyncio
# import queue
# import threading
# import time
# import webrtcvad

# # Load environment variables
# load_dotenv()

# class AudioRecorder:
#     """
#     Handles audio recording functionality with real-time streaming and VAD.
#     """
#     def __init__(self, sample_rate=16000, channels=1, chunk_size=160, vad_mode=2):
#         """
#         Initialize the AudioRecorder with recording parameters.
        
#         Args:
#             sample_rate (int): Audio sample rate in Hz
#             channels (int): Number of audio channels
#             chunk_size (int): Size of audio chunks to capture
#             vad_mode (int): VAD mode to use (default: 2)
#         """
#         self.sample_rate = sample_rate
#         self.channels = channels
#         self.chunk_size = chunk_size  # 10ms at 16kHz = 160 samples
#         self.recording = False
#         self.audio_queue = queue.Queue()
#         self.stream = None
#         self.vad = webrtcvad.Vad(vad_mode)
#         self.frame_duration = int(1000 * chunk_size / sample_rate)  # ms

#     def _audio_callback(self, indata, frames, time, status):
#         """
#         Callback function to process audio data in real-time.
#         """
#         if status:
#             print(status)
#         if self.recording:
#             self.audio_queue.put(indata.copy())

#     def start_recording(self):
#         """
#         Start recording audio in real-time.
#         """
#         self.recording = True
#         self.stream = sd.InputStream(
#             samplerate=self.sample_rate,
#             channels=self.channels,
#             callback=self._audio_callback,
#             blocksize=self.chunk_size
#         )
#         self.stream.start()
#         print("Recording started. Press Ctrl+C to stop.")

#     def stop_recording(self):
#         """
#         Stop recording audio.
#         """
#         self.recording = False
#         if self.stream:
#             self.stream.stop()
#             self.stream.close()
#         print("Recording stopped.")

#     def get_audio_chunk(self):
#         """
#         Get the next audio chunk from the queue.
#         Returns None if no data is available.
#         """
#         try:
#             return self.audio_queue.get_nowait()
#         except queue.Empty:
#             return None

#     def is_speech(self, audio_chunk):
#         # Ensure audio_chunk is mono and exactly 10ms (160 samples at 16kHz)
#         if audio_chunk.shape[1] > 1:
#             audio_chunk = audio_chunk.mean(axis=1, keepdims=True)
#         if audio_chunk.shape[0] != self.chunk_size:
#             return False
#         # Convert chunk to 16-bit PCM bytes
#         pcm_data = (audio_chunk * 32767).astype(np.int16).tobytes()
#         # VAD expects mono 16-bit PCM, 10/20/30ms frames
#         return self.vad.is_speech(pcm_data, self.sample_rate)

# class Transcriber:
#     """
#     Handles audio transcription using the Whisper model.
#     """
#     def __init__(self, model_name="base"):
#         """
#         Initialize the Transcriber with a Whisper model.
        
#         Args:
#             model_name (str): Name of the Whisper model to use
#         """
#         self.model = whisper.load_model(model_name)

#     def transcribe_chunk(self, audio_chunk, sample_rate=16000):
#         """
#         Transcribe a single audio chunk to text.
#         """
#         if audio_chunk is None or len(audio_chunk) == 0:
#             return ""
            
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
#             with wave.open(temp_file.name, 'wb') as wf:
#                 wf.setnchannels(1)
#                 wf.setsampwidth(2)
#                 wf.setframerate(sample_rate)
#                 wf.writeframes((audio_chunk * 32767).astype(np.int16).tobytes())
            
#             try:
#                 result = self.model.transcribe(temp_file.name)
#                 os.unlink(temp_file.name)
#                 return result["text"]
#             except Exception as e:
#                 print(f"Error transcribing chunk: {e}")
#                 return ""

# class ChatGPTProcessor:
#     """
#     Handles processing of transcribed text using the ChatGPT API.
#     """
#     def __init__(self):
#         """
#         Initialize the ChatGPTProcessor with API key from environment variables.
#         """
#         self.api_key = os.getenv('OPENAI_API_KEY')
#         if not self.api_key:
#             raise ValueError("OPENAI_API_KEY not found in environment variables")
#         self.client = OpenAI(api_key=self.api_key)
#         self.conversation_history = []

#     def process_text(self, text):
#         """
#         Process text using the ChatGPT API, maintaining conversation history.
        
#         Args:
#             text (str): Text to process
            
#         Returns:
#             str: ChatGPT's response
#         """
#         try:
#             # Append user message to conversation history
#             self.conversation_history.append({"role": "user", "content": text})
            
#             # Prepare messages for API call, including conversation history
#             messages = [{"role": "system", "content": "You are a helpful assistant."}] + self.conversation_history
            
#             response = self.client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=messages
#             )
            
#             # Extract assistant's response
#             assistant_response = response.choices[0].message.content
            
#             # Append assistant's response to conversation history
#             self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
#             return assistant_response
#         except Exception as e:
#             print(f"Error processing text with ChatGPT: {e}")
#             return None

# class TextToSpeech:
#     """
#     Handles text-to-speech conversion using the TTS library with Orpheus (VITS) model.
#     """
#     def __init__(self, model_name="tts_models/en/vctk/vits"):
#         """
#         Initialize the TextToSpeech with the Orpheus (VITS) model and select the first available speaker.
        
#         Args:
#             model_name (str): Name of the TTS model to use (default: VITS model)
#         """
#         self.tts = TTS(model_name=model_name)
#         self.speakers = self.tts.speakers
#         if self.speakers:
#             self.default_speaker = self.speakers[0]
#             print(f"Using default speaker: {self.default_speaker}")
#         else:
#             self.default_speaker = None
#             print("No speakers found for this model.")

#     def speak_text(self, text, output_file="response.wav", speaker=None):
#         """
#         Convert text to speech and save to a file using Orpheus.
        
#         Args:
#             text (str): Text to convert to speech
#             output_file (str): Path to save the audio file
#             speaker (str): Speaker name to use (default: first available)
            
#         Returns:
#             str: Path to the generated audio file
#         """
#         try:
#             speaker = speaker or self.default_speaker
#             self.tts.tts_to_file(text=text, file_path=output_file, speaker=speaker)
#             return output_file
#         except Exception as e:
#             print(f"Error converting text to speech: {e}")
#             return None

# async def process_audio(recorder, transcriber, chatgpt, tts):
#     """
#     Process audio chunks using VAD to detect end of speech.
#     """
#     buffer = []
#     silence_counter = 0
#     max_silence_frames = 10  # Number of silent frames before processing (tune as needed)
    
#     while recorder.recording:
#         chunk = recorder.get_audio_chunk()
#         if chunk is not None:
#             if recorder.is_speech(chunk):
#                 buffer.append(chunk)
#                 silence_counter = 0
#             else:
#                 if buffer:
#                     silence_counter += 1
#                 if silence_counter > max_silence_frames:
#                     # Process buffered speech
#                     audio_data = np.concatenate(buffer, axis=0)
#                     transcribed_text = transcriber.transcribe_chunk(audio_data)
#                     if transcribed_text.strip():
#                         print("\nTranscribed:", transcribed_text)
#                         response = chatgpt.process_text(transcribed_text)
#                         if response:
#                             print("\nChatGPT:", response)
#                             audio_file = tts.speak_text(response, "response.wav", speaker="p230")
#                             if audio_file:
#                                 print(f"Response saved as: {audio_file}")
#                     buffer = []
#                     silence_counter = 0
#         await asyncio.sleep(0.01)

# async def main():
#     """
#     Main function to orchestrate the application flow.
#     """
#     recorder = AudioRecorder()
#     transcriber = Transcriber()
#     chatgpt = ChatGPTProcessor()
#     tts = TextToSpeech()
    
#     print("Starting real-time voice processing. Press Ctrl+C to stop.")
    
#     try:
#         # Start recording in a separate thread
#         recorder.start_recording()
        
#         # Process audio in real-time
#         await process_audio(recorder, transcriber, chatgpt, tts)
        
#     except KeyboardInterrupt:
#         print("\nStopping...")
#     finally:
#         recorder.stop_recording()

# def test_tts():
#     """
#     Test function for text-to-speech functionality.
#     """
#     print("Testing Text-to-Speech with Orpheus model...")
#     tts = TextToSpeech()
#     test_text = "Hello! This is a test of the Orpheus text to speech system. How does it sound?"
#     print(f"\nConverting text: '{test_text}'")
    
#     # Use a male speaker (e.g., 'p230')
#     audio_file = tts.speak_text(test_text, "test_output.wav", speaker="p230")
#     if audio_file:
#         print(f"\nAudio saved to: {audio_file}")
#         print("You can play this file to hear the output.")

# if __name__ == "__main__":
#     # Uncomment the line you want to run:
#     # test_tts()  # Test TTS functionality
#     asyncio.run(main()) 