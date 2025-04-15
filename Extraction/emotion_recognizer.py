import torch
import torch.nn as nn
import torchaudio
import numpy as np
import pyaudio
import time
from collections import deque
from Backend.emotion_model import EmotionModel  


class EmotionRecognizer:
    def __init__(self, model_path: str, device='cpu'):
        self.device = device
        self.model = EmotionModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def predict(self, waveform: np.ndarray, sampling_rate: int) -> str:
        """
        Predict emotion from waveform.
        """
        # Resample if needed
        if sampling_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(torch.tensor(waveform))
        else:
            waveform = torch.tensor(waveform)

        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]

        input_values = waveform.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values)
            predicted_label = torch.argmax(logits, dim=1).item()

        return self.labels[predicted_label]


class RealTimeEmotionRecognizer:
    def __init__(self, model_path, device='cpu', sample_rate=16000, chunk_size=1024, duration=3):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.duration = duration
        self.device = device

        self.recognizer = EmotionRecognizer(model_path=model_path, device=device)

    def record_and_predict(self):
        """
        Records microphone audio and predicts emotion in real-time.
        """
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.chunk_size)

        print("🎙️ Real-time Emotion Recognition started... (Ctrl+C to stop)")
        audio_buffer = deque(maxlen=int(self.sample_rate / self.chunk_size * self.duration))

        try:
            while True:
                audio_chunk = stream.read(self.chunk_size)
                audio_buffer.append(audio_chunk)

                audio_data = np.frombuffer(b''.join(audio_buffer), dtype=np.int16).astype(np.float32)
                audio_data = audio_data / 32768.0  # Normalize to [-1, 1]

                emotion = self.recognizer.predict(audio_data, self.sample_rate)
                print(f"Predicted Emotion: {emotion}")

                time.sleep(self.duration)

        except KeyboardInterrupt:
            print("🛑 Stopped.")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


# Example usage
if __name__ == "__main__":
    model_path = "E:/Documents/Sem_6/PROJECT/Speech Emotion Recognisation/Prediction/models/emotion_model.pth"
    real_time_recognizer = RealTimeEmotionRecognizer(model_path=model_path)
    real_time_recognizer.record_and_predict()
