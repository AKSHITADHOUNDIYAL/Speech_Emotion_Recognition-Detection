import numpy as np
import librosa
import soundfile as sf
import pyaudio
import wave
import tempfile
import os
import logging
from io import BytesIO

def get_supported_formats():
    """Return a list of supported audio formats"""
    return ["wav", "mp3", "ogg", "flac", "m4a"]

def process_audio_file(file_path):
    """
    Process an audio file and extract features for emotion recognition.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Dictionary containing extracted audio features or None in case of error
    """
    # Check if file format is supported (based on file extension)
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension not in get_supported_formats():
        logging.error(f"Unsupported file format: {file_extension}")
        return None
    
    try:
        # Load the audio file with librosa
        y, sr = librosa.load(file_path, sr=22050)
        
        # Extract features
        features = extract_audio_features(y, sr)
        return features
    
    except Exception as e:
        logging.error(f"Error processing audio file: {str(e)}")
        return None

def extract_audio_features(y, sr):
    """
    Extract audio features for emotion recognition.
    
    Args:
        y: Audio time series
        sr: Sampling rate
        
    Returns:
        Dictionary containing extracted features (excluding waveform data)
    """
    # Calculate features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    energy = np.mean(librosa.feature.rms(y=y)[0])
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_mean = np.mean(mel_spec, axis=1)
    
    # Pack only the feature data
    features = {
        'mfccs_mean': mfccs_mean,
        'mfccs_std': mfccs_std,
        'spectral_centroid_mean': np.mean(spectral_centroid),
        'spectral_contrast_mean': np.mean(spectral_contrast),
        'spectral_rolloff_mean': np.mean(spectral_rolloff),
        'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
        'energy': energy,
        'chroma_mean': chroma_mean,
        'mel_spec_mean': mel_spec_mean
    }
    
    return features

def record_audio(duration=5, sample_rate=22050, channels=1, chunk=1024):
    """
    Record audio from the microphone.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Audio sampling rate
        channels: Number of audio channels
        chunk: Frames per buffer
        
    Returns:
        Tuple of (recorded_audio_bytes, audio_features)
    """
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Open stream
    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk
    )
    
    # Record audio
    frames = []
    for i in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save the recorded audio to a BytesIO object
    audio_bytes = BytesIO()
    wf = wave.open(audio_bytes, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    # Reset the position to the start of the BytesIO object
    audio_bytes.seek(0)
    
    # Create a temporary file for librosa processing
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file.write(audio_bytes.getvalue())
        temp_file_path = temp_file.name
    
    try:
        # Process the audio file
        y, sr = librosa.load(temp_file_path, sr=sample_rate)
        features = extract_audio_features(y, sr)
    except Exception as e:
        logging.error(f"Error processing recorded audio: {str(e)}")
        features = None
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    
    # Reset the position again to be used by st.audio
    audio_bytes.seek(0)
    
    return audio_bytes, features

# Set up logging
logging.basicConfig(level=logging.INFO)

