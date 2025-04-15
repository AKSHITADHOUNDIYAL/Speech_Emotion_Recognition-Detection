import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class EmotionModel(nn.Module):
    def __init__(self):
        super(EmotionModel, self).__init__()
        # Load the Wav2Vec2 model using Flax weights
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        
        # Define custom layers for emotion recognition
        self.fc1 = nn.Linear(1024, 512)  # Assuming Wav2Vec2 outputs 1024 features
        self.fc2 = nn.Linear(512, 6)     # Assuming 6 emotion classes (adjust as needed)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_values):
        # Pass audio features through Wav2Vec2
        outputs = self.wav2vec2(input_values)
        features = outputs.last_hidden_state.mean(dim=1)  # Take mean of all time steps
        
        # Pass through custom layers
        x = self.relu(self.fc1(features))
        x = self.fc2(x)
        
        # Get emotion class probabilities
        return self.softmax(x)
    