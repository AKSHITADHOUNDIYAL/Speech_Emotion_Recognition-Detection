import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio

import torchaudio
import torch
from torch.utils.data import Dataset , DataLoader
from transformers import Wav2Vec2Model , Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, Trainer, TrainingArguments

import warnings
warnings.filterwarnings('ignore')


# LOAD THE DATASET

# In[8]:


dataset_path = r"E:\Documents\Sem_6\PROJECT\Speech Emotion Recognisation\TESS Toronto emotional speech set data"  
paths = []
labels = []

for dirname, _, filenames in os.walk(dataset_path):
    for filename in filenames:
        if filename.endswith(".wav"): 
            paths.append(os.path.join(dirname, filename))

            label = os.path.basename(dirname) 
            labels.append(label.lower())  
    if len(paths) == 2800:  
        break

print(f"Dataset Loaded! Found {len(paths)} audio files.")


# In[9]:


len(paths)


# In[10]:


paths[:5]


# In[11]:


labels[:5]


# In[12]:


## Create a dataframe

df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
df.head()


# In[13]:


print(df['label'].unique()) 


# In[14]:


df['label'].value_counts()


# In[15]:


print(df.columns)

# CREATE CUSTOM DATASET CLASS

# In[25]:


# Convert labels to integers

label_map = {label: idx for idx, label in enumerate(df['label'].unique())}
inverse_label_map = {idx: label for label, idx in label_map.items()}
df['labels'] = df['label'].map(label_map)
df.head(2)


# In[26]:


emotion = 'oaf_pleasant_surprise'
path = np.array(df['speech'][df['labels'] == 2])[0]
data, sampling_rate = librosa.load(path)
len(data)
sampling_rate * 2


# In[27]:


#  Add Background Noise

def add_background_noise(speech, noise_level=0.005):
    noise = torch.randn_like(speech) * noise_level
    return speech + noise

# Compute Mel Spectrogram

def get_mel_spectrogram(speech, sr):
    S = librosa.feature.melspectrogram(y=speech, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB

# Compute MFCC

def get_mfcc(speech, sr, n_mfcc=13):
    return librosa.feature.mfcc(y=speech, sr=sr, n_mfcc=n_mfcc)

# Normalize Audio

def normalize_audio(speech, target_length=16000):
    if len(speech) < target_length:
        padding = target_length - len(speech)
        speech = np.pad(speech, (0, padding))
    else:
        speech = speech[:target_length]
    return speech / np.max(np.abs(speech))  # normalize volume


# In[28]:


class SpeechEmotionDataset(Dataset):
    def __init__(self, df, processor, label2id, max_length=32000):
        self.df = df
        self.processor = processor
        self.max_length = max_length
        self.label2id = label2id or {label: idx for idx, label in enumerate(df['label'].unique())}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df.iloc[idx]['speech']
        label = self.df.iloc[idx]['label']
        label_id = self.label2id[label]

        # Load the audio file
        
        speech, sr = librosa.load(audio_path, sr=16000)
        speech = normalize_audio(speech)
        speech = add_background_noise(torch.tensor(speech)).numpy()  # Convert to numpy for librosa

        # Ensure the speech is 1D array for librosa processing
        
        speech = np.squeeze(speech)

        # Pad or truncate to max_length
        
        if len(speech) > self.max_length:
            speech = speech[:self.max_length]
        else:
            speech = np.pad(speech, (0, self.max_length - len(speech)), mode='constant')

        # Preprocess using the processor (like Wav2Vec2Processor)
        
        inputs = self.processor(speech, sampling_rate=16000, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)

        input_values = inputs.input_values.squeeze(0)  # Squeeze to remove extra batch dimension
        return {
            'input_values': input_values,
            'label': torch.tensor(label_id, dtype=torch.long)
        }


# In[29]:


# split the data for train and test

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# In[30]:


# initialize the processor and model

processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
model = Wav2Vec2ForSequenceClassification.from_pretrained('facebook/wav2vec2-base', num_labels=7)


# In[31]:


# Define preprocess function

def preprocess(example):
    audio = example["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt", padding=True)
    inputs["labels"] = example["label"]
    return inputs


# In[32]:


from torch.utils.data import DataLoader

# Create label2id mapping

label2id = {label: idx for idx, label in enumerate(train_df['label'].unique())}

# Create dataset objects

train_dataset = SpeechEmotionDataset(train_df, processor, label2id)
test_dataset = SpeechEmotionDataset(test_df, processor, label2id)


# In[33]:


# create dataloaders

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# In[34]:


train_dataset = SpeechEmotionDataset(df, processor, label2id)
print(train_dataset[0]["input_values"].size())


# SET TRAINING ARGUMENTS

# In[35]:


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    report_to=[],
    num_train_epochs=3,
    fp16= False,  
    fp16_full_eval=True,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True
)

print(training_args)


# In[36]:


# create functions for computing metrics

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids  # original labels
    preds = np.argmax(pred.predictions, axis=1)  # model predicted labels
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# In[37]:


label_names = [label.split("_")[-1] for label, _ in sorted(label2id.items(), key=lambda x: x[1])]


# In[38]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(pred, label_names):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# In[39]:


from transformers import Trainer

# initialize the trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()


# In[40]:


results = trainer.evaluate()
print(results)


# In[41]:


label_names = [label.split("_")[-1] for label, _ in sorted(label2id.items(), key=lambda x: x[1])]
eval_result = trainer.predict(test_dataset)
plot_confusion_matrix(eval_result, label_names)


# TEST PREDICTION

# In[43]:


import random
import torch

idx = random.randrange(0, len(test_dataset))

print("Original Label : ", inverse_label_map[int(test_dataset[idx]['label'])])

input_values = test_dataset[idx]['input_values'].unsqueeze(0).to(model.device).type(next(model.parameters()).dtype)

with torch.no_grad():
    outputs = model(input_values)

logits = outputs.logits
predicted_class = logits.argmax(dim=-1).item()

print("Predicted Label : ", inverse_label_map[predicted_class])


# In[44]:


sad_label_id = None
for k, v in inverse_label_map.items():
    if v.lower() == "oaf_sad":
        sad_label_id = k
        break

if sad_label_id is None:
    raise ValueError("Label 'oaf_sad' not found in inverse_label_map.")

sad_indices = [i for i in range(len(test_dataset)) if test_dataset[i]['label'] == sad_label_id]

if not sad_indices:
    raise ValueError("No sample with label 'oaf_sad' found in the test dataset.")

idx = random.choice(sad_indices)

print("Original Label : ", inverse_label_map[int(test_dataset[idx]['label'])])

input_values = test_dataset[idx]['input_values'].unsqueeze(0).to(model.device).type(next(model.parameters()).dtype)

with torch.no_grad():
    outputs = model(input_values)

logits = outputs.logits
predicted_class = logits.argmax(dim=-1).item()
print("Predicted Label : ", inverse_label_map[predicted_class])


# In[45]:


# Saving a file to the results directory

df.to_csv('./results/results.csv', index=False)


# In[47]:


torch.save(model.state_dict(), "emotion_model.pth")


# In[ ]:




