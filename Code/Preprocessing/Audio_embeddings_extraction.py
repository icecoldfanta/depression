#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Use wav2vec2 to extract embeddings from all the audio files stored into different subfolders under /depression/Data/Audios


# In[ ]:


import os
import torch
import librosa
import numpy as np
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Model


# In[ ]:


# Load pre-trained Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")


# In[ ]:


# Move the model to the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# device = torch.device('cpu')
# model.to(device)

# Extract embeddings for a batch of audio files
def extract_embeddings(audio_files, min_length=16000): 
    embeddings = []
    for file in audio_files:
        speech, sr = librosa.load(file, sr=16000)
        if len(speech) < min_length:
            padding = min_length - len(speech)
            speech = np.pad(speech, (0, padding), 'constant')
        
        input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values.to(device)
        with torch.no_grad():
            hidden_states = model(input_values).last_hidden_state
        embeddings.append(hidden_states.mean(dim=1).squeeze().cpu().numpy())
    return np.array(embeddings)

# Function to process a sample of audio files
def process_sample_files(input_folder, sample_size=50):
    all_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.wav')]
    sampled_files = np.random.choice(all_files, sample_size, replace=False)
    
    embeddings = extract_embeddings(sampled_files)
    
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df['file_name'] = sampled_files  # Add file names as a column
    
    # Save to CSV
    output_file = os.path.join(input_folder, '.csv')
    embeddings_df.to_csv(output_file, index=False)
    print(f'Embeddings saved to {output_file}')
    
    return embeddings_df

# Example usage
input_folder = 'F:\\audios'  # Change this to the actual folder path
sample_size = 35  # Number of files to sample

embeddings_df = process_sample_files(input_folder, sample_size)

