#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Calculation of Euclidean distance and cosine similarity among the highest levels of labels (depression severity).


# In[ ]:


import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np

# Load the combined embeddings file
combined_file_path = 'F:\\combined_embeddings.csv'
combined_df = pd.read_csv(combined_file_path)

# Filter the DataFrame for the four highest-level labels
labels_to_include = ['mild_embeddings.csv', 'minimal_embeddings.csv', 'moderate_embeddings.csv', 'severe_embeddings.csv']
filtered_df = combined_df[combined_df['file_name'].isin(labels_to_include)]

# Separate features and labels
features = filtered_df.drop(['file_name'], axis=1)
labels = filtered_df['file_name']

# Compute the mean embeddings for other groups
mild_embeddings = features[labels == 'mild_embeddings.csv'].mean(axis=0)
minimal_embeddings = features[labels == 'minimal_embeddings.csv'].mean(axis=0)

# Compute Euclidean distances between other pairs of groups
dist_mild_minimal = euclidean(mild_embeddings, minimal_embeddings)
dist_mild_moderate = euclidean(mild_embeddings, moderate_embeddings)
dist_mild_severe = euclidean(mild_embeddings, severe_embeddings)
dist_minimal_moderate = euclidean(minimal_embeddings, moderate_embeddings)
dist_minimal_severe = euclidean(minimal_embeddings, severe_embeddings)

# Print the distances for comparison
print(f"Euclidean Distance between mild and minimal embeddings: {dist_mild_minimal}")
print(f"Euclidean Distance between mild and moderate embeddings: {dist_mild_moderate}")
print(f"Euclidean Distance between mild and severe embeddings: {dist_mild_severe}")
print(f"Euclidean Distance between minimal and moderate embeddings: {dist_minimal_moderate}")
print(f"Euclidean Distance between minimal and severe embeddings: {dist_minimal_severe}")


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


moderate_embeddings = features[labels == 'minimal_embeddings.csv'].mean(axis=0)
severe_embeddings = features[labels == 'mild_embeddings.csv'].mean(axis=0)

# Compute cosine similarity between the mean embeddings
cos_sim = cosine_similarity([moderate_embeddings], [severe_embeddings])
print(f"Cosine Similarity between moderate and severe embeddings: {cos_sim[0][0]}")

