#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#First, we combine all the embeddings into a single file (a .csv file with 769 columns. First 768 are output from wav2vec,
#and the last column is for the file name). For instance, 'mild_negative_natural' will be the label for all the embeddings
# extracted from audios under the file path '\depression\Data\Audios\mild\negative\natural'


# In[ ]:


########## Combine all embeddings with different combinationas of labels
import pandas as pd
import os

# Path to the folder containing the embedding files
folder_path = 'F:\Embeddings'

# List to hold all DataFrames
dfs = []

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        df['file_name'] = filename  # Add the file name to the 'file_name' column
        dfs.append(df)

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined DataFrame to a new CSV file
output_file_path = 'F:\\Embeddings\\combined_embeddings.csv'
combined_df.to_csv(output_file_path, index=False)


# In[ ]:


import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the combined embeddings file
combined_file_path = 'F:\\Embeddings\\combined_embeddings.csv'
combined_df = pd.read_csv(combined_file_path)

# Filter the DataFrame for the four highest-level labels
labels_to_include = ['mild_embeddings.csv', 'minimal_embeddings.csv', 'moderate_embeddings.csv', 'severe_embeddings.csv']
filtered_df = combined_df[combined_df['file_name'].isin(labels_to_include)]

# Separate features and labels
features = filtered_df.drop(['file_name'], axis=1)
labels = filtered_df['file_name']

# Perform PCA with 50 components
n_components = 50
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(features)

# Calculate explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance.cumsum()

# Print explained variance
# p = 35 is the number we found that meets the 95% threshold
print(f"Cumulative Explained Variance: {cumulative_explained_variance[:35]}")

# Calculate component loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

loadings_df = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(n_components)], index=features.columns)

print(loadings_df.iloc[:, :10])

# Visualize the cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.title('')
plt.xlabel('Number of Principal Components', fontsize=14)
plt.ylabel('Cumulative Explained Variance', fontsize=14)
plt.grid(True)

plt.savefig('D:\\pca_components.png', bbox_inches='tight', dpi=300)
plt.show()

