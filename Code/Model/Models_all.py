#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Random Forest


# In[ ]:


import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Load the training and development data
train_file = '\PCA_train_SAN_updated_DAIC_with_labels.csv'
dev_file = '\PCA_dev_DAIC_with_labels.csv'

train_df = pd.read_csv(train_file)
dev_df = pd.read_csv(dev_file)

# Separate features and labels
X_train = train_df.drop(['Depression_Levels'], axis=1).values
y_train = train_df['Depression_Levels'].values

X_dev = dev_df.drop(['Depression_Levels'], axis=1).values
y_dev = dev_df['Depression_Levels'].values

# Perform random oversampling to balance the classes
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

print(f"Original y_train shape: {y_train.shape}")
print(f"Resampled y_train shape: {y_train_resampled.shape}")


# In[ ]:


# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the RF classifier
rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='f1_weighted')

grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best model
best_rf = grid_search.best_estimator_

# Print the best parameters
print(f"Best parameters found: {grid_search.best_params_}")


# In[ ]:


# Evaluate the best model on the development set
y_dev_pred = best_rf.predict(X_dev)

# Calculate the accuracy
dev_accuracy = best_rf.score(X_dev, y_dev)
print(f'Development Accuracy: {dev_accuracy}')

# Calculate the F1 score
f1 = f1_score(y_dev, y_dev_pred, average='weighted')
print(f'Development F1 Score: {f1}')

print(classification_report(y_dev, y_dev_pred))

print(confusion_matrix(y_dev, y_dev_pred))


# In[ ]:


##################################




##################################


# In[ ]:


# Stochastic Gradient Descent


# In[ ]:


from sklearn.linear_model import SGDClassifier

train_df = pd.read_csv(train_file)
dev_df = pd.read_csv(dev_file)

X_train = train_df.drop(['Depression_Levels'], axis=1).values
y_train = train_df['Depression_Levels'].values

X_dev = dev_df.drop(['Depression_Levels'], axis=1).values
y_dev = dev_df['Depression_Levels'].values

ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

print(f"Original y_train shape: {y_train.shape}")
print(f"Resampled y_train shape: {y_train_resampled.shape}")


# In[ ]:


param_grid = {
    'loss': ['hinge', 'log', 'squared_hinge', 'perceptron'],
    'penalty': ['none', 'l2', 'l1'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['optimal', 'constant'],
    'eta0': [0.01, 0.1, 1.0],
    'max_iter': [1000, 2000, 3000]
}

sgd = SGDClassifier(random_state=42)

grid_search = GridSearchCV(estimator=sgd, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='f1_weighted')

grid_search.fit(X_train_resampled, y_train_resampled)

best_sgd = grid_search.best_estimator_

print(f"Best parameters found: {grid_search.best_params_}")


# In[ ]:


y_dev_pred = best_sgd.predict(X_dev)

dev_accuracy = best_sgd.score(X_dev, y_dev)
print(f'Development Accuracy: {dev_accuracy}')

f1 = f1_score(y_dev, y_dev_pred, average='weighted')
print(f'Development F1 Score: {f1}')

print(classification_report(y_dev, y_dev_pred))

print(confusion_matrix(y_dev, y_dev_pred))


# In[ ]:


y_dev_pred = best_sgd.predict(X_dev)

dev_accuracy = best_sgd.score(X_dev, y_dev)
print(f'Development Accuracy: {dev_accuracy}')

f1 = f1_score(y_dev, y_dev_pred, average='weighted')
print(f'Development F1 Score: {f1}')

mae = mean_absolute_error(y_dev, y_dev_pred)
print(f'Development MAE: {mae}')

print(classification_report(y_dev, y_dev_pred))

print(confusion_matrix(y_dev, y_dev_pred))


# In[ ]:


##################################




##################################


# In[ ]:


# K-Nearest Neighbors


# In[ ]:


import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score


train_df = pd.read_csv(train_file)
dev_df = pd.read_csv(dev_file)

X_train = train_df.drop(['Depression_Levels'], axis=1).values
y_train = train_df['Depression_Levels'].values

X_dev = dev_df.drop(['Depression_Levels'], axis=1).values
y_dev = dev_df['Depression_Levels'].values

ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

print(f"Original y_train shape: {y_train.shape}")
print(f"Resampled y_train shape: {y_train_resampled.shape}")

param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'brute'],
    'leaf_size': [10, 20, 30, 40]
}

# Create the KNN model
knn = KNeighborsClassifier()

grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='f1_weighted')

grid_search.fit(X_train_resampled, y_train_resampled)


best_knn = grid_search.best_estimator_

print(f"Best parameters found: {grid_search.best_params_}")

y_dev_pred = best_knn.predict(X_dev)

dev_accuracy = best_knn.score(X_dev, y_dev)
print(f'Development Accuracy: {dev_accuracy}')

f1 = f1_score(y_dev, y_dev_pred, average='weighted')
print(f'Development F1 Score: {f1}')

print(classification_report(y_dev, y_dev_pred))

print(confusion_matrix(y_dev, y_dev_pred))


# In[ ]:


##################################




##################################


# In[ ]:


# Neural Networks


# In[ ]:


import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score


train_df = pd.read_csv(train_file)
dev_df = pd.read_csv(dev_file)

X_train = train_df.drop(['Depression_Levels'], axis=1).values
y_train = train_df['Depression_Levels'].values

X_dev = dev_df.drop(['Depression_Levels'], axis=1).values
y_dev = dev_df['Depression_Levels'].values

ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

y_train_encoded = tf.keras.utils.to_categorical(y_train_resampled)
y_dev_encoded = tf.keras.utils.to_categorical(y_dev)

print(f"Original y_train shape: {y_train.shape}")
print(f"Resampled y_train shape: {y_train_resampled.shape}")
print(f"y_train_encoded shape: {y_train_encoded.shape}")
print(f"y_dev_encoded shape: {y_dev_encoded.shape}")

# Model creation function
def create_model(optimizer='adam', activation='relu', hidden_layers=(64,), dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(hidden_layers[0], input_dim=X_train_resampled.shape[1], activation=activation))
    model.add(Dropout(dropout_rate))
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(y_train_encoded.shape[1], activation='softmax'))  # Number of classes
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Create the KerasClassifier
model = KerasClassifier(build_fn=create_model, verbose=0)

param_grid = {
    'optimizer': ['adam', 'sgd', 'rmsprop'],
    'activation': ['relu', 'tanh', 'sigmoid'],
    'batch_size': [16, 32, 64, 107],
    'epochs': [10, 50, 100],
    'hidden_layers': [(64,), (64, 64), (128, 64), (128, 128)],
    'dropout_rate': [0.2, 0.3, 0.5]
}


grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

grid_search.fit(X_train_resampled, y_train_encoded)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Best parameters found: {best_params}")

dev_loss, dev_accuracy = best_model.model.evaluate(X_dev, y_dev_encoded)
print(f'Development Accuracy: {dev_accuracy}')

y_pred = best_model.model.predict(X_dev)
y_pred_classes = np.argmax(y_pred, axis=1)
y_dev_classes = np.argmax(y_dev_encoded, axis=1)

f1 = f1_score(y_dev_classes, y_pred_classes, average='weighted')
print(f'Development F1 Score: {f1}')

print(classification_report(y_dev_classes, y_pred_classes))

print(confusion_matrix(y_dev_classes, y_pred_classes))


# In[1]:


##################################




##################################


# In[2]:


# Logistic Regression


# In[ ]:


import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score

train_df = pd.read_csv(train_file)
dev_df = pd.read_csv(dev_file)

X_train = train_df.drop(['Depression_Levels'], axis=1).values
y_train = train_df['Depression_Levels'].values

X_dev = dev_df.drop(['Depression_Levels'], axis=1).values
y_dev = dev_df['Depression_Levels'].values

ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

print(f"Original y_train shape: {y_train.shape}")
print(f"Resampled y_train shape: {y_train_resampled.shape}")

param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'sag', 'saga'],
    'max_iter': [100, 200, 300]
}

# Create the LR model
log_reg = LogisticRegression(random_state=42)

grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

grid_search.fit(X_train_resampled, y_train_resampled)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Best parameters found: {best_params}")

y_dev_pred = best_model.predict(X_dev)

dev_accuracy = best_model.score(X_dev, y_dev)
print(f'Development Accuracy: {dev_accuracy}')

f1 = f1_score(y_dev, y_dev_pred, average='weighted')
print(f'Development F1 Score: {f1}')

print(classification_report(y_dev, y_dev_pred))

print(confusion_matrix(y_dev, y_dev_pred))


# In[6]:


##################################




##################################


# In[7]:


# SVM


# In[ ]:


import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score

train_df = pd.read_csv(train_file)
dev_df = pd.read_csv(dev_file)

X_train = train_df.drop(['Depression_Levels'], axis=1).values
y_train = train_df['Depression_Levels'].values

X_dev = dev_df.drop(['Depression_Levels'], axis=1).values
y_dev = dev_df['Depression_Levels'].values

ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

print(f"Original y_train shape: {y_train.shape}")
print(f"Resampled y_train shape: {y_train_resampled.shape}")

param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],  # Only relevant for 'poly' kernel
    'gamma': ['auto', 0.1, 0.01, 0.001]  # Relevant for 'rbf', 'poly', and 'sigmoid' kernels
}

# Create the SVM model
svm_model = SVC(probability=True, random_state=42)

grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best parameters and estimator
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Best parameters found: {best_params}")

y_dev_pred = best_model.predict(X_dev)

dev_accuracy = best_model.score(X_dev, y_dev)
print(f'Development Accuracy: {dev_accuracy}')

f1 = f1_score(y_dev, y_dev_pred, average='weighted')
print(f'Development F1 Score: {f1}')

print(classification_report(y_dev, y_dev_pred))

print(confusion_matrix(y_dev, y_dev_pred))


# In[3]:


##################################




##################################


# In[5]:


# Gaussian


# In[ ]:


import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score

train_df = pd.read_csv(train_file)
dev_df = pd.read_csv(dev_file)

X_train = train_df.drop(['Depression_Levels'], axis=1).values
y_train = train_df['Depression_Levels'].values

X_dev = dev_df.drop(['Depression_Levels'], axis=1).values
y_dev = dev_df['Depression_Levels'].values

ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

print(f"Original y_train shape: {y_train.shape}")
print(f"Resampled y_train shape: {y_train_resampled.shape}")

param_grid = {
    'kernel': [1.0 * RBF(length_scale=1.0), 1.0 * Matern(length_scale=1.0), 1.0 * RationalQuadratic(length_scale=1.0)],
    'alpha': [1e-2, 1e-3, 1e-4]
}


gp_model = GaussianProcessClassifier(random_state=42)


grid_search = GridSearchCV(estimator=gp_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)


grid_search.fit(X_train_resampled, y_train_resampled)


best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Best parameters found: {best_params}")


y_dev_pred = best_model.predict(X_dev)

dev_accuracy = best_model.score(X_dev, y_dev)
print(f'Development Accuracy: {dev_accuracy}')

f1 = f1_score(y_dev, y_dev_pred, average='weighted')
print(f'Development F1 Score: {f1}')

print(classification_report(y_dev, y_dev_pred))

print(confusion_matrix(y_dev, y_dev_pred))

