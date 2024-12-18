import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import lightgbm as lgb
from tqdm import tqdm

# Load the embeddings and convert to numpy
print("Loading embeddings...")
designation_embeddings = torch.load('designation_embeddings.pt').cpu().numpy()

# Load the original data to get labels
print("Loading original data...")
df = pd.read_csv('train.csv')
labels = df['class'].tolist()

# Prepare the label encoder
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    designation_embeddings, 
    encoded_labels,
    test_size=0.2, 
    random_state=42, 
    stratify=encoded_labels
)

# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Set parameters
params = {
    'objective': 'multiclass',
    'num_class': len(le.classes_),
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'verbose': -1
}

# Train model
print("Training LGBM model...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[test_data],
)

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Convert back to original labels
y_test_original = le.inverse_transform(y_test)
y_pred_original = le.inverse_transform(y_pred_labels)

# Calculate and print F1 score
f1 = f1_score(y_test_original, y_pred_original, average='weighted')
print(f"\nWeighted F1 Score: {f1:.4f}")