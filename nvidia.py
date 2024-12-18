import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
import re
# Load model

model = SentenceTransformer('intfloat/multilingual-e5-large-instruct', trust_remote_code=True).to('cuda')
model.train()


def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = text.strip()
    return text

# Custom dataset class
class TextClassificationDataset(Dataset):
    def __init__(self, designation_embeddings, description_embeddings, labels):
        self.designation_embeddings = designation_embeddings
        self.description_embeddings = description_embeddings
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(labels)

    def __len__(self):
        return len(self.designation_embeddings)
    
    def __getitem__(self, idx):
        return (self.designation_embeddings[idx], 
                self.description_embeddings[idx], 
                self.labels[idx])

# Function to get embeddings in batches
def get_embeddings(texts, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            batch_embeddings = model.encode(batch, max_length=512)
            if isinstance(batch_embeddings, np.ndarray):
                batch_embeddings = torch.from_numpy(batch_embeddings)
            embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)


# Classification head remains the same
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.3):
        super().__init__()
        # Double the input dimension as we're concatenating two embeddings
        combined_dim = input_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x1, x2):
        # Concatenate the two embeddings
        combined = torch.cat((x1, x2), dim=1)
        return self.classifier(combined)

num_classes = 27


# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
classifier = ClassificationHead(input_dim=1024, num_classes=num_classes).to(device)
optimizer = AdamW(list(model.parameters()) + list(classifier.parameters()), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.1, patience=10, verbose=True
)

# load df and split data
df = pd.read_csv('train.csv')
designations = [preprocess(text) for text in df['designation'].tolist()]
descriptions = [preprocess(text) for text in df['description'].tolist()]
labels = df['class'].tolist()

"""# Get embeddings for both columns
designation_embeddings = get_embeddings(designations)
description_embeddings = get_embeddings(descriptions)

# Normalize embeddings
designation_embeddings = F.normalize(designation_embeddings, p=2, dim=1)
description_embeddings = F.normalize(description_embeddings, p=2, dim=1)

# Save embeddings
torch.save(designation_embeddings, 'designation_embeddings.pt')
torch.save(description_embeddings, 'description_embeddings.pt')"""

# load
designation_embeddings = torch.load('designation_embeddings.pt')
description_embeddings = torch.load('description_embeddings.pt')

# Split the data
X_train_des, X_test_des, X_train_desc, X_test_desc, y_train, y_test = train_test_split(
    designation_embeddings, description_embeddings, labels, 
    test_size=0.2, random_state=42, stratify=labels
)

# Create datasets and dataloaders
train_dataset = TextClassificationDataset(X_train_des, X_train_desc, y_train)
test_dataset = TextClassificationDataset(X_test_des, X_test_desc, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop
num_epochs = 500
# Early stopping parameters
best_f1 = 0
patience = 50
patience_counter = 0
best_model_state = None

for epoch in range(num_epochs):
    # Training
    classifier.train()
    total_loss = 0
    for batch_des, batch_desc, batch_labels in train_dataloader:
        batch_des = batch_des.to(device)
        batch_desc = batch_desc.to(device)
        batch_labels = batch_labels.to(device)
        
        outputs = classifier(batch_des, batch_desc)
        loss = criterion(outputs, batch_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Evaluation
    classifier.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_des, batch_desc, batch_labels in test_dataloader:
            batch_des = batch_des.to(device)
            batch_desc = batch_desc.to(device)
            outputs = classifier(batch_des, batch_desc)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.numpy())
    
    # Convert predictions back to original labels
    original_preds = train_dataset.label_encoder.inverse_transform(all_preds)
    original_labels = train_dataset.label_encoder.inverse_transform(all_labels)
    
    # Calculate F1 score
    f1 = f1_score(original_labels, original_preds, average='weighted')
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataloader):.4f}, F1-Score: {f1:.4f}")
    
    # Early stopping logic
    if f1 > best_f1:
        best_f1 = f1
        patience_counter = 0
        best_model_state = classifier.state_dict()
        # Save best model
        torch.save(best_model_state, 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            classifier.load_state_dict(best_model_state)
            break
    
    # Add learning rate scheduler
    scheduler.step(f1)  # Update learning rate based on F1 score

    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

print(f"Best F1 score: {best_f1:.4f}")

# Save final model state
torch.save(classifier.state_dict(), 'last_model.pt')
