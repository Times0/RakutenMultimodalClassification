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
import rich
from rich import print as rprint
import os
import hashlib
from datetime import datetime
import json

# Configuration parameters
CONFIG = {
    "embedder": "intfloat/multilingual-e5-large-instruct",
    "embedding_dimension": 1024,
    "use_description": True,
}

# Log configuration
rprint("[bold blue]Training Configuration:[/bold blue]")
rprint(f"🤖 Embedder: [green]{CONFIG['embedder']}[/green]")
rprint(f"📊 Embedding Dimension: [green]{CONFIG['embedding_dimension']}[/green]")
rprint(f"📝 Using Description Field: [green]{CONFIG['use_description']}[/green]")

USE_DESCRIPTION = CONFIG["use_description"]
EMBEDDINGS_DIMENSION = CONFIG["embedding_dimension"]

# Load model
model = SentenceTransformer(CONFIG["embedder"], trust_remote_code=True).to('cuda')
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
        assert designation_embeddings.shape[1] == EMBEDDINGS_DIMENSION, f"Designation embeddings dimension mismatch. Expected {EMBEDDINGS_DIMENSION}, got {designation_embeddings.shape[1]}"
        if USE_DESCRIPTION:
            assert description_embeddings.shape[1] == EMBEDDINGS_DIMENSION, f"Description embeddings dimension mismatch. Expected {EMBEDDINGS_DIMENSION}, got {description_embeddings.shape[1]}"
        self.designation_embeddings = designation_embeddings
        self.description_embeddings = description_embeddings if USE_DESCRIPTION else None
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(labels)

    def __len__(self):
        return len(self.designation_embeddings)
    
    def __getitem__(self, idx):
        if USE_DESCRIPTION:
            return (self.designation_embeddings[idx], 
                    self.description_embeddings[idx], 
                    self.labels[idx])
        return (self.designation_embeddings[idx], self.labels[idx])

# Function to get embeddings in batches
def get_embeddings(texts, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            batch_embeddings = model.encode(batch, max_length=EMBEDDINGS_DIMENSION)
            if isinstance(batch_embeddings, np.ndarray):
                batch_embeddings = torch.from_numpy(batch_embeddings)
            assert batch_embeddings.shape[1] == EMBEDDINGS_DIMENSION, f"Model output dimension mismatch. Expected {EMBEDDINGS_DIMENSION}, got {batch_embeddings.shape[1]}"
            embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)


# Classification head remains the same
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.3):
        super().__init__()
        # Input dimension depends on whether we're using description
        combined_dim = input_dim * 2 if USE_DESCRIPTION else input_dim
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
    
    def forward(self, x1, x2=None):
        if USE_DESCRIPTION:
            combined = torch.cat((x1, x2), dim=1)
        else:
            combined = x1
        return self.classifier(combined)

num_classes = 27


# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
classifier = ClassificationHead(input_dim=EMBEDDINGS_DIMENSION, num_classes=num_classes).to(device)
optimizer = AdamW(list(model.parameters()) + list(classifier.parameters()), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.1, patience=10, verbose=True
)

# load df and split data
df = pd.read_csv('output_X_train_update.csv')
designations = [preprocess(text) for text in df['designation'].tolist()]
descriptions = [preprocess(text) for text in df['description'].tolist()]
labels = df['class'].tolist()

def get_cache_path(texts, embedder_name):
    """Generate a unique cache path based on input texts and embedder"""
    # Create a hash of the texts and embedder to use as cache key
    text_hash = hashlib.md5(''.join(texts).encode()).hexdigest()
    embedder_hash = hashlib.md5(embedder_name.encode()).hexdigest()
    return f'cache/embeddings_{embedder_hash}_{text_hash}.pt'

def load_or_compute_embeddings(texts, embedder_name, batch_size=32):
    """Load embeddings from cache if they exist, otherwise compute and cache them"""
    # Create cache directory if it doesn't exist
    os.makedirs('cache', exist_ok=True)
    
    cache_path = get_cache_path(texts, embedder_name)
    
    # Try to load from cache
    if os.path.exists(cache_path):
        print(f"Loading embeddings from cache: {cache_path}")
        return torch.load(cache_path)
    
    # Compute embeddings
    print("Computing new embeddings...")
    embeddings = get_embeddings(texts, batch_size)
    
    # Save to cache
    print(f"Saving embeddings to cache: {cache_path}")
    torch.save(embeddings, cache_path)
    
    return embeddings

# Get embeddings for both columns
designation_embeddings = load_or_compute_embeddings(designations, CONFIG["embedder"])
description_embeddings = load_or_compute_embeddings(descriptions, CONFIG["embedder"])

# Normalize embeddings
designation_embeddings = F.normalize(designation_embeddings, p=2, dim=1)
description_embeddings = F.normalize(description_embeddings, p=2, dim=1)

# Remove or comment out the old load/save code
# torch.save(designation_embeddings, 'designation_embeddings.pt')
# torch.save(description_embeddings, 'description_embeddings.pt')
# designation_embeddings = torch.load('designation_embeddings.pt')
# description_embeddings = torch.load('description_embeddings.pt')

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
    for batch in train_dataloader:
        if USE_DESCRIPTION:
            batch_des, batch_desc, batch_labels = batch
            batch_desc = batch_desc.to(device)
        else:
            batch_des, batch_labels = batch
            batch_desc = None
        
        batch_des = batch_des.to(device)
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
        for batch in test_dataloader:
            if USE_DESCRIPTION:
                batch_des, batch_desc, batch_labels = batch
                batch_desc = batch_desc.to(device)
            else:
                batch_des, batch_labels = batch
                batch_desc = None
            
            batch_des = batch_des.to(device)
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

# Save training results
results = {
    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "best_f1": float(best_f1),
    "config": CONFIG,
}

# Save results to a JSON file
results_filename = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(results_filename, 'w') as f:
    json.dump(results, f, indent=4)

rprint(f"[bold green]✅ Training results saved to:[/bold green] {results_filename}")

print(f"Best F1 score: {best_f1:.4f}")

# Save final model state
torch.save(classifier.state_dict(), 'last_model.pt')
