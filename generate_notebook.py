import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """\
# Image Classification with Transfer Learning (PyTorch)

This notebook demonstrates how to train a convolutional neural network (CNN) to classify images into 6 distinct tags.
Given the small size of the training dataset, training a deep model from scratch would likely result in severe overfitting.
Therefore, we will use **Transfer Learning** with a pre-trained **MobileNetV2** model.

## 1. Setup and Imports
"""

code_imports = """\
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""

text_data = """\
## 2. Data Preparation and Augmentation

Data augmentation is crucial when working with small datasets. We'll apply random crops, horizontal flips, and color jittering to artificially expand our training data and improve model generalization.
"""

code_data = """\
# Define data directories
train_dir = 'data_train'
test_dir = 'data_test'

# Define transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
image_datasets = {
    'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'test': datasets.ImageFolder(test_dir, data_transforms['test'])
}

# Create dataloaders
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=4)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

print(f"Training images: {dataset_sizes['train']}")
print(f"Testing images: {dataset_sizes['test']}")
print(f"Classes: {class_names}")
"""

text_model = """\
## 3. Model Construction

We'll load a pre-trained MobileNetV2 architecture. We freeze the early layers (feature extractor) and only train the final classification head tailored to our 6 classes. MobileNetV2 is specifically chosen because it's lightweight, fast, and highly effective for standard image classification.
"""

code_model = """\
# Load pre-trained MobileNetV2
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier head
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(class_names))

# Move model to device
model = model.to(device)

# Define Loss function and optimizer
criterion = nn.CrossEntropyLoss()
# Optimize only the classifier parameters initially
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

print(model.classifier)
"""

text_train = """\
## 4. Training Loop

We train the model over a specified number of epochs. In each epoch, we compute the loss, update the model weights using backpropagation, and track our training accuracy.
"""

code_train = """\
num_epochs = 10

print("Starting Training...")
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)
    
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    # Iterate over data
    for inputs, labels in dataloaders['train']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        # Backward pass + optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    epoch_loss = running_loss / dataset_sizes['train']
    epoch_acc = running_corrects.double() / dataset_sizes['train']
    
    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

print("Training Complete!")
"""

text_eval = """\
## 5. Evaluation

After training, we evaluate the model's performance on the unseen `data_test` set. We use metrics like Accuracy, Precision, Recall, and F1-score. A Confusion Matrix will give us granular insight into which classes are being misclassified.
"""

code_eval = """\
model.eval()

all_preds = []
all_labels = []

# Disable gradient calculation for inference
with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

# Plot Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
"""

text_conclusion = """\
## 6. Future Improvements

Given more time and resources, the solution could be improved via:
1. **Data Expansion:** The most critical issue is the lack of data. Collecting more diverse representative images for each class is paramount.
2. **Fine-Tuning:** Unfreezing the last few layers of the MobileNetV2 feature extractor and training them with a very small learning rate to adapt to the specific dataset features.
3. **Hyperparameter Optimization:** Systematically tuning learning rate, batch size, and weight decay using tools like Optuna.
4. **Ensemble Methods:** Combining predictions from multiple models (e.g., ResNet, EfficientNet) to increase robustness.
5. **Handling Class Imbalance:** If some tags have significantly fewer images, applying class-weighted loss functions.
"""

# Append cells to the notebook
nb.cells = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(text_data),
    nbf.v4.new_code_cell(code_data),
    nbf.v4.new_markdown_cell(text_model),
    nbf.v4.new_code_cell(code_model),
    nbf.v4.new_markdown_cell(text_train),
    nbf.v4.new_code_cell(code_train),
    nbf.v4.new_markdown_cell(text_eval),
    nbf.v4.new_code_cell(code_eval),
    nbf.v4.new_markdown_cell(text_conclusion)
]

# Write out the notebook file
with open('Image_Classification_Solution.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Successfully generated Image_Classification_Solution.ipynb")
