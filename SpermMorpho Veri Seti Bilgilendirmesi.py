import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
from PIL import Image
import os
import gdown

# Download dataset from Google Drive
url = "https://drive.google.com/uc?id=1QAXEANHEbCf86dnUkuehiyHdvk59QOB1"
output = "dataset.zip"

if not os.path.exists("dataset"):
    gdown.download(url, output, quiet=False)
    # Extract the zip file
    import zipfile
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall("dataset")

# Define dataset class if needed
class SpermDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = []
        self.image_paths = []
        self.labels = []

        if not os.path.exists(data_dir):
            raise ValueError(f"Dataset directory {data_dir} does not exist.")

        self.classes = os.listdir(data_dir)
        if not self.classes:
            raise ValueError(f"No classes found in dataset directory {data_dir}.")

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        for label, cls in enumerate(self.classes):
            class_dir = os.path.join(data_dir, cls)
            if not os.path.isdir(class_dir):
                print(f"Skipping non-directory: {class_dir}")
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path) and os.path.splitext(img_name)[1].lower() in valid_extensions:
                    self.image_paths.append(img_path)
                    self.labels.append(label)

        if not self.image_paths:
            raise ValueError(f"No valid image files found in dataset directory {data_dir}.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define paths
data_dir = "./dataset/train"  # Update to point to the correct subdirectory

# Check dataset structure
def check_dataset_structure(data_dir):
    print(f"Checking contents of {data_dir}:")
    for cls in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, cls)
        if os.path.isdir(class_dir):
            print(f"Class '{cls}' contains {len(os.listdir(class_dir))} files.")
        else:
            print(f"Skipping non-directory: {class_dir}")

check_dataset_structure(data_dir)

dataset = SpermDataset(data_dir, transform=transform)

# Check if dataset is empty
if len(dataset) == 0:
    raise ValueError("The dataset is empty. Please check the data directory or file structure.")

# Debugging information
print(f"Total images in dataset: {len(dataset)}")
print(f"Classes in dataset: {dataset.classes}")
for cls in dataset.classes:
    class_dir = os.path.join(data_dir, cls)
    if os.path.isdir(class_dir):
        print(f"Class: {cls}, Files: {len(os.listdir(class_dir))}")

# Split data
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model (if needed)
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "sperm_model.pth")

# Load the saved model
model.load_state_dict(torch.load("sperm_model.pth"))
model.eval()

# Evaluation
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Confusion matrix and classification report
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=dataset.classes)

# Plot confusion matrix
df_cm = pd.DataFrame(cm, index=dataset.classes, columns=dataset.classes)
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, cmap="OrRd", fmt="d", annot_kws={"size": 10}, cbar=False)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png")
plt.show()

# Save classification report
with open("classification_report.txt", "w") as f:
    f.write(report)
