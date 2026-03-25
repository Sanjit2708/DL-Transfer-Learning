# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Transfer Learning uses a pre-trained model (like one trained on ImageNet) to solve a new but related task, improving efficiency and performance. VGG19 is a 19-layer CNN that extracts features using convolutional layers and classifies using fully connected layers. In transfer learning, the convolutional layers are usually frozen, and only the final layers are retrained for the new task.## Neural Network Model

## Theory
Transfer learning is a technique in deep learning where a pre-trained model is reused to solve a new but related problem, reducing the need for large datasets and training time. In this experiment, the VGG19 model, pre-trained on ImageNet, is used for image classification. The convolutional layers extract general features like edges and patterns, while new fully connected layers are added for the specific dataset. This approach improves accuracy and avoids overfitting even with limited data. Thus, transfer learning with VGG19 enables efficient and effective image classification.

## Neural Network Model
<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/43600277-359e-457b-9767-e954c59aac57" />

## DESIGN STEPS
### STEP 1: 
Collect and organize your images into folders (one folder per class).<br>
Resize all images to 224 × 224 (because VGG19 expects this size).<br>
### STEP 2: 
Use a library like TensorFlow or Keras.<br>
Normalize pixel values (scale between 0 and 1).<br>
Apply data augmentation (rotation, flip, zoom) to improve accuracy.<br>
### STEP 3: 
Load VGG19 with pre-trained ImageNet weights.<br>
Remove the top (final classification layer).<br>
Freeze the base layers so their weights do not change during training.<br>
### STEP 4: 
#### Add:
Flatten layer<br>
Dense (fully connected) layer<br>
Dropout layer (to reduce overfitting)<br>
Final Dense layer with Softmax (number of neurons = number of classes)<br>
### STEP 5: 
#### Choose:
Optimizer (e.g., Adam)<br>
Loss function (e.g., categorical crossentropy)<br>
Metrics (accuracy)<br>
Train the model using training data.<br>
Validate using validation data.<br>
### STEP 6: 
Test the model on unseen data.<br>
If accuracy is low:<br>
Unfreeze some top VGG19 layers.<br>
Train again with a low learning rate (fine-tuning).<br>
Save the final trained model.<br>
## PROGRAM

### Name:Sanjit A

### Register Number:212224220087

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
from torchvision.models import  VGG19_Weights
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

## Step 1: Load and Preprocess Data
# Define transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for pre-trained model input
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for pre-trained models
])

!unzip -qq ./chip_data.zip -d data

# Load dataset from a folder (structured as: dataset/class_name/images)
dataset_path = "./data/dataset/"
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)

# Display some input images
def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(5, 5))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0)  # Convert tensor format (C, H, W) to (H, W, C)
        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")
    plt.show()

# Show sample images from the training dataset
show_sample_images(train_dataset)

# Get the total number of samples in the training dataset
print(f"Total number of training samples: {len(train_dataset)}")

# Get the shape of the first image in the dataset
first_image, label = train_dataset[0]
print(f"Shape of the first image: {first_image.shape}")

# Get the total number of samples in the testing dataset

print(f"Total number of training samples: {len(test_dataset)}")

# Get the shape of the first image in the dataset

first_image, label = test_dataset[0]
print(f"Shape of the first image: {first_image.shape}")

# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

## Step 2: Load Pretrained Model and Modify for Transfer Learning
# Load a pre-trained VGG19 model
# write your code here
model=models.vgg19(weights=VGG19_Weights.DEFAULT)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torchsummary import summary
# Print model summary
summary(model, input_size=(3, 224, 224))

# Modify the final fully connected layer to match the dataset classes
# Write your code here
model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Freeze all layers except the final layer
for param in model.features.parameters():
    param.requires_grad = False  # Freeze feature extractor layers

# Include the Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

## Step 3: Train the Model
def train_model(model, train_loader,test_loader,num_epochs=10):
    # Write your code here
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
      running_loss = 0.0
      for images , labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
      train_losses.append(running_loss / len(train_loader))
        # Compute validation loss
        # Write your code here
      model.eval()
      val_loss = 0.0
      with torch.no_grad():
          for val_images, val_labels in test_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            outputs = model(val_images)
            loss = criterion(outputs, val_labels.unsqueeze(1).float())
            val_loss += loss.item()
      val_losses.append(val_loss / len(test_loader))
      model.train()

      print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print('Name: Sanjit A')
    print('Register Number: 212224220087')
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Train the model
# Write your code here
train_model(model, train_loader,test_loader,num_epochs=10)

## Step 4: Test the Model and Compute Confusion Matrix & Classification Report
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print('Name: Sanjit A')
    print('Register Number: 212224220087')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print('Name: Sanjit A')
    print('Register Number: 212224220087')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

# Evaluate the model
# write your code here
test_model(model, test_loader)

## Step 5: Predict on a Single Image and Display It
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)

        # Apply sigmoid to get probability, threshold at 0.5
        prob = torch.sigmoid(output)
        predicted = (prob > 0.5).int().item()


    class_names = class_names = dataset.classes
    # Display the image
    image_to_display = transforms.ToPILImage()(image)
    plt.figure(figsize=(4, 4))
    plt.imshow(image_to_display)
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted]}')
    plt.axis("off")
    plt.show()

    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted]}')

# Example Prediction
predict_image(model, image_index=80, dataset=test_dataset)


#Example Prediction
predict_image(model, image_index=25, dataset=test_dataset)


```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="1119" height="743" alt="image" src="https://github.com/user-attachments/assets/c5f3d398-62a0-4458-955f-1e103728eb90" />

## Confusion Matrix

<img width="863" height="735" alt="image" src="https://github.com/user-attachments/assets/8b25a2fe-e856-4cc5-9056-9d9e5b7b25eb" />

## Classification Report
<img width="559" height="249" alt="image" src="https://github.com/user-attachments/assets/22ed66dc-1df3-428f-951f-df69341d5f7c" />

### New Sample Data Prediction
<img width="452" height="509" alt="image" src="https://github.com/user-attachments/assets/01a560b6-4ed1-4d62-ba8e-9e5507c2857a" />

## RESULT
Thus, the image classification model using transfer learning with VGG19 architecture for the given dataset has been executed successfully.
