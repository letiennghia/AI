import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
from PIL import Image, ImageEnhance
import math
import os
import pandas as pd
from skimage import io

from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # to tensor before normalize
])
def load_image(file):
    return Image.open(file)
def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')
os.listdir('../input/hsgs-hackathon2022/train_data/Train_labels')
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=transform):
        video_files = os.listdir(img_dir)
        self.labels = list()
        self.imgs = list()
        for file_name in video_files: 
            cur_csv = pd.read_csv(os.path.join(label_dir, file_name+'.csv'))
            cur_len = len(cur_csv.index)
            for itr in range(cur_len):
                img_file_dir = os.path.join(os.path.join(img_dir, file_name), cur_csv.iloc[itr, 0] + '.PNG')
                self.imgs.append(img_file_dir) # image directory
                self.labels.append([cur_csv.iloc[itr, 0], cur_csv.iloc[itr, 1]]) # label
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
            
    def __getitem__(self, index):
        with open(self.imgs[index], 'rb') as f:
            image = load_image(f).convert('RGB')
        label = self.labels[index][1]
        filename = self.labels[index][0]
        
#         image, label = self.spm_transform(image, label)
        
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.labels)
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channel = 3
num_classes = 2
learning_rate = 1e-3
batch_size = 64
num_epochs = 200

# Load Data

dataset = CustomImageDataset(
    img_dir='../input/hsgs-hackathon2022/train_data/Train',
    label_dir='../input/hsgs-hackathon2022/train_data/Train_labels'
)
# Dataset is actually a lot larger ~25k images, just took out 10 pictures
# to upload to Github. It's enough to understand the structure and scale
# if you got more images.
n_train = math.floor(len(dataset)) 
train_set=dataset
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)


# Model
model = torchvision.models.googlenet(pretrained=True)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")

# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)
