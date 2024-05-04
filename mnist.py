import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import matplotlib.pyplot as plt
import os

from kan import KAN

print("Using PyTorch version:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data parameters
batch_size = 512
epochs = 500
data_dir = "./data"
dsize = 10000  # Subset size for feature extraction and training

# Load MNIST data
train_dataset = datasets.MNIST(
    f"{data_dir}", train=True, download=True, transform=ToTensor()
)
test_dataset = datasets.MNIST(
    f"{data_dir}", train=False, download=True, transform=ToTensor()
)


# Feature extractor CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x


# Preprocess data and extract features
def preprocess_data(dataset, dsize):
    mean = torch.mean(dataset.data[:dsize].to(torch.float32))
    std = torch.std(dataset.data[:dsize].to(torch.float32))
    model = CNN().to(device)

    dataset["train_input"] = (
        torch.flatten(
            model(((dataset.data[:dsize] - mean) / std).unsqueeze(1)).squeeze(1),
            start_dim=1,
        )
        .long()
        .to(device)
    )
    dataset["test_input"] = (
        torch.flatten(
            model(((dataset.test.data[:dsize] - mean) / std).unsqueeze(1)).squeeze(1),
            start_dim=1,
        )
        .long()
        .to(device)
    )

    dataset["train_label"] = dataset.targets[:dsize].long().to(device)
    dataset["test_label"] = dataset.test.targets[:dsize].long().to(device)

    return dataset


# Prepare data
dataset = {}
dataset = preprocess_data(train_dataset, dsize)

print("Train input shape:", dataset["train_input"].shape)
print("Test input shape:", dataset["test_input"].shape)
print("Train label shape:", dataset["train_label"].shape)
print("Test label shape:", dataset["test_label"].shape)


# Define KAN model
model = KAN(width=[dataset["train_input"].shape[1], 15, 10], grid=10, k=4)
model.to(device)


# Train the model
def train_model(model, dataset, epochs, batch_size, lr, loss_fn):
    results = model.train(
        dataset,
        opt="Adam",
        steps=epochs,
        lr=lr,
        batch=batch_size,
        loss_fn=loss_fn,
    )
    return results


results = train_model(model, dataset, epochs, batch_size, 0.05, nn.CrossEntropyLoss())
torch.save(model.state_dict(), "kan.pth")


# Load model and test accuracy
def test_accuracy(model, dataset):
    with torch.no_grad():
        predictions = torch.argmax(model(dataset["test_input"]), dim=1)
        correct = (predictions == dataset["test_label"]).float()
        accuracy = correct.mean()
    return accuracy * 100


model.load_state_dict(torch.load("kan.pth"))
acc = test_accuracy(model, dataset)

print(f"Test accuracy: {acc.item() * 100:.2f}%")

plt.plot(results["train_loss"], label="train_loss")
plt.plot(results["test_loss"], label="test_loss")
plt.legend()
plt.savefig("loss.png")

test_predictions = np.argmax(model(dataset["test_input"]).detach().numpy(), axis=1)
test_labels = dataset["test_label"]

labels_ = [i for i in range(10)]
cm = confusion_matrix(test_labels, test_predictions, labels=labels_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_)
disp.plot()

plt.savefig("confusion_matrix.png")
