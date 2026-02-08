import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
import os
from tqdm import trange
from torch.utils.data import random_split

class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes=10, reduced_channels=100):
        super(ModifiedResNet18, self).__init__()
        original_resnet = models.resnet18(weights="DEFAULT")

        # Keep all layers up to layer3 (optional for now)
        self.features = nn.Sequential(
            original_resnet.conv1,
            original_resnet.bn1,
            original_resnet.relu,
            original_resnet.maxpool,
            original_resnet.layer1,
            original_resnet.layer2,
            original_resnet.layer3
        )

        self.reduce =nn.Sequential(nn.Conv2d(256, reduced_channels, kernel_size=3,padding=1),nn.BatchNorm2d(reduced_channels),nn.ReLU(inplace=False))

        # Classifier

        self.classifier = nn.Linear(reduced_channels*14**2, num_classes) # we have a better resolution of 14x14 after the last conv layer

    def forward(self, x):
        x = self.features(x)
        x = self.reduce(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

resnet100 = ModifiedResNet18(num_classes=10,reduced_channels=100)
resnet50 = ModifiedResNet18(num_classes=10,reduced_channels=50)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load CIFAR-10 test dataset
#dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
dataset = STL10(root='/data', download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

split = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [split, len(dataset) - split])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50.to(device)
for name, param in resnet50.named_parameters():
    if 'reduce' not in name and 'classifier' not in name:
        param.requires_grad = False


optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, resnet50.parameters()), lr=1e-3)

def evaluation_from_checkpoint(model, checkpoint_data):
    model.load_state_dict(checkpoint_data['model_state_dict'])
    return model

def train(model, train_loader, val_loader, optimizer, criterion, batch_size=32, learning_rate=1e-3, num_epochs=21):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists('Checkpoints'):
        os.mkdir('Checkpoints')
    model.to(device)

    for epoch in trange(num_epochs):
        # ---------- Training ----------
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {train_loss}")
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
            }, os.path.join('Checkpoints', f'checkpoint_ep{epoch}.pth'))

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()

                # Accuracy (if classification)
                _, predicted = torch.max(val_outputs, 1)
                correct += (predicted == val_labels).sum().item()
                total += val_labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = 100.0 * correct / total

        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    return model

train(resnet50, train_loader, val_loader, optimizer, criterion)

resnet50_checkpoint_data = torch.load("Checkpoints/checkpoint_ep20.pth", weights_only=True)
resnet50 = evaluation_from_checkpoint(resnet50, resnet50_checkpoint_data)

resnet50.eval()