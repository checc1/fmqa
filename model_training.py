import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
import os
from tqdm import trange
from torch.utils.data import random_split
import sys


class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes, reduced_channels):
        super(ModifiedResNet18, self).__init__()
        original_resnet = models.resnet18(weights="DEFAULT")

        # Keep all layers up to layer3 (optional for now)
        self.features = nn.Sequential(
            original_resnet.conv1,
            original_resnet.bn1,
            original_resnet.relu,
            original_resnet.maxpool,
            original_resnet.layer1,
            nn.ReLU(inplace=False),
            original_resnet.layer2,
            nn.ReLU(inplace=False),
            original_resnet.layer3,
            nn.ReLU(inplace=False),
            #original_resnet.layer4,
            #nn.ReLU(inplace=False),
        )

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=256, out_channels=reduced_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=reduced_channels),
            nn.ReLU(inplace=False),
        )
        #self.reduce =nn.Sequential(nn.Conv2d(256, reduced_channels, kernel_size=3,padding=1),nn.BatchNorm2d(reduced_channels),nn.ReLU(inplace=False))

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
#dataset = STL10(root='/data', download=True, transform=transform)
#dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

#split = int(0.9 * len(dataset))
#train_dataset, val_dataset = random_split(dataset, [split, len(dataset) - split])
train_loader = DataLoader(torch.load(os.path.join("dataset", "train_dataset.pth"), weights_only=False), batch_size=128, shuffle=True)
val_loader = DataLoader(torch.load(os.path.join("dataset", "val_dataset.pth"), weights_only=False), batch_size=128, shuffle=False)
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

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, reducedFeatures):

    device = torch.device("cuda" )
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

        #print(f"Epoch {epoch + 1}, Loss: {train_loss}")

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        correctVal = 0
        correctTrain = 0
        total = 0
        totalTrain = 0

        with torch.no_grad():

            for train_inputs, train_labels in train_loader:
                train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
                train_outputs = model(train_inputs)

                # Accuracy (if classification)
                _, predictedTrain = torch.max(train_outputs, 1)
                correctTrain += (predictedTrain == train_labels).sum().item()
                totalTrain += train_labels.size(0)

            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()

                # Accuracy (if classification)
                _, predicted = torch.max(val_outputs, 1)
                correctVal += (predicted == val_labels).sum().item()
                total += val_labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = 100 * correctVal / total
        train_accuracy = 100 * correctTrain / totalTrain


        if epoch % 1 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy

            }, os.path.join('newCheckpoints', f'checkpoint_ep{epoch}_fullResnetMod{reducedFeatures}.pth'))

        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")

    return model


epochs = int(sys.argv[1]); fmaps = int(sys.argv[2])
train(resnet50, train_loader, val_loader, optimizer, criterion, epochs, fmaps)

#resnet50_checkpoint_data = torch.load("Checkpoints/checkpoint_ep20.pth", weights_only=True)
#resnet50 = evaluation_from_checkpoint(resnet50, resnet50_checkpoint_data)

#resnet50.eval()