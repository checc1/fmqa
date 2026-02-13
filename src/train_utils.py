import torch
from torchvision import datasets, transforms
from torchvision.datasets import OxfordIIITPet, STL10, Caltech101
import torchvision.models as models
import torch.nn as nn
from torch.autograd import gradcheck
from warnings import filterwarnings
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import trange
import os


def evaluation_from_checkpoint(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    return model


def train(model, train_loader, val_loader, optimizer, criterion, batch_size=32, learning_rate=0.001, num_epochs=300):
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