from train_utils import train
from models import ModifiedResNet18
import torch
from loaders import train_loader, val_loader
import sys


if __name__ == "__main__":

    epochs = int(sys.argv[1])

    model = ModifiedResNet18(num_classes=10, reduced_channels=256)
    for name, param in model.named_parameters():
        if 'reduce' not in name and 'classifier' not in name:
            param.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(model, train_loader, val_loader, optimizer, criterion, num_epochs=epochs, device=dev, batch_size=32, learning_rate=0.001)