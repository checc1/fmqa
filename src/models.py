import torch
import torchvision.models as models
import torch.nn as nn



class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes=10, reduced_channels=256):
        super(ModifiedResNet18, self).__init__()
        #original_resnet = models.resnet18(pretrained=True)
        original_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Keep all layers up to layer3 (optional for now)
        self.features = nn.Sequential(
            original_resnet.conv1,
            original_resnet.bn1,
            original_resnet.relu,
            original_resnet.maxpool,
            original_resnet.layer1,
            original_resnet.layer2,
            original_resnet.layer3,
            #original_resnet.layer4
        )

        # Reduce number of output features from 512 to 256 (activation block for the analysis)
        self.reduce =nn.Sequential( nn.Conv2d(256, reduced_channels, kernel_size=3,padding=1), nn.ReLU(inplace=False),nn.BatchNorm2d(reduced_channels))

        # Classifier

        self.classifier = nn.Linear(reduced_channels*14**2, num_classes) # we have a better resolution of 14x14 after the last conv layer

    def forward(self, x):
        x = self.features(x)
        x = self.reduce(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x