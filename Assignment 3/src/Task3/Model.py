import torchvision
from torch import nn


class ResNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512*4,10)

        for param in self.model.parameters():  # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters():  # Unfreeze the last fully - connected
            param.requires_grad = True  # layer
        for param in self.model.layer4.parameters():  # Unfreeze the last 5 c on vo lu ti on al
            param.requires_grad = True  # layers

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=8)
        x = self.model(x)
        return x