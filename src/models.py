"""Small CNN trained from scratch, and a frozen ImageNet ResNet-18."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

NUM_CLASSES = 5


class SmallCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, inputs):
        features = self.features(inputs)
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled)


def build_resnet(num_classes=NUM_CLASSES):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(name):
    if name == "cnn":
        return SmallCNN()
    if name == "resnet":
        return build_resnet()
    raise ValueError("model must be cnn or resnet.")


def trainable_parameters(model, name):
    if name == "resnet":
        return model.fc.parameters()
    if name == "cnn":
        return model.parameters()
    raise ValueError("model must be cnn or resnet.")


def set_model_mode(model, name, training):
    if name == "cnn":
        model.train(training)
        return
    if name == "resnet":
        model.eval()
        model.fc.train(training)
        return
    raise ValueError("model must be cnn or resnet.")


if __name__ == "__main__":
    sample = torch.zeros(2, 3, 224, 224)
    cnn = build_model("cnn")
    logits = cnn(sample)
    feature_shape = cnn.features(sample).shape
    trainable = sum(parameter.numel() for parameter in cnn.parameters() if parameter.requires_grad)
    print(f"cnn features {tuple(feature_shape)} logits {tuple(logits.shape)} trainable {trainable}")
