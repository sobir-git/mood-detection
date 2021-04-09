from torch import nn
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.models import mobilenet


class FERModel(nn.Module):
    """Facial expression recognition model."""

    def __init__(self, n_classes=7):
        super().__init__()
        self.net = mobilenet.MobileNetV2(1)
        self.reduction = nn.Linear(1280, 64, bias=False)
        self.reduction.weight.requires_grad = False
        self.output = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.embed(x)
        out = self.output(x)
        return out

    def embed(self, x):
        x = self.net.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
        x = self.reduction(x)
        return x


class MoodModel(nn.Module):
    """Mood level regression model."""

    def __init__(self, net: FERModel = None):
        super().__init__()
        if net is None:
            net = FERModel(7)
        self.net = net
        self.reduction = nn.Conv2d(1280, 16, 1, bias=False)
        self.reduction.weight.requires_grad = False
        self.output = nn.Linear(16, 1)
        self.drop = nn.Dropout()

    def forward(self, x):
        x = self.net.net.features(x)
        x = self.reduction(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
        x = self.drop(x)
        x = self.output(x)
        return x
