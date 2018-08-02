import torch.nn as nn

from spn.modules import SoftProposal, SpatialSumOverMap

from .resnet import *
from .vgg import *
from .alexnet import *


class SPNetWSL(nn.Module):
    def __init__(self, model, num_classes, num_maps, pooling):
        super(SPNetWSL, self).__init__()

        self.features = nn.Sequential(*list(model.features.children())[:-1])
        self.spatial_pooling = pooling

        # classification layer
        self.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(num_maps, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.spatial_pooling(x)
        x = x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def resnet18_sp(num_classes, num_maps=1024, **kwargs):
    model = resnet18(pretrained=False, **kwargs)
    num_features = list(list(model.features.children())[-2][1].children())[3].out_channels
    pooling = nn.Sequential()
    pooling.add_module('adconv', nn.Conv2d(num_features, num_maps, kernel_size=3, stride=1, padding=1, groups=2, bias=True))
    # num_maps=num_features
    pooling.add_module('maps', nn.ReLU())
    pooling.add_module('sp', SoftProposal())
    pooling.add_module('sum', SpatialSumOverMap())
    return SPNetWSL(model, num_classes, num_maps, pooling)


def resnet34_sp(num_classes, num_maps=1024, **kwargs):
    model = resnet34(pretrained=False, **kwargs)
    num_features = list(list(model.features.children())[-2][1].children())[3].out_channels
    pooling = nn.Sequential()
    pooling.add_module('adconv', nn.Conv2d(num_features, num_maps, kernel_size=3, stride=1, padding=1, groups=2, bias=True))
    # num_maps=num_features
    pooling.add_module('maps', nn.ReLU())
    pooling.add_module('sp', SoftProposal())
    pooling.add_module('sum', SpatialSumOverMap())
    return SPNetWSL(model, num_classes, num_maps, pooling)


def vgg11_sp(num_classes, batch_norm=False, num_maps=1024, **kwargs):
    model = vgg11(pretrained=False, batch_norm=batch_norm, **kwargs)
    num_features = model.features[25 if batch_norm else 18].out_channels
    pooling = nn.Sequential()
    pooling.add_module('adconv', nn.Conv2d(num_features, num_maps, kernel_size=3, stride=1, padding=1, groups=2, bias=True))
    # num_maps=num_features
    pooling.add_module('maps', nn.ReLU())
    pooling.add_module('sp', SoftProposal())
    pooling.add_module('sum', SpatialSumOverMap())
    return SPNetWSL(model, num_classes, num_maps, pooling)


def vgg13_sp(num_classes, batch_norm=False, num_maps=1024, **kwargs):
    model = vgg13(pretrained=False, batch_norm=batch_norm, **kwargs)
    num_features = model.features[31 if batch_norm else 22].out_channels
    pooling = nn.Sequential()
    pooling.add_module('adconv', nn.Conv2d(num_features, num_maps, kernel_size=3, stride=1, padding=1, groups=2, bias=True))
    # num_maps=num_features
    pooling.add_module('maps', nn.ReLU())
    pooling.add_module('sp', SoftProposal())
    pooling.add_module('sum', SpatialSumOverMap())
    return SPNetWSL(model, num_classes, num_maps, pooling)


def vgg16_sp(num_classes, batch_norm=False, num_maps=1024, **kwargs):
    model = vgg16(pretrained=False, batch_norm=batch_norm, **kwargs)
    num_features = model.features[40 if batch_norm else 28].out_channels
    pooling = nn.Sequential()
    pooling.add_module('adconv', nn.Conv2d(num_features, num_maps, kernel_size=3, stride=1, padding=1, groups=2, bias=True))
    # num_maps=num_features
    pooling.add_module('maps', nn.ReLU())
    pooling.add_module('sp', SoftProposal())
    pooling.add_module('sum', SpatialSumOverMap())
    return SPNetWSL(model, num_classes, num_maps, pooling)


def alexnet_sp(num_classes, num_maps=1024, **kwargs):
    model = alexnet(pretrained=False, **kwargs)
    num_features = model.features[10].out_channels
    pooling = nn.Sequential()
    pooling.add_module('adconv', nn.Conv2d(num_features, num_maps, kernel_size=3, stride=1, padding=1, groups=2, bias=True))
    pooling.add_module('maps', nn.ReLU())
    pooling.add_module('sp', SoftProposal())
    pooling.add_module('sum', SpatialSumOverMap())
    return SPNetWSL(model, num_classes, num_maps, pooling)
