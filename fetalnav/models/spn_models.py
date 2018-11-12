import torch
import torch.nn as nn

from spn.modules import SoftProposal, SpatialSumOverMap

from .resnet import *
from .vgg import *
from .alexnet import *


class SPNetWSL(nn.Module):
    def __init__(self, model, num_classes, num_features, num_maps):
        super(SPNetWSL, self).__init__()

        self.features = nn.Sequential(*list(model.features.children())[:-1])
        self.spatial_poolings = nn.ModuleList([self.get_pooling(num_features, num_maps) for i in range(num_classes)])

        # classification layer
        self.classifiers = nn.ModuleList([nn.Linear(num_maps, 1) for i in range(num_classes)])

    def get_pooling(self, num_features, num_maps):
        pooling = nn.Sequential()
        pooling.add_module('adconv',
                           nn.Conv2d(num_features, num_maps, kernel_size=3, stride=1, padding=1, groups=2, bias=True))
        # num_maps=num_features
        pooling.add_module('maps', nn.ReLU())
        pooling.add_module('sp', SoftProposal(couple=True, factor=None))
        pooling.add_module('sum', SpatialSumOverMap())
        return pooling

    def forward(self, x):
        x = self.features(x)
        xs = [p(x) for p in self.spatial_poolings]
        xs = [xsi.view(xsi.size(0), -1) for xsi in xs]
        xs = [c(xs[idx]) for idx, c in enumerate(self.classifiers)]
        return torch.stack(xs, dim=1).squeeze()


def resnet18_sp(num_classes, num_maps=1024, **kwargs):
    model = resnet18(pretrained=False, **kwargs)
    num_features = list(list(model.features.children())[-2][1].children())[3].out_channels
    return SPNetWSL(model, num_classes, num_features, num_maps)


def resnet34_sp(num_classes, num_maps=1024, **kwargs):
    model = resnet34(pretrained=False, **kwargs)
    num_features = list(list(model.features.children())[-2][1].children())[3].out_channels
    return SPNetWSL(model, num_classes, num_features, num_maps)


def vgg11_sp(num_classes, batch_norm=False, num_maps=1024, **kwargs):
    model = vgg11(pretrained=False, batch_norm=batch_norm, **kwargs)
    num_features = model.features[25 if batch_norm else 18].out_channels
    return SPNetWSL(model, num_classes, num_features, num_maps)


def vgg13_sp(num_classes, batch_norm=False, num_maps=1024, **kwargs):
    model = vgg13(pretrained=False, batch_norm=batch_norm, **kwargs)
    num_features = model.features[31 if batch_norm else 22].out_channels
    return SPNetWSL(model, num_classes, num_features, num_maps)


def vgg16_sp(num_classes, batch_norm=False, num_maps=1024, **kwargs):
    model = vgg16(pretrained=False, batch_norm=batch_norm, **kwargs)
    num_features = model.features[40 if batch_norm else 28].out_channels
    return SPNetWSL(model, num_classes, num_features, num_maps)


def alexnet_sp(num_classes, num_maps=1024, **kwargs):
    model = alexnet(pretrained=False, **kwargs)
    num_features = model.features[10].out_channels
    return SPNetWSL(model, num_classes, num_features, num_maps)
