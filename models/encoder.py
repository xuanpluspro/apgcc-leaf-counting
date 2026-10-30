"""VGG16-BN feature stages matching APGCC's public Encoder.py slicing.

Stage weights come from a trusted local torchvision ImageNet checkpoint.
No downloads and no allocation of VGG's unused 123M-parameter classifier.
"""
from pathlib import Path

import torch
from torch import nn


class VGG16BN(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        layers, channels = [], 3
        for width in (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                      512, 512, 512, 'M', 512, 512, 512, 'M'):
            if width == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.extend([nn.Conv2d(channels, width, 3, padding=1),
                               nn.BatchNorm2d(width), nn.ReLU(inplace=True)])
                channels = width
        features = nn.Sequential(*layers)
        for module in features.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(module.bias)
        if pretrained:
            path = Path(pretrained)
            if not path.is_file():
                raise FileNotFoundError(f'Missing VGG16-BN weights: {path}')
            state = torch.load(path, map_location='cpu', weights_only=True)
            state = {key.removeprefix('features.'): value for key, value in state.items()
                     if key.startswith('features.')}
            features.load_state_dict(state, strict=True)
        children = list(features.children())
        self.body1 = nn.Sequential(*children[:13])
        self.body2 = nn.Sequential(*children[13:23])
        self.body3 = nn.Sequential(*children[23:33])
        self.body4 = nn.Sequential(*children[33:43])

    def forward(self, images):
        result = []
        for body in (self.body1, self.body2, self.body3, self.body4):
            images = body(images)
            result.append(images)
        return result
