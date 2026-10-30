"""APGCC leaf-counting reproduction. See REPRODUCTION.md for deviations."""
from dataclasses import asdict, dataclass

from torch import nn

from .decoder import IFI_Decoder_Model
from .encoder import VGG16BN


@dataclass
class ModelConfig:
    inner_planes: int = 64
    head_layers: tuple = (1024, 512, 256, 256)
    pos_dim: int = 32
    aux_number: tuple = (2, 2)
    aux_range: tuple = (2., 8.)
    aux_enabled: bool = True

    def to_dict(self):
        return asdict(self)


class APGCC(nn.Module):
    def __init__(self, config=None, pretrained=None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        if (len(cfg.aux_number) != 2 or min(cfg.aux_number) < 0 or
                not 0 < cfg.aux_range[0] < cfg.aux_range[1]):
            raise ValueError('Invalid auxiliary point configuration')
        self.encoder = VGG16BN(pretrained)
        self.decoder = IFI_Decoder_Model(
            in_planes=[128, 256, 512, 512], feat_layers=[3, 4], inner_planes=cfg.inner_planes,
            head_layers=cfg.head_layers, ultra_pe=True, pos_dim=cfg.pos_dim,
            local=True, require_grad=True, num_anchor_points=4, row=2, line=2,
            AUX_EN=cfg.aux_enabled, AUX_NUMBER=cfg.aux_number, AUX_RANGE=cfg.aux_range,
            AUX_kwargs=dict(pos_coef=1., neg_coef=1., pos_loc=.0002, neg_loc=.0002))

    def forward(self, images, targets=None):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError('Expected Bx3xHxW RGB input')
        if min(images.shape[-2:]) < 32 or any(n % 16 for n in images.shape[-2:]):
            raise ValueError('Pad input height and width to multiples of 16, with minimum 32')
        return self.decoder(images, self.encoder(images), targets)
