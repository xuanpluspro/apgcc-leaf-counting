# Sources and attribution

- `models/modules.py` and `models/decoder.py` are adapted from
  [AaronCIH/APGCC](https://github.com/AaronCIH/APGCC), commit
  `e3e997bf592a70fd34233a432a974200d8c0c847`, MIT License.
  Original comments attributing P2PNet/Tencent, DETR/Meta and other module sources are retained.
  Local changes include device portability, four-corner accumulation repair, arbitrary-point
  IFI querying and reconstructed auxiliary supervision. See REPRODUCTION.md.
- `models/encoder.py` implements the standard VGG16-BN architecture and the feature slices
  used in APGCC. Pretrained files are supplied by the user and are not redistributed here.
- Matching and loss design follow APGCC/P2PNet; implementation is adapted for per-image
  memory use and empty annotations rather than copying the broken public training loop.
- `counting_metrics.py` and `progress.py` reuse the common protocol/helpers already used in
  the user's [PET](https://github.com/xuanpluspro/PET) and
  [P2RLoss](https://github.com/xuanpluspro/p2rloss) adapters.
  Coordinates are converted explicitly at the APGCC evaluation boundary.
- Training, data adapter, checkpointing, inference, benchmark and tests are the leaf-counting
  integration provided in this repository. They are not an official APGCC training release.

The upstream MIT notice is included in LICENSE. This repository does not redistribute
datasets, pretrained weights or trained checkpoints. Cite the APGCC paper when using its method.
