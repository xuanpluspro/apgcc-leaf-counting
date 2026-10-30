"""RiceW3999: VOC-like folders, one complete image ID per UTF-8 line."""
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
MEAN = torch.tensor([.485, .456, .406]).view(3, 1, 1)
STD = torch.tensor([.229, .224, .225]).view(3, 1, 1)


def image_tensor(image):
    array = np.array(image.convert('RGB'), dtype=np.float32, copy=True) / 255.
    return (torch.from_numpy(array).permute(2, 0, 1) - MEAN) / STD


def pad_tensor(image):
    height, width = image.shape[-2:]
    # Zero in normalized space is ImageNet-mean RGB. Original dimensions
    # are carried separately, and predictions outside them are discarded.
    return F.pad(image, (0, (-width) % 16, 0, (-height) % 16))


def read_points(path, width, height):
    points = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f'{path}:{line_number}: expected exactly two pixel coordinates x y')
        x, y = map(float, fields)
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
            raise ValueError(f'{path}:{line_number}: point {(x,y)} outside {width}x{height}')
        points.append((x, y))
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


class RiceW3999(Dataset):
    def __init__(self, root, split='train', patch_size=512, augment=True):
        self.root, self.split = Path(root).expanduser().resolve(), split
        self.training = split == 'train' and augment
        self.patch_size = patch_size
        if patch_size < 32 or patch_size % 16:
            raise ValueError('patch_size must be >=32 and a multiple of 16')
        self.manifest = self.root / 'ImageSets' / f'{split}.txt'
        ids = [line.strip() for line in self.manifest.read_text(encoding='utf-8-sig').splitlines()
               if line.strip()]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f'Empty or duplicate IDs in {self.manifest}')
        images = {}
        for path in (self.root / 'JPEGImages').iterdir():
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                if path.stem in images:
                    raise ValueError(f'Ambiguous image ID {path.stem}; multiple extensions')
                images[path.stem] = path
        self.samples = []
        self.ids = ids
        for name in ids:
            if '/' in name or '\\' in name or name in ('.', '..'):
                raise ValueError(f'Expected a filename stem, got {name!r}')
            annotation = self.root / 'PointAnnotations' / f'{name}.txt'
            if name not in images or not annotation.is_file():
                raise FileNotFoundError(f'Missing image or annotation for {name!r}')
            self.samples.append((name, images[name], annotation))

    def fingerprint(self):
        """Hash IDs, annotation bytes, image sizes/mtimes to detect changed inputs.

        Image pixels are not hashed here; use --check-data --check-overlap when
        auditing a split. Moving files may conservatively invalidate this hash.
        """
        digest = hashlib.sha256()
        for name, image, annotation in self.samples:
            stat = image.stat()
            digest.update(json.dumps([name, stat.st_size, stat.st_mtime_ns]).encode())
            digest.update(annotation.read_bytes())
        return digest.hexdigest()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name, image_path, annotation = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        points = read_points(annotation, *image.size)
        if self.training:
            # One online patch per source image per visit. Resize coordinates
            # with actual rounded scale factors, rather than nominal scale.
            width, height = image.size
            scale = random.uniform(.7, 1.3)
            scale = max(scale, self.patch_size / min(width, height))
            new_width = max(self.patch_size, round(width * scale))
            new_height = max(self.patch_size, round(height * scale))
            image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
            points *= np.asarray([new_width / width, new_height / height], dtype=np.float32)
            left = random.randint(0, new_width - self.patch_size)
            top = random.randint(0, new_height - self.patch_size)
            image = image.crop((left, top, left + self.patch_size, top + self.patch_size))
            points -= np.asarray([left, top], dtype=np.float32)
            valid = ((points[:, 0] >= 0) & (points[:, 0] < self.patch_size) &
                     (points[:, 1] >= 0) & (points[:, 1] < self.patch_size))
            points = points[valid].copy()
            if random.random() < .5:
                image = ImageOps.mirror(image)
                points[:, 0] = self.patch_size - 1 - points[:, 0]
                # Pixel-centre annotations may be fractional near the edge.
                points[:, 0] = np.clip(points[:, 0], 0, self.patch_size - 1)
        width, height = image.size
        tensor = pad_tensor(image_tensor(image))
        return tensor, dict(points=torch.from_numpy(points.copy()), image_id=name,
                            size=(height, width), path=str(image_path))


def collate(batch):
    images, targets = zip(*batch)
    if len({tuple(image.shape) for image in images}) != 1:
        raise ValueError('Mixed full-image sizes require validation batch_size=1')
    return torch.stack(images), list(targets)


def seed_worker(worker_id):
    value = torch.initial_seed() % (2 ** 32)
    random.seed(value)
    np.random.seed(value)
