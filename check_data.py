"""Fast image-header/point checks; optional full decoding and pixel-duplicate audit."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from dataset import RiceW3999, read_points
from progress import progress_bar


def audit(root, check_overlap=False, *, decode=False, show_progress=True):
    root = Path(root).expanduser().resolve()
    reports, ids, image_hashes = {}, {}, {}
    datasets = {}
    for split in ('train', 'val', 'test'):
        if (root/'ImageSets'/f'{split}.txt').is_file():
            datasets[split] = RiceW3999(root, split, augment=False)
            ids[split] = set(datasets[split].ids)
    if 'train' not in datasets or 'val' not in datasets:
        raise ValueError('Both train.txt and val.txt are required')
    # Detect cheap manifest errors before reading thousands of image headers.
    for left in ids:
        for right in ids:
            if left < right and ids[left] & ids[right]:
                raise ValueError(f'{left}/{right} IDs overlap')
    for split, dataset in datasets.items():
        counts, sizes = [], {}
        mode = 'pixels' if check_overlap else 'decode' if decode else 'headers'
        # JSON stays on stdout; progress.py writes transient bars to stderr/TTY.
        if show_progress:
            context = progress_bar(dataset.samples, desc=f'Check {split} ({mode})', unit='image')
        else:
            from contextlib import nullcontext
            context = nullcontext(dataset.samples)
        with context as samples:
            for name, path, annotation in samples:
                try:
                    with Image.open(path) as source:
                        # Image.open is lazy for standard JPEG/PNG: read dimensions
                        # without loading pixels or converting every image to RGB.
                        width, height = source.size
                        points = read_points(annotation, width, height)
                        if decode or check_overlap:
                            source.load()
                        if check_overlap:
                            image = source.convert('RGB')
                            digest = hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
                            if digest in image_hashes:
                                previous = image_hashes[digest]
                                raise ValueError(f'Duplicate decoded image: {previous} vs {(split,name)}')
                            image_hashes[digest] = (split, name)
                except (OSError, ValueError) as error:
                    raise ValueError(f'{split}/{name}: {error}') from error
                size = f'{width}x{height}'
                sizes[size] = sizes.get(size, 0) + 1
                counts.append(len(points))
        reports[split] = dict(images=len(dataset), points=sum(counts), empty_images=counts.count(0),
                             mean_points=sum(counts)/len(counts), max_points=max(counts), sizes=sizes)
    return reports


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data-path', type=Path, required=True)
    p.add_argument('--decode', action='store_true', help='Fully decode images to detect pixel-data corruption; slower')
    p.add_argument('--check-overlap', action='store_true', help='Fully decode and hash pixels to detect duplicates; slower')
    p.add_argument('--no-progress', action='store_true', help='Disable terminal progress bars')
    args = p.parse_args()
    print(json.dumps(audit(args.data_path, args.check_overlap, decode=args.decode,
                           show_progress=not args.no_progress), indent=2))
