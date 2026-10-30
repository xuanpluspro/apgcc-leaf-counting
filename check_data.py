"""Read every image/point pair before training; optional decoded-image leakage check."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from dataset import RiceW3999, read_points


def audit(root, check_overlap=False):
    reports, ids, image_hashes = {}, {}, {}
    for split in ('train','val','test'):
        if not (root/'ImageSets'/f'{split}.txt').is_file():
            continue
        dataset = RiceW3999(root,split,augment=False)
        ids[split] = set(dataset.ids)
        counts, sizes = [], {}
        for name,path,annotation in dataset.samples:
            with Image.open(path) as source:
                image = source.convert('RGB')
                points = read_points(annotation,*image.size)
                size = f'{image.width}x{image.height}'
                sizes[size] = sizes.get(size,0) + 1
                counts.append(len(points))
                if check_overlap:
                    digest=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
                    if digest in image_hashes:
                        previous=image_hashes[digest]
                        raise ValueError(f'Duplicate decoded image: {previous} vs {(split,name)}')
                    image_hashes[digest]=(split,name)
        reports[split]=dict(images=len(dataset),points=sum(counts),empty_images=counts.count(0),
                            mean_points=sum(counts)/len(counts),max_points=max(counts),sizes=sizes)
    if 'train' not in reports or 'val' not in reports:
        raise ValueError('Both train.txt and val.txt are required')
    for left in ids:
        for right in ids:
            if left < right and ids[left] & ids[right]:
                raise ValueError(f'{left}/{right} IDs overlap')
    return reports


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--data-path',type=Path,required=True)
    p.add_argument('--check-overlap',action='store_true',help='Also hash decoded image pixels; slower')
    args=p.parse_args()
    print(json.dumps(audit(args.data_path,args.check_overlap),indent=2))
