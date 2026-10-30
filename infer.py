"""Predict leaf-tip xy coordinates and overlays; full-image or owned-overlap tiles."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from runtime import resolve_device
from PIL import Image, ImageDraw

from dataset import EXTENSIONS, image_tensor, pad_tensor
from engine import select_points
from evaluate import load_model
from progress import progress_bar


def tile_axis(length, tile_size, overlap):
    if not tile_size or length <= tile_size:
        return [(0, length, 0., float(length))]
    if not 0 <= overlap < tile_size:
        raise ValueError('overlap must satisfy 0 <= overlap < tile-size')
    starts = list(range(0, length-tile_size+1, tile_size-overlap))
    if starts[-1] != length-tile_size:
        starts.append(length-tile_size)
    bounds = [0.] + [(left+tile_size+right)/2 for left,right in zip(starts, starts[1:])] + [float(length)]
    return [(start, start+tile_size, bounds[i], bounds[i+1]) for i,start in enumerate(starts)]


@torch.inference_mode()
def predict(model, image, device, threshold=.5, tile_size=0, overlap=64):
    width, height = image.size
    xs = tile_axis(width, tile_size, overlap)
    ys = tile_axis(height, tile_size, overlap)
    final = []
    for top,bottom,own_top,own_bottom in ys:
        for left,right,own_left,own_right in xs:
            crop = image.crop((left,top,right,bottom))
            tensor = pad_tensor(image_tensor(crop)).unsqueeze(0).to(device)
            output = model(tensor)
            points, scores = select_points(output, 0, (bottom-top,right-left), threshold)
            points = points.cpu().numpy() + np.asarray([left,top])
            scores = scores.cpu().numpy()
            keep = ((points[:,0] >= own_left) & (points[:,0] < own_right) &
                    (points[:,1] >= own_top) & (points[:,1] < own_bottom))
            final.extend((float(x),float(y),float(score)) for (x,y),score in zip(points[keep],scores[keep]))
    return final


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True, help='RGB image or image directory')
    p.add_argument('--output', type=Path, default=Path('predictions'))
    p.add_argument('--device', default='cuda')
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--tile-size', type=int, default=0, help='0: full-image inference; e.g. 1024 for large images')
    p.add_argument('--overlap', type=int, default=64)
    args = p.parse_args()
    if not 0 < args.threshold < 1 or args.tile_size < 0 or not 0 <= args.overlap < (args.tile_size or 10**9):
        p.error('Invalid threshold, tile size or overlap')
    device = resolve_device(args.device)
    model, epoch = load_model(args.checkpoint, device)
    paths = sorted(path for path in args.input.iterdir() if path.suffix.lower() in EXTENSIONS) if args.input.is_dir() else [args.input]
    if not paths:
        raise ValueError('No input images found')
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError('Duplicate image stems would overwrite predictions')
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with progress_bar(paths, desc='Inference', unit='image') as bar:
        for path in bar:
            with Image.open(path) as source:
                image = source.convert('RGB')
            points = predict(model,image,device,args.threshold,args.tile_size,args.overlap)
            (args.output/f'{path.stem}.txt').write_text(''.join(f'{x:.3f} {y:.3f}\n' for x,y,_ in points))
            (args.output/f'{path.stem}.json').write_text(json.dumps(dict(
                image=str(path), count=len(points), width=image.width,height=image.height,
                coordinate_order='xy', points=[dict(x=x,y=y,score=s) for x,y,s in points]),indent=2))
            draw = ImageDraw.Draw(image)
            for x,y,score in points:
                draw.ellipse((x-3,y-3,x+3,y+3),fill='red')
            draw.text((8,8),f'Count: {len(points)}',fill='red',stroke_width=1)
            image.save(args.output/f'{path.stem}_pred.jpg',quality=95)
            rows.append(dict(image_id=path.stem,pred_count=len(points)))
    with (args.output/'counts.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['image_id','pred_count'])
        writer.writeheader();writer.writerows(rows)


if __name__ == '__main__':
    main()
