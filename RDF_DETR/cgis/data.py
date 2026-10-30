"""Standalone, immutable pseudo-reference import and dictionary letterbox dataset."""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import math
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch

from .common import CATEGORIES, SEED, read_json, sha256, write_json

IMAGE_SIZE = 896
EXPECTED = {'train': 9275, 'val': 1159, 'test': 1159}
PILOT = {'train': 500, 'val': 100, 'test': 0}


def _profile(profile):
    if profile not in ('pilot', 'full'):
        raise ValueError(f'Unknown profile: {profile}')
    return profile


def _split_names(root, split, profile):
    _profile(profile)
    if split not in EXPECTED:
        raise ValueError(f'Unknown split: {split}')
    path = Path(root) / 'data' / 'splits' / profile / f'{split}.txt'
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def select_stratified(images, annotations, count, seed=SEED):
    """Deterministic round-robin over width/density quantile cells."""
    if count > len(images):
        raise ValueError('Subset larger than source split')
    if count == 0:
        return []
    density = Counter(a['image_id'] for a in annotations if not a.get('ignore', False))
    rows = sorted(images, key=lambda x: x['file_name'])
    widths = np.quantile([i['width'] for i in rows], [.25, .5, .75])
    densities = np.quantile([density[i['id']] for i in rows], [.25, .5, .75])
    groups = defaultdict(list)
    for item in rows:
        key = (int(np.searchsorted(widths, item['width'], side='right')),
               int(np.searchsorted(densities, density[item['id']], side='right')))
        groups[key].append(item['file_name'])
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups)
    rng.shuffle(keys)
    chosen = []
    while len(chosen) < count:
        for key in keys:
            if groups[key]:
                chosen.append(groups[key].pop())
                if len(chosen) == count:
                    break
    chosen_set = set(chosen)
    # Original deterministic source order is useful for batch-level resume checks.
    return [i['file_name'] for i in images if i['file_name'] in chosen_set]


def prepare_data(root, profile='pilot'):
    """Check the externally supplied dataset and manifests."""
    root = Path(root).resolve()
    _profile(profile)
    manifest_path = root / 'data' / 'import_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('Missing data/import_manifest.json; see README.md for required external resources')
    manifest = read_json(manifest_path)
    for relative, digest in manifest['files'].items():
        path = root / relative
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f'Data file missing or changed: {relative}')
    pngs = list((root / 'data' / 'images').glob('*.png'))
    if len(pngs) != 11593:
        raise RuntimeError(f'Expected 11593 images, found {len(pngs)}')
    names = {s: _split_names(root, s, profile) for s in EXPECTED}
    all_full = {s: set(_split_names(root, s, 'full')) for s in EXPECTED}
    if any(all_full[a] & all_full[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise RuntimeError('Split leakage: image names overlap')
    for split, selected in names.items():
        expected = (PILOT if profile == 'pilot' else EXPECTED)[split]
        if len(selected) != expected or len(set(selected)) != len(selected):
            raise RuntimeError(f'Invalid {profile}/{split} count')
        if not set(selected) <= all_full[split]:
            raise RuntimeError(f'{profile}/{split} contains images outside its frozen source split')
    # Verify every image byte; server startup cost is small compared with training.
    for name, digest in manifest['image_sha256'].items():
        if sha256(root / 'data' / 'images' / name) != digest:
            raise RuntimeError(f'Image changed: {name}')
    return {
        'root': str(root), 'profile': profile, 'reference_type': 'pseudo',
        'splits': names, 'counts': {s: len(v) for s, v in names.items()},
        'split_files': {s: str(root / 'data' / 'splits' / profile / f'{s}.txt') for s in names},
        'annotations': {s: str(root / 'data' / 'derived' / f'{s}.json') for s in names},
        'manifest_sha256': sha256(manifest_path), 'seed': SEED,
    }


@lru_cache(maxsize=3)
def _read_derived(path, mtime_ns):
    return read_json(path)


def load_references(root, split='val', profile='pilot'):
    root = Path(root)
    names = set(_split_names(root, split, profile))
    path = root / 'data' / 'derived' / f'{split}.json'
    source = _read_derived(str(path.resolve()), path.stat().st_mtime_ns)
    images = [im.copy() for im in source['images'] if im['file_name'] in names]
    ids = {im['id'] for im in images}
    return {
        'schema_version': '1.0', 'reference_type': 'pseudo', 'split': split,
        'profile': profile, 'images': images,
        'annotations': [a.copy() for a in source['annotations'] if a['image_id'] in ids],
        'categories': source['categories'],
    }


def letterbox_geometry(width, height, image_size=IMAGE_SIZE):
    scale = min(image_size / width, image_size / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    pad_x, pad_y = (image_size - new_w) // 2, (image_size - new_h) // 2
    # The true per-axis scales include image resize rounding.
    return new_w, new_h, pad_x, pad_y, new_w / width, new_h / height


def normalize_boxes(boxes, geometry, image_size=IMAGE_SIZE):
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    if not len(boxes):
        return boxes
    _, _, px, py, sx, sy = geometry
    out = boxes.clone()
    out[:, 0] = (boxes[:, 0] + boxes[:, 2] / 2) * sx + px
    out[:, 1] = (boxes[:, 1] + boxes[:, 3] / 2) * sy + py
    out[:, 2] = boxes[:, 2] * sx
    out[:, 3] = boxes[:, 3] * sy
    return out / image_size


def restore_boxes(boxes, target, image_size=IMAGE_SIZE):
    """Normalized canvas cxcywh -> clipped original-image xywh; handles rounding."""
    boxes = torch.as_tensor(boxes).clone().reshape(-1, 4)
    if not len(boxes):
        return boxes
    oh, ow = [int(x) for x in target['orig_size']]
    vh, vw = [int(x) for x in target['valid_size']]
    px, py = [float(x) for x in target['pad']]
    sx, sy = vw / ow, vh / oh
    xy0 = boxes[:, :2] * image_size - boxes[:, 2:] * image_size / 2
    xy1 = boxes[:, :2] * image_size + boxes[:, 2:] * image_size / 2
    xy0[:, 0] = ((xy0[:, 0] - px) / sx).clamp(0, ow)
    xy0[:, 1] = ((xy0[:, 1] - py) / sy).clamp(0, oh)
    xy1[:, 0] = ((xy1[:, 0] - px) / sx).clamp(0, ow)
    xy1[:, 1] = ((xy1[:, 1] - py) / sy).clamp(0, oh)
    return torch.cat((xy0, (xy1 - xy0).clamp(min=0)), -1)


class DictionaryDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', profile='pilot', augment=False, role_file=None, image_size=IMAGE_SIZE):
        self.root, self.split, self.profile = Path(root), split, profile
        self.image_size, self.augment = image_size, augment
        self.references = load_references(root, split, profile)
        image_by_name = {im['file_name']: im for im in self.references['images']}
        self.images = [image_by_name[n] for n in _split_names(root, split, profile)]
        self.by_image = defaultdict(list)
        for a in self.references['annotations']:
            self.by_image[a['image_id']].append(a)
        self.role_enabled = [False] * 4
        self.role_pairs = {}
        if role_file:
            if split != 'train':
                raise ValueError('Role training labels must never be loaded into evaluation datasets')
            role = read_json(role_file)
            if role.get('profile') != profile or role.get('split') != 'train':
                raise ValueError('Role file split/profile mismatch')
            self.role_enabled = role['role_enabled']
            self.role_pairs = role['pairs_by_image']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        meta = self.images[index]
        with Image.open(self.root / 'data' / 'images' / meta['file_name']) as handle:
            image = handle.convert('RGB')
        width, height = image.size
        if (width, height) != (meta['width'], meta['height']):
            raise RuntimeError(f'Image dimension changed: {meta["file_name"]}')
        geometry = letterbox_geometry(width, height, self.image_size)
        new_w, new_h, px, py, sx, sy = geometry
        if self.augment:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(.9, 1.1))
            if random.random() < .2:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(.1, .5)))
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.
        if self.augment and random.random() < .15:
            pixels = np.clip(pixels + np.random.normal(0, .005, pixels.shape), 0, 1).astype(np.float32)
        canvas = torch.ones((3, self.image_size, self.image_size), dtype=torch.float32)
        canvas[:, py:py + new_h, px:px + new_w] = torch.from_numpy(pixels.copy()).permute(2, 0, 1)
        annotations = self.by_image[meta['id']]
        active = [a for a in annotations if not a.get('ignore', False)]
        ignored = [a for a in annotations if a.get('ignore', False)]
        pairs = [p for p in self.role_pairs.get(meta['file_name'], []) if self.role_enabled[p['label']]]
        target = {
            'boxes': normalize_boxes([a['bbox'] for a in active], geometry, self.image_size),
            'labels': torch.tensor([a['category_id'] - 1 for a in active], dtype=torch.int64),
            'weights': torch.tensor([a.get('pseudo_weight', 1.) for a in active], dtype=torch.float32),
            'ignore_boxes': normalize_boxes([a['bbox'] for a in ignored], geometry, self.image_size),
            'ignore_labels': torch.tensor([a['category_id'] - 1 for a in ignored], dtype=torch.int64),
            'image_id': torch.tensor(meta['id'], dtype=torch.int64),
            'orig_size': torch.tensor([height, width], dtype=torch.int64),
            'size': torch.tensor([self.image_size, self.image_size]),
            'valid_size': torch.tensor([new_h, new_w], dtype=torch.int64),
            'valid_box': torch.tensor([px, py, px + new_w, py + new_h], dtype=torch.float32),
            'scale': min(self.image_size / width, self.image_size / height),
            'scale_xy': torch.tensor([sx, sy], dtype=torch.float32), 'pad': [px, py],
            'file_name': meta['file_name'],
            'role_positive_boxes': normalize_boxes([p['positive_bbox'] for p in pairs], geometry, self.image_size),
            'role_negative_boxes': normalize_boxes([p['negative_bbox'] for p in pairs], geometry, self.image_size),
            'role_pair_labels': torch.tensor([p['label'] for p in pairs], dtype=torch.int64),
            'role_pair_weights': torch.tensor([p['weight'] for p in pairs], dtype=torch.float32),
        }
        return canvas, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
