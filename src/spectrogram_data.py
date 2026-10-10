"""Deterministic, patient-disjoint spectrogram data and exact latent statistics."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_manifest(path):
    path = Path(path)
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Manifest is empty')
    required = {'path', 'patient_id', 'split'}
    if not required.issubset(rows[0]):
        raise ValueError(f'Manifest requires {sorted(required)}')
    seen, patients = set(), {}
    for row in rows:
        rel = Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts or not row['patient_id']:
            raise ValueError('Paths must be relative without ..; patient IDs must be nonempty')
        if row['split'] not in {'train', 'val', 'test'}:
            raise ValueError('Split must be train, val, or test')
        if row['path'] in seen:
            raise ValueError(f"Duplicate sample: {row['path']}")
        seen.add(row['path'])
        previous = patients.setdefault(row['patient_id'], row['split'])
        if previous != row['split']:
            raise ValueError(f"Patient leakage: {row['patient_id']}")
    caches = [str(Path(r['path']).with_suffix('.npy')) for r in rows]
    if len(set(caches)) != len(caches):
        raise ValueError('Cache filename collision')
    return rows


def prepare_manifest(root, output, seed=42, layout='auto'):
    """Preserve existing split folders, or assign splits to patient-root datasets."""
    root, output = Path(root), Path(output)
    split_names = ('train', 'val', 'test')
    if layout not in {'auto', 'presplit', 'patient-folders'}:
        raise ValueError('Unknown dataset layout')
    present = [name for name in split_names if (root / name).is_dir()]
    if layout == 'auto':
        layout = 'presplit' if present else 'patient-folders'
    rows = []

    def pngs(folder):
        return sorted(p for p in folder.rglob('*')
                      if p.is_file() and p.suffix.lower() == '.png')

    if layout == 'presplit':
        if len(present) != 3:
            raise ValueError('Presplit layout requires train, val, and test directories')
        patients = {}
        # Only scan these directories: root-level documents and other folders are ignored.
        for split in split_names:
            paths = pngs(root / split)
            if not paths:
                raise ValueError(f'No PNG images found in {split} split')
            for path in paths:
                rel = path.relative_to(root)
                if len(rel.parts) < 3:
                    raise ValueError('Expected DATA_ROOT/split/patient_id/[optional subfolders]/image.png')
                patient = rel.parts[1]
                previous = patients.setdefault(patient, split)
                if previous != split:
                    raise ValueError(f'Patient leakage: {patient} appears in {previous} and {split}')
                rows.append(dict(path=rel.as_posix(), patient_id=patient, split=split, role='target'))
    else:
        paths = pngs(root)
        if not paths:
            raise ValueError('No PNG images found')
        if any(len(p.relative_to(root).parts) < 2 for p in paths):
            raise ValueError('Expected DATA_ROOT/patient_id/[optional subfolders]/image.png')
        patients = sorted({p.relative_to(root).parts[0] for p in paths})
        if len(patients) < 3:
            raise ValueError('At least three patients are required for train/val/test')
        patients.sort(key=lambda p: hashlib.sha256(f'{seed}:{p}'.encode()).hexdigest())
        n_val = max(1, round(len(patients) * .15))
        n_test = max(1, round(len(patients) * .15))
        splits = {p: ('val' if i < n_val else 'test' if i < n_val + n_test else 'train')
                  for i, p in enumerate(patients)}
        for path in paths:
            rel = path.relative_to(root)
            patient = rel.parts[0]
            rows.append(dict(path=rel.as_posix(), patient_id=patient, split=splits[patient], role='target'))
    # Validate all assignments before creating the CSV.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'patient_id', 'split', 'role'])
        writer.writeheader()
        writer.writerows(rows)
    return read_manifest(output)


class SpectrogramDataset(Dataset):
    def __init__(self, root, rows):
        self.root, self.rows = Path(root), rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['path']) as im:
            if im.size != (256, 256):
                raise ValueError(f"Expected 256x256: {row['path']}, got {im.size}")
            if im.mode not in {'L', 'RGB'}:
                raise ValueError(f"Expected 8-bit L or RGB PNG: {row['path']}")
            arr = np.asarray(im.convert('RGB'), dtype=np.uint8).copy()
        # Validate grayscale RGB rather than silently changing an intensity mapping.
        if not (np.array_equal(arr[..., 0], arr[..., 1]) and
                np.array_equal(arr[..., 0], arr[..., 2])):
            raise ValueError(f"RGB channels differ; expected grayscale export: {row['path']}")
        x = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255)
        return x, index


class ChannelMoments:
    """Float64 parallel-Welford population moments over B,H,W, one value per C."""
    def __init__(self):
        self.count, self.mean, self.m2 = 0, None, None

    def update(self, z):
        x = z.detach().double().cpu().permute(1, 0, 2, 3).reshape(z.shape[1], -1)
        if not torch.isfinite(x).all():
            raise ValueError('Nonfinite latent')
        n = x.shape[1]
        mean = x.mean(1)
        m2 = ((x - mean[:, None]) ** 2).sum(1)
        if self.count == 0:
            self.count, self.mean, self.m2 = n, mean, m2
            return
        delta = mean - self.mean
        total = self.count + n
        self.m2 += m2 + delta.square() * self.count * n / total
        self.mean += delta * n / total
        self.count = total

    def result(self):
        if not self.count:
            raise ValueError('No training latents for normalization')
        return {'mean': self.mean.float().view(1, -1, 1, 1),
                'var': (self.m2 / self.count).float().view(1, -1, 1, 1)}


def write_loss_plot(history, path):
    """Dependency-free SVG, rewritten after each completed epoch."""
    width, height = 800, 440
    left, top, right, bottom = 85, 45, 760, 370
    values = [r[k] for r in history for k in ('train_l1', 'val_l1')]
    maximum = max(max(values) * 1.1, 1e-8)
    xmax = max(1, len(history) - 1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" font-size="14" fill="#222">',
             '<text x="85" y="25">Spectrogram decoder reconstruction loss</text>']
    for i in range(6):
        y = bottom - i / 5 * (bottom - top)
        parts += [f'<path d="M{left},{y} H{right}" stroke="#ddd"/>',
                  f'<text x="10" y="{y+5}">{maximum*i/5:.5f}</text>']
    for i, row in enumerate(history):
        x = left + i / xmax * (right - left)
        if len(history) <= 20 or i % max(1, len(history)//10) == 0:
            parts.append(f'<text x="{x}" y="395" text-anchor="middle">{row["epoch"]}</text>')
    for key, color, label, label_x in [('train_l1', '#2563eb', 'Training', 440),
                                        ('val_l1', '#e45b20', 'Validation', 590)]:
        points = [(left + i / xmax * (right-left), bottom - row[key]/maximum*(bottom-top))
                  for i, row in enumerate(history)]
        coordinates = ' '.join(f'{x:.2f},{y:.2f}' for x,y in points)
        parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.extend(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>' for x,y in points)
        parts.append(f'<text x="{label_x}" y="25" fill="{color}">{label}</text>')
    parts += ['<text x="400" y="430">Epoch</text>', '</g></svg>']
    Path(path).write_text('\n'.join(parts))


def write_batch_loss_plot(batches, epochs, path, steps_per_epoch, objective=False, width=1600, start_step=0, zoom=False):
    """Plot raw batch losses and epoch validation at their global batch steps.

    Keep CSV data complete; use min/max buckets to bound SVG size while retaining
    visible spikes for long runs.
    """
    if not batches:
        return
    points = [(r['global_step'], r['loss' if objective else 'l1']) for r in batches
              if r['global_step'] >= start_step]
    if not points:
        return
    # Trailing mean is computed from every original batch, before display reduction.
    raw_values = np.asarray([p[1] for p in points], dtype=np.float64)
    cumulative = np.concatenate(([0.], np.cumsum(raw_values)))
    end = np.arange(1, len(points) + 1)
    begin = np.maximum(0, end - 200)
    averages = (cumulative[end] - cumulative[begin]) / (end - begin)
    stride = max(1, len(points) // width)
    smooth_indices = sorted(set(range(0, len(points), stride)) | {len(points)-1})
    smoothed = [(points[i][0], float(averages[i])) for i in smooth_indices]
    point_budget = max(1400, width * 2)
    if len(points) > point_budget:
        buckets = point_budget // 2
        bucket_size = (len(points) + buckets - 1) // buckets
        reduced = []
        for start in range(0, len(points), bucket_size):
            bucket = points[start:start + bucket_size]
            reduced.extend(sorted({min(bucket, key=lambda p: p[1]),
                                   max(bucket, key=lambda p: p[1])}))
        points = [points[0], *reduced, points[-1]]
    validation = [(r['epoch'] * steps_per_epoch, r['val_loss' if objective else 'val_l1']) for r in epochs if r['epoch'] * steps_per_epoch >= start_step]
    xmax = max(1, batches[-1]['global_step'])
    values = [p[1] for p in points + validation]
    ymin = min(values) if zoom else 0.
    ymax = max(max(values), ymin + 1e-8)
    margin = (ymax-ymin)*.1
    ymax += margin
    if zoom:
        ymin -= margin
    xmin = points[0][0] if zoom else 0
    span = width - 125
    def coordinate(point):
        x, y = point
        return 85 + (x-xmin) / max(1, xmax-xmin) * span, 370 - (y-ymin) / (ymax-ymin) * 305
    title = 'Composite reconstruction objective' if objective else 'Reconstruction L1 by batch step'
    if zoom:
        title += f' (zoom: steps {xmin} onwards; nonzero Y axis)'
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="440">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" font-size="13" fill="#222">',
             f'<text x="85" y="25">{title}</text>',
             '<text x="85" y="45" fill="#2563eb">Training batch (raw)</text>',
             '<text x="290" y="45" fill="#164e63">Training trailing mean (200 batches)</text>',
             f'<text x="{width-210}" y="45" fill="#e45b20">Validation (epoch)</text>']
    for i in range(6):
        y = 370 - i * 61
        x = 85 + i / 5 * span
        parts += [f'<path d="M85,{y} H{width-40}" stroke="#ddd"/>',
                  f'<text x="10" y="{y+5}">{ymin+(ymax-ymin)*i/5:.5f}</text>',
                  f'<text x="{x}" y="395" text-anchor="middle">{round(xmin+(xmax-xmin)*i/5)}</text>']
    for series, color in [(points, '#2563eb'), (smoothed, '#164e63'), (validation, '#e45b20')]:
        coords = [coordinate(p) for p in series]
        joined = ' '.join(f'{x:.2f},{y:.2f}' for x,y in coords)
        opacity = .3 if color == '#2563eb' else 1
        thickness = 1 if color == '#2563eb' else 2
        parts.append(f'<polyline points="{joined}" fill="none" stroke="{color}" stroke-width="{thickness}" opacity="{opacity}"/>')
        if color == '#e45b20' or len(coords) == 1:
            parts.extend(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>' for x,y in coords)
    parts += [f'<text x="{width/2-100}" y="430">Global training batch step</text>', '</g></svg>']
    destination = Path(path)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text('\n'.join(parts))
    temporary.replace(destination)
