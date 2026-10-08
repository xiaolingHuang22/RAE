#!/usr/bin/env python3
"""Single-device Stage-1 spectrogram training and portable latent handoff."""
import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import platform
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import transformers
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from PIL import Image
from transformers import AutoImageProcessor, AutoConfig
from huggingface_hub import snapshot_download

from spectrogram_data import (SpectrogramDataset, ChannelMoments, read_manifest,
                              prepare_manifest, write_loss_plot)
from stage1 import RAE


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def device_for(name):
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for small smoke tests')
    return device


def load_model(config, device):
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    params = dict(cfg['stage_1']['params'])
    # Resolve local config/checkpoint paths against this config, not the working directory.
    base = Path(config).resolve().parent
    for key in ('encoder_config_path', 'decoder_config_path', 'pretrained_decoder_path',
                'normalization_stat_path'):
        value = params.get(key)
        if value and not Path(value).is_absolute() and (base / value).exists():
            params[key] = str(base / value)
    enc = dict(params['encoder_params'])
    if (base / enc['dinov2_path']).exists():
        enc['dinov2_path'] = str(base / enc['dinov2_path'])
    params['encoder_params'] = enc
    model = RAE(**params).to(device)
    model.encoder.requires_grad_(False)
    model.encoder.eval()
    return model, cfg


def loader(root, rows, batch, workers, shuffle=False):
    return DataLoader(SpectrogramDataset(root, rows), batch_size=batch,
                      shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=False)


def amp_context(device, precision):
    if precision == 'fp32':
        return nullcontext()
    if device.type != 'cuda':
        raise ValueError('Mixed precision requires CUDA in this workflow')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU does not support bf16; use fp16 or fp32')
    return torch.autocast('cuda', dtype=torch.bfloat16 if precision == 'bf16' else torch.float16)


@torch.no_grad()
def evaluate(model, batches, device, rows, preview_dir=None):
    model.eval()
    records = []
    for images, indices in batches:
        images = images.to(device)
        recon = model(images).float()
        if not torch.isfinite(recon).all():
            raise ValueError('Nonfinite reconstruction')
        error = recon - images
        for i, index in enumerate(indices.tolist()):
            # Profiles compare mean intensity over time / frequency, preserving axis order.
            record = dict(rows[index])
            record.update(l1=error[i].abs().mean().item(), mse=error[i].square().mean().item(),
                          frequency_profile_l1=error[i].mean(dim=(0, 2)).abs().mean().item(),
                          temporal_profile_l1=error[i].mean(dim=(0, 1)).abs().mean().item(),
                          clipped_fraction=((recon[i] < 0) | (recon[i] > 1)).float().mean().item())
            records.append(record)
            if preview_dir is not None and len(records) <= 16:
                out = Path(preview_dir)
                out.mkdir(parents=True, exist_ok=True)
                pair = torch.cat([images[i], recon[i].clamp(0, 1)], dim=2)
                arr = pair.cpu().permute(1, 2, 0).numpy()
                Image.fromarray(np.round(arr * 255).astype(np.uint8)).save(out / f'{len(records):03d}.png')
                np.save(out / f'{len(records):03d}_reconstruction.npy',
                        recon[i].cpu().permute(1, 2, 0).numpy(), allow_pickle=False)
    if not records:
        raise ValueError('Evaluation split is empty')
    patients = sorted({r['patient_id'] for r in records})
    metrics = ['l1', 'mse', 'frequency_profile_l1', 'temporal_profile_l1', 'clipped_fraction']
    by_patient = {p: {k: float(np.mean([r[k] for r in records if r['patient_id'] == p]))
                      for k in metrics} for p in patients}
    return {'image_mean': {k: float(np.mean([r[k] for r in records])) for k in metrics},
            'patient_mean': {k: float(np.mean([v[k] for v in by_patient.values()])) for k in metrics},
            'patients': by_patient, 'samples': records}


def train(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()) and not args.resume:
        raise ValueError('Use an empty output directory to avoid mixing experiments')
    device = device_for(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rows = read_manifest(args.manifest)
    training = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == 'val']
    if not training or not validation:
        raise ValueError('Both train and val patients are required')
    if args.resume:
        previous = json.loads((out / 'run.json').read_text())['arguments']
        for key in ('data_root', 'epochs', 'lr', 'seed', 'precision', 'batch_size'):
            if previous[key] != vars(args)[key]:
                raise ValueError(f'Resume requires unchanged {key}')
        if sha256(args.manifest) != sha256(out / 'samples.csv'):
            raise ValueError('Resume manifest differs from saved patient split')
        cfg = OmegaConf.to_container(OmegaConf.load(out / 'resolved_config.yaml'), resolve=True)
    else:
        cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    params = cfg['stage_1']['params']
    if params.get('normalization_stat_path'):
        raise ValueError('Train with raw latents; export computes training-only statistics')
    if float(params.get('noise_tau', 0)) != 0:
        raise ValueError('The initial baseline requires noise_tau: 0')
    # Resolve and record the exact HF snapshot, including processor assets.
    enc_id = params['encoder_params']['dinov2_path']
    revision = params['encoder_params'].get('revision', 'main')
    if not Path(enc_id).exists():
        snapshot = snapshot_download(enc_id, revision=revision,
                                     allow_patterns=['*.json', '*.safetensors', '*.bin'])
        resolved_revision = Path(snapshot).name
    else:
        snapshot, resolved_revision = str(Path(enc_id).resolve()), 'local-checkpoint'
    params['encoder_config_path'] = snapshot
    params['encoder_params']['dinov2_path'] = snapshot
    params['encoder_params'].pop('revision', None)
    params['decoder_config_path'] = str(Path(params['decoder_config_path']).resolve())
    checkpoint = Path(params['pretrained_decoder_path']).resolve()
    params['pretrained_decoder_path'] = str(checkpoint)
    params['strict_decoder_loading'] = True
    if not args.resume:
        OmegaConf.save(OmegaConf.create(cfg), out / 'resolved_config.yaml')
        shutil.copyfile(args.manifest, out / 'samples.csv')
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = 'unknown'
    if not args.resume:
        save_json(out / 'run.json', {'encoder_id': enc_id, 'encoder_revision': resolved_revision,
                  'initial_decoder_sha256': sha256(checkpoint), 'repository_commit': commit,
                  'seed': args.seed, 'arguments': vars(args),
                  'software': {'python': platform.python_version(), 'torch': str(torch.__version__),
                               'transformers': transformers.__version__, 'numpy': np.__version__}})
    model, _ = load_model(out / 'resolved_config.yaml', device)
    batches = loader(args.data_root, training, args.batch_size, args.workers, True)
    val_batches = loader(args.data_root, validation, args.batch_size, args.workers)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=args.lr,
                                  betas=(.9, .95), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs,
                                                          eta_min=args.lr * .1)
    scaler = torch.amp.GradScaler('cuda', enabled=args.precision == 'fp16')
    history, best, first_epoch = [], float('inf'), 1
    if args.resume:
        state = torch.load(out / 'last_training.pt', map_location='cpu', weights_only=True)
        model.decoder.load_state_dict(state['decoder'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        history = state['history']
        first_epoch = state['epoch'] + 1
        best = json.loads((out / 'best_validation.json').read_text())['patient_mean']['l1']
        torch.set_rng_state(state['rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        baseline = evaluate(model, val_batches, device, validation, out / 'baseline')
        save_json(out / 'baseline_metrics.json', baseline)
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        model.encoder.eval()
        total, count = 0., 0
        for step, (images, _) in enumerate(batches, 1):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, args.precision):
                recon = model(images)
                loss = (recon.float() - images).abs().mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(images)
            count += len(images)
            if step % args.log_every == 0:
                print(f'Epoch {epoch} batch {step}/{len(batches)} L1={loss.item():.6f}', flush=True)
        metrics = evaluate(model, val_batches, device, validation)
        # Patient weighting avoids allowing patients with many channels to dominate selection.
        val = metrics['patient_mean']['l1']
        history.append({'epoch': epoch, 'train_l1': total / count,
                        'val_l1': val, 'val_image_l1': metrics['image_mean']['l1'],
                        'val_mse': metrics['patient_mean']['mse'], 'lr': optimizer.param_groups[0]['lr']})
        scheduler.step()
        with (out / 'losses.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        write_loss_plot(history, out / 'loss_plot.svg')
        if val < best:
            best = val
            torch.save(model.decoder.state_dict(), out / 'best_decoder.pt')
            save_json(out / 'best_validation.json', {'epoch': epoch, **metrics})
        temporary = out / 'last_training.tmp'
        torch.save({'epoch': epoch, 'decoder': model.decoder.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(), 'history': history, 'rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}, temporary)
        temporary.replace(out / 'last_training.pt')
        print(f'Epoch {epoch}: train L1={total/count:.6f}, val patient L1={val:.6f}', flush=True)
    model.decoder.load_state_dict(torch.load(out / 'best_decoder.pt', map_location=device, weights_only=True))
    save_json(out / 'validation_metrics.json', evaluate(model, val_batches, device, validation, out / 'validation'))


def export(args):
    run, out = Path(args.run), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Use an empty handoff directory')
    device = device_for(args.device)
    rows = read_manifest(args.manifest)
    # Export can contain extra condition/negative samples, but cannot change patient splits.
    original = read_manifest(run / 'samples.csv')
    splits = {r['patient_id']: r['split'] for r in original}
    for r in rows:
        if r['patient_id'] not in splits or splits[r['patient_id']] != r['split']:
            raise ValueError('Export patients and splits must match the training run')
    stats_rows = [r for r in original if r['split'] == 'train']
    model, cfg = load_model(run / 'resolved_config.yaml', device)
    model.decoder.load_state_dict(torch.load(run / 'best_decoder.pt', map_location=device, weights_only=True))
    model.eval()
    model.noise_tau = 0
    moments = ChannelMoments()
    with torch.inference_mode():
        for images, _ in loader(args.data_root, stats_rows, args.batch_size, args.workers):
            moments.update(model.encode(images.to(device)))
    stats = moments.result()
    model.eps = 1e-5
    torch.save(stats, out / 'normalization.pt')
    np.save(out / 'latent_mean.npy', stats['mean'].numpy().reshape(-1), allow_pickle=False)
    np.save(out / 'latent_std.npy', torch.sqrt(stats['var'] + model.eps).numpy().reshape(-1), allow_pickle=False)
    model.latent_mean, model.latent_var = stats['mean'], stats['var']
    model.do_normalization = True
    model.eps = 1e-5
    encoder_dir = out / 'encoder'
    encoder_state = dict(model.encoder.encoder.state_dict())
    # The official wrapper removes these affine parameters. Store identity values in
    # HF format so reloading is complete; Dinov2withNorm removes them again.
    if model.encoder.encoder.layernorm.weight is None:
        encoder_state['layernorm.weight'] = torch.ones(model.latent_dim, device=device)
        encoder_state['layernorm.bias'] = torch.zeros(model.latent_dim, device=device)
    model.encoder.encoder.save_pretrained(encoder_dir, state_dict=encoder_state)
    AutoImageProcessor.from_pretrained(cfg['stage_1']['params']['encoder_config_path']).save_pretrained(encoder_dir)
    AutoConfig.from_pretrained(cfg['stage_1']['params']['decoder_config_path']).save_pretrained(out / 'decoder_config')
    shutil.copyfile(run / 'best_decoder.pt', out / 'decoder.pt')
    shutil.copyfile(args.manifest, out / 'samples.csv')
    for name in ('baseline_metrics.json', 'validation_metrics.json', 'losses.csv', 'loss_plot.svg'):
        shutil.copyfile(run / name, out / name)
    src = Path(__file__).resolve().parent
    shutil.copytree(src / 'stage1', out / 'runtime' / 'stage1', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('spectrogram_data.py', 'spectrogram_rae.py'):
        shutil.copyfile(src / name, out / 'runtime' / name)
    shutil.copyfile(src.parent / 'LICENSE', out / 'LICENSE')
    shutil.copyfile(src.parent / 'requirements.txt', out / 'requirements.txt')
    shutil.copyfile(src.parent / 'SPECTROGRAMS.md', out / 'SPECTROGRAMS.md')
    cfg['stage_1']['params'].update(encoder_config_path='encoder', decoder_config_path='decoder_config',
                  pretrained_decoder_path='decoder.pt', normalization_stat_path='normalization.pt',
                  noise_tau=0., eps=1e-5, strict_decoder_loading=True)
    cfg['stage_1']['params']['encoder_params']['dinov2_path'] = 'encoder'
    OmegaConf.save(OmegaConf.create({'stage_1': cfg['stage_1']}), out / 'inference.yaml')
    shape, roundtrip = None, None
    with torch.inference_mode():
        for images, indices in loader(args.data_root, rows, args.batch_size, args.workers):
            images = images.to(device)
            z = model.encode(images).float()
            for i, index in enumerate(indices.tolist()):
                array = z[i].cpu().permute(1, 2, 0).contiguous().numpy()
                if not np.isfinite(array).all():
                    raise ValueError('Nonfinite cache latent')
                shape = list(array.shape)
                path = out / 'latents' / Path(rows[index]['path']).with_suffix('.npy')
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, array, allow_pickle=False)
                if roundtrip is None:
                    loaded = np.load(path, allow_pickle=False)
                    np.testing.assert_array_equal(array, loaded)
                    restored = torch.from_numpy(loaded.copy()).permute(2, 0, 1).unsqueeze(0).to(device)
                    direct = model.decode(z[i:i+1])
                    disk = model.decode(restored)
                    torch.testing.assert_close(direct, disk, rtol=1e-5, atol=1e-6)
                    roundtrip = {'cache_path': str(path.relative_to(out)), 'max_decode_error': (direct-disk).abs().max().item()}
    if shape is None:
        raise ValueError('No cache outputs')
    provenance = json.loads((run / 'run.json').read_text())
    files = [p for p in out.rglob('*') if p.is_file() and 'latents' not in p.relative_to(out).parts]
    save_json(out / 'manifest.json', {'format_version': 1, **provenance,
        'decoder': {'architecture': 'GeneralDecoder ViT-XL', 'checkpoint': 'decoder.pt',
                    'selection': json.loads((run / 'best_validation.json').read_text())['epoch']},
        'input': {'shape': [3,256,256], 'range': [0,1], 'grayscale_rgb': True,
                  'encoder_resize': 'bicubic 224x224 align_corners=False; no crop or augmentation',
                  'segment_seconds': 120, 'spectrogram_parameters': 'External source mapping unchanged; dB/STFT parameters must be supplied by data owner'},
        'latent': {'shape': shape, 'dtype': 'float32', 'layout': 'HWC',
                   'output': 'RAE.encode tensor; Dinov2withNorm final last_hidden_state',
                   'cls_tokens': False, 'register_tokens': False, 'layernorm_affine': False},
        'normalization': {'applied_by': 'RAE.encode once; RAE.decode inverses once',
                          'formula': '(z-mean)/sqrt(var+1e-5)', 'eps': 1e-5,
                          'axes': 'training samples and spatial positions, per channel',
                          'variance': 'population', 'training_samples': len(stats_rows),
                          'stats': 'normalization.pt', 'mean': 'latent_mean.npy', 'std': 'latent_std.npy'},
        'patient_split': 'samples.csv', 'roundtrip': roundtrip,
        'hashes': {str(p.relative_to(out)): sha256(p) for p in files}})
    # Verify the exported local assets reload and yield identical wrapper latents.
    reloaded, _ = load_model(out / 'inference.yaml', device)
    reloaded.eval()
    probe, _ = SpectrogramDataset(args.data_root, rows)[0]
    with torch.inference_mode():
        probe = probe.unsqueeze(0).to(device)
        torch.testing.assert_close(model.encode(probe), reloaded.encode(probe), rtol=1e-5, atol=1e-6)
    print(f'Exported {len(rows)} latents, shape {shape}, to {out}', flush=True)


def decode(args):
    bundle = Path(args.bundle)
    metadata = json.loads((bundle / 'manifest.json').read_text())
    for rel, expected in metadata['hashes'].items():
        if sha256(bundle / rel) != expected:
            raise ValueError(f'Bundle checksum mismatch: {rel}')
    device = device_for(args.device)
    model, _ = load_model(bundle / 'inference.yaml', device)
    model.eval()
    source, out = Path(args.latents), Path(args.output)
    paths = [source] if source.is_file() else sorted(source.rglob('*.npy'))
    if not paths:
        raise ValueError('No latent .npy files found')
    with torch.inference_mode():
        for path in paths:
            array = np.load(path, allow_pickle=False)
            if list(array.shape) != metadata['latent']['shape'] or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f'Invalid HWC float32 latent: {path}')
            z = torch.from_numpy(array.copy()).permute(2,0,1).unsqueeze(0).to(device)
            recon = model.decode(z)[0].cpu().permute(1,2,0).numpy()
            if not np.isfinite(recon).all():
                raise ValueError(f'Nonfinite decoded output: {path}')
            rel = Path(path.name) if source.is_file() else path.relative_to(source)
            dest = out / rel.with_suffix('.png')
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() or dest.with_suffix('.npy').exists():
                raise FileExistsError(dest)
            np.save(dest.with_suffix('.npy'), recon, allow_pickle=False)
            # Grayscale visualization, raw RGB reconstruction retained above.
            gray = np.round(np.clip(recon.mean(2), 0, 1) * 255).astype(np.uint8)
            Image.fromarray(gray).save(dest)


def report(args):
    run = Path(args.run)
    rows = [r for r in read_manifest(run / 'samples.csv') if r['split'] == args.split]
    device = device_for(args.device)
    model, _ = load_model(run / 'resolved_config.yaml', device)
    model.decoder.load_state_dict(torch.load(run / 'best_decoder.pt', map_location=device, weights_only=True))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    result = evaluate(model, loader(args.data_root, rows, args.batch_size, args.workers), device, rows, out / 'previews')
    save_json(out / 'metrics.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--layout', choices=['auto', 'presplit', 'patient-folders'], default='auto',
                   help='Auto preserves existing train/val/test folders; presplit requires them')
    p.add_argument('--data-root', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--seed', type=int, default=42)
    for command in ('train', 'export', 'evaluate'):
        p = sub.add_parser(command)
        p.add_argument('--data-root', required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--device', default='cuda')
        p.add_argument('--batch-size', type=int, default=4)
        p.add_argument('--workers', type=int, default=4)
        if command != 'evaluate':
            p.add_argument('--manifest', required=True)
        if command == 'train':
            p.add_argument('--config', default='configs/stage1/training/spectrogram.yaml')
            p.add_argument('--epochs', type=int, default=30)
            p.add_argument('--lr', type=float, default=2e-5)
            p.add_argument('--seed', type=int, default=42)
            p.add_argument('--precision', choices=['fp32','fp16','bf16'], default='fp32')
            p.add_argument('--log-every', type=int, default=20)
            p.add_argument('--resume', action='store_true', help='Resume last completed epoch with unchanged settings')
        else:
            p.add_argument('--run', required=True)
        if command == 'evaluate':
            p.add_argument('--split', choices=['val','test'], default='test')
    p = sub.add_parser('decode')
    p.add_argument('--bundle', required=True)
    p.add_argument('--latents', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if hasattr(args, 'batch_size') and args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.command == 'train' and (args.epochs < 1 or args.lr <= 0 or args.log_every < 1):
        parser.error('epochs, lr, and log-every must be positive')
    if args.command == 'prepare':
        rows = prepare_manifest(args.data_root, args.manifest, args.seed, args.layout)
        print(f'Wrote {len(rows)} records to {args.manifest}')
    else:
        {'train': train, 'export': export, 'decode': decode, 'evaluate': report}[args.command](args)


if __name__ == '__main__':
    main()
