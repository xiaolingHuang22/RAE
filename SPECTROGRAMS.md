# Stage 1 RAE for grayscale spectrograms

This workflow freezes the authors' DINOv2-with-registers encoder and fine-tunes
only the pretrained ViT-XL decoder with L1 reconstruction loss. It does not train
an adapted encoder or a Stage 2 diffusion model. The original ImageNet trainer
and configurations are unchanged.

## Data and patient splits

Images must be 256×256, 8-bit grayscale PNGs (L), or RGB PNGs with identical
channels. Each image represents one patient's channel over 120 seconds. No
cropping, flips, contrast scaling, or resizing is applied by the dataset loader.
Values are divided by 255 and supplied as three identical channels. The official
RAE wrapper then resizes to 224×224 using bicubic interpolation and applies its
DINO input normalization. Time/frequency orientation and the original dB mapping
remain as supplied. Bicubic resizing can slightly overshoot the input range,
consistent with the authors' wrapper.

Arrange images as:

```text
/path/to/spectrograms/
  patient_001/
    channel_01/segment_000.png
    channel_02/segment_000.png
  patient_002/
    ...
```

The first directory under the dataset root is the patient ID. Use the same ID for
all recordings belonging to that patient. Do not place role/split directories
above patients. Exclude any patients or recordings you judge unsuitable before
preparing the manifest.

The generated split is approximately 70% train, 15% validation, 15% test **by
patient count**, with at least one patient in each held-out split (minimum three
patients total). Small datasets therefore have different proportions. Assignment
uses a reproducible seed-dependent hash ordering. Freeze the CSV once prepared;
regenerating it after adding patients can change assignments. For a larger study,
review diagnosis/site distributions and explicitly edit the patient-level split
before training. Leakage checks reject patients assigned to multiple splits.

You can instead supply your own CSV with columns `path,patient_id,split,role`.
Paths are relative to the dataset root. Split is `train`, `val`, or `test`; `role`
is descriptive (e.g. target/condition/negative), not a class label. Every training
row in the CSV participates in reconstruction training. Channel, recording,
segment, and preprocessing metadata can be added as extra columns and are
preserved in reports and the copied CSV. Distinct paths must produce distinct
cache filenames after changing their extension to `.npy`.

## Setup on the lab server

Run commands from the RAE repository root. Install a matching CUDA-enabled
PyTorch/torchvision pair appropriate for your server, then:

```bash
python -m pip install -r requirements.txt
python -m pip install Pillow huggingface_hub
hf download nyu-visionx/RAE-collections \
  decoders/dinov2/wReg_base/ViTXL_n08/model.pt --local-dir models
```

Only this Stage 1 decoder checkpoint is required from the RAE collection. DINO
weights and processor assets are fetched separately when training starts. Internet
access is required initially; subsequent cache/bundle operations use local assets.
If your HF CLI is absent, install the current `huggingface_hub` package providing
`hf`. There is no matplotlib dependency: training produces a standalone SVG plot.

Review `configs/stage1/training/spectrogram.yaml`. Its default encoder revision
`main` is resolved to an immutable HF snapshot and recorded in `run.json`; you can
replace it with a specific HF commit before training. Local encoder paths are
also accepted, with exported asset hashes establishing their identity.

## Prepare and train

Replace the placeholder paths below. The output directory must be empty for a
new run. Training uses one GPU; choose a device such as `cuda:1` if needed.

```bash
python src/spectrogram_rae.py prepare \
  --data-root /path/to/spectrograms \
  --manifest /path/to/experiment/samples.csv --seed 42

python src/spectrogram_rae.py train \
  --config configs/stage1/training/spectrogram.yaml \
  --data-root /path/to/spectrograms \
  --manifest /path/to/experiment/samples.csv \
  --output /path/to/experiment/run \
  --batch-size 4 --epochs 30 --lr 2e-5 --precision bf16 --device cuda
```

These are starting settings, not scientifically validated hyperparameters. Use
`--precision fp16` if your GPU lacks bf16, or `fp32` for the initial numerical
baseline. Reduce batch size to 1–2 if the XL decoder exceeds GPU memory. The
workflow retains partial final batches, clips decoder gradients at norm 1, and
uses AdamW with cosine learning-rate decay. Latent noise, GAN, and LPIPS are
excluded from this first baseline. No EMA model is used; validation selects actual
decoder weights. CPU execution is supported for small verification models.

Training first evaluates the pretrained decoder on validation patients. Every
completed epoch then evaluates the current decoder in fp32 and updates:

- `loss_plot.svg`: training and validation L1 on the same plot, viewable in a browser.
- `losses.csv`: exact epoch metrics and learning rate for plotting elsewhere.
- `best_decoder.pt`: native decoder state dict with lowest patient-averaged validation L1.
- `best_validation.json`: metrics and selected epoch.
- `last_training.pt`: last completed epoch, optimizer/scheduler/scaler and PyTorch RNG states.
- `resolved_config.yaml`, `run.json`, and `samples.csv`: configuration and provenance.
- `baseline/` and `validation/`: input/reconstruction pairs and unclipped float RGB reconstructions for up to 16 samples, before training and after loading the best decoder.
- `baseline_metrics.json` and `validation_metrics.json`: full per-image and per-patient reports.

Training L1 is image-weighted over optimization batches; validation L1 on the plot
is patient-weighted, so patients with many channels do not dominate checkpoint
selection. `val_image_l1` is also in the CSV for a directly image-weighted comparison.
Training losses are measured during optimization; validation uses the end-of-epoch
weights. The plot is refreshed after validation each epoch, while batch losses
are printed during training. It can be refreshed in a browser while the run is
active.

To recover from an interruption, repeat the exact training command with
`--resume`. It resumes at the next epoch after `last_training.pt`; partial epoch
work is repeated. Dataset root, manifest, epochs, LR, seed, precision, and batch
size must match. Keep the original images unchanged. Recovery is not a guarantee
of bitwise reproducibility across GPU/software versions.

Reports include L1, MSE, frequency/time mean-profile errors, and fraction of RGB
outputs outside `[0,1]`. Metrics use unclipped RGB tensors. PNG previews clip and
quantize; decoded preview PNGs average RGB into grayscale. Profile errors provide
simple axis-preservation diagnostics, not validated spike-detection or band-power
metrics. Compare important transients and bands visually and validate downstream
features separately. Test patients are never used for training or checkpoint
selection.

## Export the handoff and cache

```bash
python src/spectrogram_rae.py export \
  --run /path/to/experiment/run \
  --data-root /path/to/spectrograms \
  --manifest /path/to/experiment/samples.csv \
  --output /path/to/experiment/handoff \
  --batch-size 4 --device cuda
```

Use an empty handoff directory. An expanded export CSV may include additional
condition/negative images, but each patient must already exist in the training
run and retain its original split. Statistics always use the original training
rows, not the expanded export set. All paths still resolve under `--data-root`.

Export calculates exact streaming population mean/variance per channel over
training samples and spatial locations. No validation/test image contributes.
The wrapper applies `(z-mean)/sqrt(var+1e-5)` once during encoding and inverses it
once during decoding. Do not normalize exported `.npy` files again.

The expected latent shape for the default DINOv2-B model is `[16,16,768]`. The
export records the actual shape. Each `.npy` is unbatched HWC float32, finite only,
with pickle disabled. Paths mirror the image paths under `handoff/latents/`.
`RAE.encode()` returns a tensor directly. Features are final patch tokens with
CLS and all four register tokens removed; the authors' wrapper disables the
final layer norm's affine parameters.

The handoff contains local HF encoder/processor files, native decoder weights,
decoder config, inference YAML, normalization `.pt`, mean/std `.npy` arrays,
patient/sample CSV, manifest with asset hashes, validation reports, loss history,
loss plot, runtime source, requirements, and license. The manifest explicitly
marks external STFT/dB settings as needing data-owner metadata: fill in the real
spectrogram-generation parameters before the scientific handoff. They cannot be
inferred from an exported PNG. Encoder features stay fixed when only the decoder
is trained; cache regeneration is required if encoder/preprocessing/statistics
change, not simply because decoder weights change.

Export checks encode→save→load→decode and reloads the local encoder assets to
confirm wrapper latents match. Native HF export stores identity final layer-norm
affine weights for completeness; the official wrapper removes them again.

## Decode generated latents in the other environment

After installing compatible dependencies, the exported runtime can operate
without this repository checkout:

```bash
python /path/to/handoff/runtime/spectrogram_rae.py decode \
  --bundle /path/to/handoff \
  --latents /path/to/generated_latents \
  --output /path/to/reconstructed_spectrograms --device cuda
```

`--latents` accepts a single `.npy` or a directory tree. Input must match the
manifest's normalized HWC float32 contract. Model/config/statistics checksums
are checked before decoding. Outputs preserve directory structure and include
an unclipped RGB float `.npy` reconstruction plus a grayscale PNG preview.
Existing outputs are not overwritten. Generated reconstructions may have small
RGB channel differences even though targets are grayscale.

The consuming model needs 768-channel support with the default encoder. Do not
use a four-channel VAE contract or an arbitrary DINO hidden layer.

## Final held-out test report

After model/settings selection is complete:

```bash
python src/spectrogram_rae.py evaluate \
  --run /path/to/experiment/run --split test \
  --data-root /path/to/spectrograms \
  --output /path/to/experiment/test_report --device cuda
```

Copy this final report alongside the handoff. The encoder is frozen throughout;
decoder reconstruction improvements do not imply improved encoder features for
your downstream task.

## Offline verification

```bash
python -m unittest discover -s tests -v
```

Tests use a tiny randomly initialized real RAE and synthetic grayscale images.
They cover patient-leakage rejection, grayscale validation, exact statistics,
training/validation plots, checkpoint recovery, frozen encoder identity, cache
round-trip, bundle reload, decoding, and held-out reporting. They do not establish
scientific performance or full-model GPU memory requirements.
