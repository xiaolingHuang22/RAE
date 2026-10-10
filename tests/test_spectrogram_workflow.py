"""Offline integration tests using a tiny real RAE, not pretrained scientific validation."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from transformers import (Dinov2WithRegistersConfig, Dinov2WithRegistersModel,
                          ViTMAEConfig, BitImageProcessor)
from spectrogram_data import ChannelMoments, SpectrogramDataset, prepare_manifest, read_manifest
from spectrogram_rae import train, export, decode, report, reconstruction_metrics, training_objective
from stage1 import RAE


class WorkflowTests(unittest.TestCase):
    def test_reconstruction_metrics(self):
        torch.set_num_threads(1)
        target = torch.full((2, 3, 32, 32), .5)
        same = reconstruction_metrics(target, target)
        torch.testing.assert_close(same['mse'], torch.zeros(2))
        torch.testing.assert_close(same['rmse'], torch.zeros(2))
        torch.testing.assert_close(same['ssim'], torch.ones(2), atol=1e-4, rtol=0)
        torch.testing.assert_close(same['psnr'], torch.full((2,), 120.))
        changed = reconstruction_metrics(target + .1, target)
        torch.testing.assert_close(changed['mse'], torch.full((2,), .01))
        torch.testing.assert_close(changed['rmse'], torch.full((2,), .1))
        torch.testing.assert_close(changed['psnr'], torch.full((2,), 20.))
        expected_ssim = (2*.5*.6 + .01**2)/(.5**2 + .6**2 + .01**2)
        torch.testing.assert_close(changed['ssim'], torch.full((2,), expected_ssim), atol=5e-4, rtol=0)

    def test_structure_loss_gradients(self):
        torch.set_num_threads(1)
        target = torch.rand(2, 3, 32, 32)
        prediction = (target * .8).detach().requires_grad_(True)
        settings = dict(mode='spectrogram', ssim_weight=.1, gradient_weight=.2, profile_weight=.1)
        loss, terms = training_objective(prediction, target, settings)
        self.assertGreater(loss.item(), terms['l1'].item())
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().item(), 0)
        identity, _ = training_objective(target, target, settings)
        self.assertLess(abs(identity.item()), 1e-5)
        l1, _ = training_objective(prediction, target, dict(mode='l1'))
        torch.testing.assert_close(l1, (prediction-target).abs().mean())

    def test_statistics(self):
        torch.manual_seed(2)
        z = torch.randn(7, 5, 4, 3)
        moments = ChannelMoments()
        moments.update(z[:2])
        moments.update(z[2:])
        stats = moments.result()
        torch.testing.assert_close(stats['mean'].flatten(), z.mean((0, 2, 3)))
        torch.testing.assert_close(stats['var'].flatten(), z.var((0, 2, 3), unbiased=False))
        restored = ((z-stats['mean']) / torch.sqrt(stats['var']+1e-5)) * torch.sqrt(stats['var']+1e-5) + stats['mean']
        torch.testing.assert_close(z, restored)

    def test_leakage_and_grayscale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'samples.csv'
            manifest.write_text('path,patient_id,split\na.png,p1,train\nb.png,p1,val\n')
            with self.assertRaisesRegex(ValueError, 'leakage'):
                read_manifest(manifest)
            rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            rgb[..., 0] = 1
            Image.fromarray(rgb).save(root / 'color.png')
            with self.assertRaisesRegex(ValueError, 'channels differ'):
                SpectrogramDataset(root, [{'path': 'color.png'}])[0]

    def test_presplit_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, patient in [('train', 'p1'), ('val', 'p2'), ('test', 'p3')]:
                folder = root / split / patient / 'channel_01'
                folder.mkdir(parents=True)
                for segment in range(2):
                    Image.new('L', (256, 256), 100).save(folder / f'segment_{segment}.PNG')
            (root / 'document1.txt').write_text('Unrelated document')
            documents = root / 'documents'
            documents.mkdir()
            Image.new('L', (10, 10)).save(documents / 'illustration.png')
            rows = prepare_manifest(root, root / 'samples.csv', seed=999)
            self.assertEqual(len(rows), 6)
            self.assertEqual({r['patient_id']: r['split'] for r in rows},
                             {'p1': 'train', 'p2': 'val', 'p3': 'test'})
            for row in rows:
                image, _ = SpectrogramDataset(root, [row])[0]
                self.assertEqual(tuple(image.shape), (3, 256, 256))
                self.assertEqual(Path(row['path']).parts[0], row['split'])
            duplicate = root / 'test' / 'p1'
            duplicate.mkdir()
            Image.new('L', (256, 256)).save(duplicate / 'image.png')
            with self.assertRaisesRegex(ValueError, 'Patient leakage'):
                prepare_manifest(root, root / 'invalid.csv')
            self.assertFalse((root / 'invalid.csv').exists())

    def test_incomplete_presplit_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'train').mkdir()
            with self.assertRaisesRegex(ValueError, 'requires train, val, and test'):
                prepare_manifest(root, root / 'invalid.csv')
            (root / 'val').mkdir()
            (root / 'test').mkdir()
            with self.assertRaisesRegex(ValueError, 'No PNG images'):
                prepare_manifest(root, root / 'invalid.csv')

    def test_complete_offline_workflow(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            for i in range(3):
                patient = data / ('train', 'val', 'test')[i] / f'patient_{i}'
                patient.mkdir(parents=True)
                image = np.random.default_rng(i).integers(0, 256, (256, 256), dtype=np.uint8)
                Image.fromarray(image).convert('RGB').save(patient / 'channel_segment.png')
            manifest = root / 'samples.csv'
            rows = prepare_manifest(data, manifest)
            self.assertEqual({r['split'] for r in rows}, {'train', 'val', 'test'})
            encoder = root / 'encoder'
            Dinov2WithRegistersModel(Dinov2WithRegistersConfig(hidden_size=24,
                 num_hidden_layers=1, num_attention_heads=4, intermediate_size=48,
                 num_register_tokens=4, image_size=224, patch_size=14)).save_pretrained(encoder)
            BitImageProcessor(image_mean=[.485,.456,.406], image_std=[.229,.224,.225]).save_pretrained(encoder)
            decoder = root / 'decoder_config'
            ViTMAEConfig(hidden_size=24, decoder_hidden_size=32, decoder_num_attention_heads=4,
                         decoder_num_hidden_layers=1, decoder_intermediate_size=64).save_pretrained(decoder)
            params = dict(encoder_cls='Dinov2withNorm', encoder_config_path=str(encoder),
                          encoder_params={'dinov2_path': str(encoder), 'normalize': True},
                          encoder_input_size=224, decoder_config_path=str(decoder),
                          reshape_to_2d=True, noise_tau=0.)
            model = RAE(**params)
            initial = root / 'initial.pt'
            torch.save(model.decoder.state_dict(), initial)
            encoder_before = {k:v.clone() for k,v in model.encoder.state_dict().items()}
            params.update(pretrained_decoder_path=str(initial), strict_decoder_loading=True)
            config = root / 'config.yaml'
            OmegaConf.save(OmegaConf.create({'stage_1': {'target':'stage1.RAE', 'params':params}}), config)
            common = dict(data_root=str(data), device='cpu', batch_size=1, workers=0)
            run = root / 'run'
            train(SimpleNamespace(**common, output=str(run), config=str(config), manifest=str(manifest),
                                  epochs=1, lr=2e-5, seed=42, precision='fp32', log_every=1, resume=False))
            train(SimpleNamespace(**common, output=str(run), config=str(config), manifest=str(manifest),
                                  epochs=1, lr=2e-5, seed=42, precision='fp32', log_every=1, resume=True))
            self.assertTrue((run / 'loss_plot.svg').exists())
            self.assertIn('Validation (epoch)', (run / 'loss_plot.svg').read_text())
            self.assertIn('Global training batch step', (run / 'loss_plot.svg').read_text())
            with (run / 'batch_losses.csv').open(newline='') as f:
                batch_records = list(csv.DictReader(f))
            self.assertEqual(len(batch_records), 1)
            self.assertEqual(batch_records[0]['global_step'], '1')
            self.assertGreater(float(batch_records[0]['l1']), 0)
            self.assertTrue((run / 'epoch_loss_plot.svg').exists())
            metrics = json.loads((run / 'validation_metrics.json').read_text())
            for scope in ('image_mean', 'patient_mean'):
                for metric in ('mse', 'rmse', 'ssim', 'psnr'):
                    self.assertTrue(np.isfinite(metrics[scope][metric]))
            handoff = root / 'handoff'
            export(SimpleNamespace(**common, run=str(run), output=str(handoff), manifest=str(manifest)))
            metadata = json.loads((handoff / 'manifest.json').read_text())
            self.assertEqual(metadata['latent']['shape'], [16,16,24])
            self.assertEqual(metadata['normalization']['training_samples'], 1)
            self.assertLessEqual(metadata['roundtrip']['max_decode_error'], 1e-6)
            decoded = root / 'decoded'
            decode(SimpleNamespace(bundle=str(handoff), latents=str(handoff / 'latents'),
                                   output=str(decoded), device='cpu'))
            self.assertEqual(len(list(decoded.rglob('*.png'))), 3)
            report(SimpleNamespace(**common, run=str(run), output=str(root / 'test'), split='test'))
            relocated = root / 'relocated_initial.pt'
            relocated.write_bytes(initial.read_bytes())
            baseline_out = root / 'baseline_report'
            report(SimpleNamespace(**common, run=str(run), output=str(baseline_out), split='val',
                                   decoder='baseline', decoder_checkpoint=str(relocated)))
            baseline = json.loads((baseline_out / 'metrics.json').read_text())
            original_baseline = json.loads((run / 'baseline_metrics.json').read_text())
            self.assertEqual(baseline['image_mean'], original_baseline['image_mean'])
            self.assertEqual(baseline['evaluation']['decoder'], 'baseline')
            self.assertEqual(baseline['evaluation']['sample_count'], 1)
            wrong = root / 'wrong.pt'
            wrong.write_bytes(b'not the original checkpoint')
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                report(SimpleNamespace(**common, run=str(run), output=str(root / 'wrong_report'),
                                       split='val', decoder='baseline', decoder_checkpoint=str(wrong)))
            structured_run = root / 'structured_run'
            structured_args = dict(**common, output=str(structured_run), config=str(config),
                                   manifest=str(manifest), epochs=1, lr=2e-5, seed=42,
                                   precision='fp32', log_every=1, resume=False, loss='spectrogram',
                                   ssim_weight=.1, gradient_weight=.2, profile_weight=.1)
            train(SimpleNamespace(**structured_args))
            structured_args['resume'] = True
            train(SimpleNamespace(**structured_args))
            self.assertTrue((structured_run / 'objective_plot.svg').exists())
            selection = json.loads((structured_run / 'best_validation.json').read_text())
            self.assertEqual(selection['loss_settings']['mode'], 'spectrogram')
            self.assertTrue(np.isfinite(selection['selection_objective']))
            # Reloaded exported encoder matches the original frozen encoder parameters exactly.
            from spectrogram_rae import load_model
            restored, _ = load_model(handoff / 'inference.yaml', torch.device('cpu'))
            for k, value in encoder_before.items():
                torch.testing.assert_close(restored.encoder.state_dict()[k], value, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
