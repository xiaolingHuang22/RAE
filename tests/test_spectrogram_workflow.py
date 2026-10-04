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
from spectrogram_rae import train, export, decode, report
from stage1 import RAE


class WorkflowTests(unittest.TestCase):
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

    def test_complete_offline_workflow(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            for i in range(3):
                patient = data / f'patient_{i}'
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
            self.assertIn('Validation', (run / 'loss_plot.svg').read_text())
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
            # Reloaded exported encoder matches the original frozen encoder parameters exactly.
            from spectrogram_rae import load_model
            restored, _ = load_model(handoff / 'inference.yaml', torch.device('cpu'))
            for k, value in encoder_before.items():
                torch.testing.assert_close(restored.encoder.state_dict()[k], value, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
