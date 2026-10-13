"""Run local encoded examples: python -m gencine --help."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import imageio_ffmpeg
import numpy as np
from safetensors.torch import save_file
import torch
from tqdm import tqdm

from .checkpoint import load_models
from .inference import generate
from .io import load_example, rgb_to_frames, save_video, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument('--example')
    select.add_argument('--all', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, help='Override the saved CUDA float32 -> BF16 noise')
    parser.add_argument('--save-validation-tensors', action='store_true')
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'BF16 CUDA GPU required'
    torch.set_num_threads(8)
    manifest = json.loads(args.manifest.read_text())
    config = json.loads((args.weights / 'model.json').read_text())
    assert config['bundle_id'] == manifest['weight_bundle']
    weight_hashes = {name: sha256(args.weights / f'{name}.safetensors')
                     for name in tqdm(config['weight_sha256'], desc='Verify weights')}
    for name, digest in weight_hashes.items():
        assert digest == config['weight_sha256'][name], f'Weight checksum mismatch: {name}'
    started = time.perf_counter()
    models = load_models(args.weights)
    load_seconds = time.perf_counter() - started
    names = list(manifest['examples']) if args.all else [args.example]
    for name in names:
        inputs, metadata = load_example(args.manifest, name)
        folder = args.output / name if args.all else args.output
        folder.mkdir(parents=True, exist_ok=True)
        assert not (folder / 'run.json').exists(), f'Choose a new output directory: {folder}'
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        rgb, trace = generate(models, inputs, metadata['settings'], seed=args.seed, save_validation=args.save_validation_tensors)
        frames = rgb_to_frames(rgb)
        settings = metadata['settings']
        assert len(frames) == settings['num_frames'] and frames[0].size == (settings['width'], settings['height'])
        save_video(frames, folder / 'video.mp4', settings['fps'], settings['quality'])
        if args.save_validation_tensors:
            trace['rgb'] = rgb.cpu()
            save_file({k:v.contiguous() for k,v in trace.items()}, str(folder / 'validation.safetensors'))
            np.save(folder / 'rgb_uint8.npy', np.stack([np.asarray(f) for f in frames]))
        run = {'example_id':name, 'input_sha256':metadata['inputs_sha256'], 'weight_hashes':weight_hashes,
               'settings':settings,
               'noise_source':'saved initial_noise' if args.seed is None else 'CUDA float32 -> BF16',
               'effective_seed':settings['seed'] if args.seed is None else args.seed,
               'attention':'torch_sdpa', 'gpu':torch.cuda.get_device_name(), 'cuda':torch.version.cuda,
               'python':platform.python_version(), 'load_seconds':load_seconds,
               'run_seconds':time.perf_counter()-started, 'peak_vram_bytes':torch.cuda.max_memory_allocated(),
               'video_sha256':sha256(folder / 'video.mp4'), 'frame_count':len(frames),
               'dimensions':[settings['height'],settings['width']],
               'packages':{n:importlib.metadata.version(n) for n in ['gencine','torch','einops','safetensors','numpy','Pillow','tqdm','imageio','imageio-ffmpeg']},
               'ffmpeg':imageio_ffmpeg.get_ffmpeg_version(),
               'manifest_sha256':sha256(args.manifest),
               'model_config_sha256':sha256(args.weights / 'model.json'),
               'implementation_sha256':{str(p.relative_to(Path(__file__).parent)):sha256(p)
                                        for p in Path(__file__).parent.rglob('*.py')},
               'project_modules':{n:str(m.__file__) for n,m in sys.modules.items() if n.startswith('gencine') and getattr(m,'__file__',None)}}
        write_json(folder / 'run.json', run)
        write_json(folder / 'imported_modules.json',
                   {n:str(m.__file__) for n,m in sys.modules.items() if getattr(m,'__file__',None)})
        print(f'Completed {name}: {folder}', flush=True)


if __name__ == '__main__':
    main()
