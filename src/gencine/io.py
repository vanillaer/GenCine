"""Encoded input loading and RGB output conversion."""
import hashlib
import json
from pathlib import Path

from einops import reduce
import imageio
import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch
from tqdm import tqdm


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def load_example(manifest_path, name):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest['schema_version'] == 1
    entry = manifest['examples'][name]
    path = manifest_path.parent / entry['metadata']
    assert sha256(path) == entry['metadata_sha256']
    metadata = json.loads(path.read_text())
    assert metadata['schema_version'] == 1 and metadata['example_id'] == name
    assert metadata['weight_bundle'] == manifest['weight_bundle']
    tensor_path = path.parent / metadata['inputs']
    assert sha256(tensor_path) == metadata['inputs_sha256']
    tensors = load_file(str(tensor_path))
    assert set(tensors) == set(metadata['tensors'])
    for key, tensor in tensors.items():
        expected = metadata['tensors'][key]
        assert list(tensor.shape) == expected['shape'] and str(tensor.dtype) == expected['dtype'], key
        assert torch.isfinite(tensor).all(), key
    return tensors, metadata


def rgb_to_frames(rgb):
    video = reduce(rgb, 'B C T H W -> T H W C', reduction='mean')
    frames = []
    for image in video:
        image = ((image + 1) * (255 / 2)).clip(0, 255)
        image = image.to(device='cpu', dtype=torch.uint8)
        frames.append(Image.fromarray(image.numpy()))
    return frames


def save_video(frames, path, fps, quality):
    writer = imageio.get_writer(str(path), fps=fps, quality=quality)
    for frame in tqdm(frames, desc='Saving video'):
        writer.append_data(np.array(frame))
    writer.close()
