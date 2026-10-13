---
purpose: "Describe the version-1 gencine conditioning fixtures."
parent_docs:
  - path: "../README.md"
    role: "Runtime installation and execution."
child_docs: []
---

# Encoded examples

`manifest.json` lists the active examples and hashes each `metadata.json`.
Metadata hashes the accompanying `inputs.safetensors` and records the prompt,
selected variant (0), seed, settings, and `gencine` weight bundle.

| Example | Dataset | Control kernels | Selected trajectory columns |
|---|---|---:|---:|
| `kitchen` | SpatialVID-HQ | 2 | 2,067 |
| `hike` | DAVIS | 1 | 3,528 |
| `bus` | DAVIS | 1 | 513 |
| `bear` | DAVIS | 2 | 1,074 |
| `rhino` | DAVIS | 1 | 985 |
| `cows` | DAVIS | 1 | 672 |
| `camel` | DAVIS | 1 | 494 |
| `dog` | DAVIS | 3 | 2,912 |

| Tensor | Shape | Role |
|---|---|---|
| `context` | 1×512×4096 | Positive T5 embedding |
| `negative_context` | 1×512×4096 | Empty-prompt embedding for CFG |
| `clip_feature` | 1×257×1280 | Encoded reference CLIP features |
| `reference_latents` | 1×16×1×60×104 | Reference prefix input |
| `vace_context` | 1×60×6×30×52 | Selected dense geometry condition |
| `y` | 1×32×6×60×104 | Full-resolution identity latents followed by zeros |
| `scale_cond` | 1×3 | Per-axis geometry scale ratio, cast to BF16 during inference |
| `initial_noise` | 1×16×6×60×104 | Seed-0 CUDA float32 noise converted to BF16 |

VACE channels retain this order: points (16), point mask (4), displacement
(16), displacement mask (4), partial identity (16), partial displacement mask
(4). The identity field inside VACE has half the spatial resolution of `y`.
Mask values and tensor dtypes are preserved. Target-video latents are excluded.
Each active example includes a genuine empty-prompt embedding for negative conditioning.
Point and displacement masks mark unknown regions; the partial displacement
mask marks the selected conditioning region.

Each cache contains conditioning for **21 frames**, encoded with T5, CLIP,
the reference VAE and the geometry VAE. The temporal latent grid is six frames.

Metadata selects 21 frames, 50 steps, shift 5, strength 1, VACE scale 1, CFG 3,
480×832, untiled decode, and MP4 FPS/quality 5. Frame count is determined by the
prepared inputs and metadata; another length requires a newly prepared cache.
`--seed N` regenerates noise at the native latent shape. Text edits do not alter
embeddings.

## Ground truth and reference trajectories

Each example directory contains these additional files, hashed in
`metadata.json.auxiliary_files`:

- `ground_truth.mp4`: aligned RGB frames 0–20, displayed at the same 5 FPS as
  generated videos. Original DAVIS clips are 20 FPS; kitchen is 24 FPS.
- `reference.png`: the exact first RGB frame used for image conditioning.
- `trajectory_overlay.mp4`: GT RGB with up to 96 fixed selected trajectory
  columns projected into each frame.
- `reference_tracks.npz`: the original 2D tracks and the selected 3D tracks used
  to construct the encoded geometry controls.

These are reference-video trajectory **estimates** from CoWTracker and ViPE,
not annotated ground-truth 3D trajectories. Ground-truth RGB and auxiliary
trajectory files are for inspection; inference consumes `inputs.safetensors`.

| Archive field | Meaning |
|---|---|
| `frame_indices` | Native frame indices 0–20 |
| `traj_2d`, `source_traj_vis` | Original aligned pixel tracks and tracker visibility |
| `selected_traj_3d_cam0` | `(21,N,3)` selected positions in camera-0 coordinates |
| `selected_traj_vis` | `(21,N)` valid support mask; ignore positions where false |
| `selected_source_ids` | Original trajectory column for each selected column |
| `selected_kernel_ids` | Control kernel owning each selected column |
| `intrinsics`, `world_to_camera` | Per-frame camera calibration and transforms |

Some selected columns can share an original trajectory ID across different
kernels. Invalid selected positions can contain infinity; always apply
`selected_traj_vis`. Position units follow the estimated reconstruction.

All eight examples contain nonempty controls.
