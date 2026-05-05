# CoInteract Training Guide

This guide explains how to prepare a training CSV for CoInteract. A 5-sample demo dataset is provided at [`./demodataset/`](./demodataset/) — you can use it as a reference for directory layout and CSV format. 

> **Tip.** You can vibe-code the entire data-preparation pipeline, or simply disable MoE and HOI co-generation to greatly simplify the workflow.

---

## 1. Data Preparation

Per training sample (one CSV row):

| Item | Purpose |
| ---- | ------- |
| `input_video.mp4` | 81-frame target clip (the one you want to generate). |
| `motion_video.mp4` | The 73 frames **immediately preceding** `input_video`, used as motion context. Leave the column empty for the first clip of a source. |
| `audio.wav` | Aligned speech covering the 81-frame `input_video` window (≈ 3.24 s at 25 fps). |
| `person_image.jpg` | Reference image of the person. |
| `product_image.jpg` (optional) | Reference image of the interacting product. Clean / white background works best. |
| `hoi_video.mp4` (optional) | HOI-depth video aligned with `input_video`, used for **Spatially-Structured Co-Generation**. |
| `pose_video.mp4` (optional) | DWPose skeleton video aligned with `input_video`, used as a **pose-driven condition**. |
| `face` bbox (JSON string) | Per-frame face box — supervises the **face expert** in the Human-Aware MoE router. |
| `hand_object` bbox (JSON string) | Per-frame hand / object boxes — supervises the **hand expert** in the Human-Aware MoE router. |
| `height`, `width` | Original pixel size of the video (the space the bboxes live in). |
| `prompt` | Text description of the clip. |

---

## 2. How to Process the Raw Source

### Video clips
- **fps = 25** (hard-coded); other values break audio-visual alignment.
- `input_video`: 81 frames.
- `motion_video`: the 73 frames directly before `input_video`. Empty allowed.
- Use `ffmpeg` for frame-accurate cutting.

### Audio
- Mono, **16 kHz**, covering the 81-frame `input_video` window (≈ 3.24 s at 25 fps).

### Person / product images
Pick **any single frame** from `input_video` and feed it to an image-editing model (e.g. Qwen-Image-Edit / FLUX-based editors) to obtain:
- `person_image`: the same frame with the product removed / edited out, leaving a clean portrait of the person.
- `product_image`: the same frame with only the product isolated on a clean / white background.

Any resolution is fine; the loader resizes and pads internally.

### `face` / `hand_object` bboxes (for MoE router supervision)

These two columns drive the **Human-Aware MoE router**: `face` bboxes activate the `face_expert`, `hand_object` bboxes activate the `hand_expert`. We obtain both from [**hand_object_detector**](https://github.com/ddshan/hand_object_detector) — run it on each frame of `input_video` to get per-frame hand / object / face boxes, then serialize as JSON.

Both are JSON strings keyed by **latent frame index** (with `num_frames=81` and VAE temporal stride 4, valid keys are `frame_0` … `frame_19`). Coordinates are in the **original pixel space** (i.e. matching the `height` / `width` columns).

```json
// face
{"frame_2": [x1, y1, x2, y2], "frame_10": [x1, y1, x2, y2]}

// hand_object  (l_h / r_h / obj are all optional)
{"frame_2": {"l_h": [..], "r_h": [..], "obj": [..]},
 "frame_10": {"r_h": [..], "obj": [..]}}
```

Any stable face / hand / object detector works. It is fine to annotate only a subset of the 20 latent frames. Missing or malformed JSON is treated as empty (no supervision for that sample).

### `hoi_video` (for Spatially-Structured Co-Generation)

This is the auxiliary HOI stream that the model co-generates alongside the RGB video. We build it by:
1. Running [**SAM3**](https://github.com/facebookresearch/sam3) on `input_video` to obtain a per-frame mask of the interacting object.
2. Running [**SAM-3D-Body**](https://github.com/facebookresearch/sam-3d-body) on `input_video` to render a per-frame human mesh video.
3. Using the object mask from step (1) to composite the object on top of the rendered mesh video from step (2).

The resulting video must have the **same resolution, frame count, and fps** as `input_video`. If absent, the branch falls back to zero-conditioning. Enabled at training time via `--use_hoi_branch`.

> The user-facing CSV column is `hoi_video` and the CLI flag is `--use_hoi_branch`. `--use_depth_branch` is kept as a legacy alias for backward compatibility.

### `pose_video` (for pose-driven generation, optional)

An additional pose-conditioning video aligned with `input_video`. We extract it with [**DWPose**](https://github.com/IDEA-Research/DWPose) — run DWPose on each frame of `input_video` and render the resulting 2D skeleton as an RGB video of the **same resolution, frame count, and fps**. Leave the column empty to train a purely audio-driven model.

---

## 3. Final CSV Schema

Column names are case-sensitive. Paths are relative to `--dataset_base_path`.

| Column | Type | Required | Notes |
| ------ | ---- | -------- | ----- |
| `prompt` | str | ✅ | Text description. |
| `input_video` | path | ✅ | 81 frames @ 25 fps. |
| `motion_video` | path | ⚠️ empty for 1st clip | 73 frames preceding `input_video`. |
| `audio` | path | ✅ | Mono 16 kHz wav. |
| `person_image` | path | ✅ | Person reference image. |
| `product_image` | path | ❌ | Product reference image. |
| `scale` | float | ❌ | RoPE spatial scale for `product_image` (default `1.0`). |
| `hoi_video` | path | ❌ | HOI-depth video (for co-generation). |
| `pose_video` | path | ❌ | DWPose skeleton video (for pose-driven conditioning). |
| `face` | JSON str | ✅ | Face bbox for MoE face-expert supervision. |
| `hand_object` | JSON str | ✅ | Hand/object bbox for MoE hand-expert supervision. |
| `height` | int | ✅ | Original video height. |
| `width` | int | ✅ | Original video width. |

**Reference dataset**: see [`./demodataset/`](./demodataset/) for 5 fully-populated samples and a working `data.csv`.


## 4. Launch Training

Once your `data.csv` is ready, edit the paths in [`../../../train.sh`](../../../train.sh) and run:

```bash
cd Cointeract
bash train.sh
```
