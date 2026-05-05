"""
CoInteract Batch Inference Script

Generate speech-driven human-object interaction videos from a CSV file.
Each row in the CSV should contain: audio, person_image, prompt columns.
Optional columns: prompt2, prompt3, product_image, scale, pose_video.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import torch
import numpy as np
from PIL import Image
import librosa
import pandas as pd
from pathlib import Path
import torchvision.transforms.functional as TF

from diffsynth import VideoData, save_video_with_audio
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig, WanVideoUnit_S2V
from diffsynth.models.utils import load_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="CoInteract Batch Inference")
    # Model paths
    parser.add_argument("--base_model_path", type=str, default="./models/Wan2.2-S2V-14B",
                        help="Path to Wan2.2-S2V-14B base model directory")
    parser.add_argument("--audio_encoder_path", type=str, default="./models/chinese-wav2vec2-large",
                        help="Path to chinese-wav2vec2-large model directory")
    parser.add_argument("--lora_path", type=str, default="./models/CoInteract/checkpoint.safetensors",
                        help="Path to LoRA checkpoint (safetensors)")
    parser.add_argument("--lora_alpha", type=float, default=1.0,
                        help="LoRA alpha scale factor")
    # MoE config
    parser.add_argument("--use_moe", action="store_true", default=True,
                        help="Enable Human-Aware MoE FFN")
    parser.add_argument("--expert_hidden_dim", type=int, default=256,
                        help="Hidden dimension for MoE expert networks")
    parser.add_argument("--use_audio_face_mask", action="store_true", default=False,
                        help="Enable Audio Face Mask (audio controls face region only)")
    # Generation config
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to input CSV file")
    # Project root: directory where this script lives
    _PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument("--data_base_path", type=str, default=_PROJECT_ROOT,
                        help="Base path for resolving relative paths in CSV (default: project root)")
    parser.add_argument("--output_dir", type=str, default="./output_videos",
                        help="Output directory for generated videos")
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=80,
                        help="Number of frames per clip (80 frames = 3.24 seconds at 25fps)")
    parser.add_argument("--num_clips", type=int, default=3,
                        help="Number of clips to generate per sample")
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--cfg_scale", type=float, default=7.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--sigma_shift", type=float, default=7.0,
                        help="Noise schedule time shift parameter")
    parser.add_argument("--negative_prompt", type=str,
                        default="Blurry, worst quality, blurred details, static frame, "
                                "violent emotions, rapid hand shaking, subtitles, ugly, "
                                "deformed, extra fingers, poorly drawn hands, poorly drawn face")
    return parser.parse_args()


def resize_and_pad(image: Image.Image, target_height: int, target_width: int,
                   pad_color=(0, 0, 0)) -> Image.Image:
    """
    Resize image preserving aspect ratio, then pad to target size (center-aligned).
    Consistent with training-time ImageResizeAndPad preprocessing.
    """
    width, height = image.size
    scale = min(target_width / width, target_height / height)
    scale = scale / 1.25  # Scale down slightly to avoid cropping artifacts

    new_width = round(width * scale)
    new_height = round(height * scale)

    interpolation = (TF.InterpolationMode.LANCZOS if scale < 1
                     else TF.InterpolationMode.BILINEAR)
    image = TF.resize(image, (new_height, new_width), interpolation=interpolation)

    pad_left = (target_width - new_width) // 2
    pad_right = target_width - new_width - pad_left
    pad_top = (target_height - new_height) // 2
    pad_bottom = target_height - new_height - pad_top
    image = TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=pad_color)

    return image


def speech_to_video(pipe, prompts, person_image, audio_path,
                    product_image=None, product_image_scale=1.0,
                    negative_prompt="", num_clips=None,
                    audio_sample_rate=16000, pose_video_path=None,
                    num_frames=80, height=448, width=832,
                    num_inference_steps=40, fps=25, motion_frames=73,
                    save_path=None, sigma_shift=5.0, cfg_scale=5.0):
    """
    Generate video from speech audio and reference image.
    
    Args:
        prompts: list of prompt strings, one per clip. If fewer prompts than clips,
                 the last prompt is reused for remaining clips.
    """
    input_audio, sample_rate = librosa.load(audio_path, sr=audio_sample_rate)
    pose_video = (VideoData(pose_video_path, height=height, width=width)
                  if pose_video_path else None)

    audio_embeds, pose_latents, num_repeat = WanVideoUnit_S2V.pre_calculate_audio_pose(
        pipe=pipe,
        input_audio=input_audio,
        audio_sample_rate=sample_rate,
        s2v_pose_video=pose_video,
        num_frames=num_frames + 1,
        height=height,
        width=width,
        fps=fps,
    )
    num_repeat = min(num_repeat, num_clips) if num_clips is not None else num_repeat
    print(f"Generating {num_repeat} video clip(s)...")

    motion_videos = []
    video = []

    for r in range(num_repeat):
        current_prompt = prompts[min(r, len(prompts) - 1)]

        s2v_pose_latents = pose_latents[r] if pose_latents is not None else None
        current_clip = pipe(
            prompt=current_prompt,
            person_image=person_image,
            product_image=product_image,
            product_image_scale=product_image_scale,
            negative_prompt=negative_prompt,
            seed=r,
            num_frames=num_frames + 1,
            height=height,
            width=width,
            audio_embeds=audio_embeds[r],
            s2v_pose_latents=s2v_pose_latents,
            motion_video=motion_videos,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            cfg_scale=cfg_scale,
        )
        current_clip = current_clip[-num_frames:]

        overlap_frames_num = min(motion_frames, len(current_clip))
        motion_videos = motion_videos[overlap_frames_num:] + current_clip[-overlap_frames_num:]
        video.extend(current_clip)
        save_video_with_audio(video, save_path, audio_path, fps=25, quality=5)
        print(f"  Processed clip {r+1}/{num_repeat}")

    return video


def load_moe_weights(model, state_dict, device="cuda", dtype=torch.bfloat16):
    """Load MoE-specific weights (router, hand_expert, face_expert) from checkpoint."""
    moe_keys = ['router', 'hand_expert', 'face_expert']
    moe_state_dict = {}

    for key in state_dict:
        if not any(k in key for k in moe_keys):
            continue
        model_key = key[len('diffusion_model.'):] if key.startswith('diffusion_model.') else key
        moe_state_dict[model_key] = state_dict[key].to(device=device, dtype=dtype)

    if moe_state_dict:
        model.load_state_dict(moe_state_dict, strict=False)
        print(f"[MoE] Loaded {len(moe_state_dict)} MoE weights")
    else:
        print("[MoE] Warning: No MoE weights found in checkpoint")

    return len(moe_state_dict)


def load_model(args):
    """Load pipeline with all model components."""
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=[
                f"{args.base_model_path}/diffusion_pytorch_model-00001-of-00004.safetensors",
                f"{args.base_model_path}/diffusion_pytorch_model-00002-of-00004.safetensors",
                f"{args.base_model_path}/diffusion_pytorch_model-00003-of-00004.safetensors",
                f"{args.base_model_path}/diffusion_pytorch_model-00004-of-00004.safetensors",
            ]),
            ModelConfig(path=f"{args.base_model_path}/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(path=f"{args.audio_encoder_path}/pytorch_model.bin"),
            ModelConfig(path=f"{args.base_model_path}/Wan2.1_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(path=f"{args.base_model_path}/google/umt5-xxl"),
        audio_processor_config=ModelConfig(path=args.audio_encoder_path),
    )

    # Enable Human-Aware MoE FFN
    if args.use_moe:
        if hasattr(pipe.dit, 'enable_moe_ffn'):
            pipe.dit.enable_moe_ffn(expert_hidden_dim=args.expert_hidden_dim)
            print(f"[MoE] Enabled with expert_hidden_dim={args.expert_hidden_dim}")

    # Enable Audio Face Mask
    if args.use_audio_face_mask and args.use_moe:
        if hasattr(pipe.dit, 'set_audio_face_mask_config'):
            pipe.dit.set_audio_face_mask_config(
                use_audio_face_mask=True,
                audio_mask_train_source="router",
            )
            print("[Audio Face Mask] Enabled (source=router)")

    # Load LoRA + MoE weights
    if args.lora_path is not None:
        print(f"Loading checkpoint: {args.lora_path}")
        pipe.load_lora(module=pipe.dit, lora_config=args.lora_path, alpha=args.lora_alpha)
        print(f"  LoRA loaded (alpha={args.lora_alpha})")

        if args.use_moe:
            ckpt_state = load_state_dict(args.lora_path, torch_dtype=torch.bfloat16, device="cuda")
            load_moe_weights(pipe.dit, ckpt_state, device="cuda", dtype=torch.bfloat16)
            pipe.dit = pipe.dit.to(device="cuda", dtype=torch.bfloat16)

    # Enable VRAM management: automatically offload idle models (text_encoder, vae,
    # audio_encoder) to CPU and only keep the active model on GPU.
    pipe.vram_management_enabled = True

    return pipe


def process_batch(args, pipe):
    """Process all samples from the CSV file."""
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.csv_path)
    total = len(df)
    print(f"Total samples: {total}")

    for idx, row in df.iterrows():
        try:
            audio_rel = row['audio']
            image_rel = row['person_image']

            # Read prompts: prompt (required), prompt2/prompt3 (optional)
            prompt1 = str(row['prompt']).strip()
            prompt2 = str(row['prompt2']).strip() if 'prompt2' in df.columns and pd.notna(row.get('prompt2')) else None
            prompt3 = str(row['prompt3']).strip() if 'prompt3' in df.columns and pd.notna(row.get('prompt3')) else None

            # Build prompts list: [prompt1] or [prompt1, prompt2] or [prompt1, prompt2, prompt3]
            prompts = [prompt1]
            if prompt2:
                prompts.append(prompt2)
            if prompt3:
                prompts.append(prompt3)

            # Resolve paths
            audio_path = os.path.join(args.data_base_path, audio_rel)
            image_path = (image_rel if os.path.isabs(image_rel)
                          else os.path.join(args.data_base_path, image_rel))

            # Product reference image (optional column)
            product_ref_path = None
            if 'product_image' in df.columns:
                val = row.get('product_image')
                if val is not None and pd.notna(val) and str(val).strip():
                    product_ref_path = (str(val) if os.path.isabs(str(val))
                                        else os.path.join(args.data_base_path, str(val)))

            product_image_scale = float(row.get('scale', 1.0)) if 'scale' in df.columns else 1.0

            # Pose-driven reference video (optional column)
            pose_video_path = None
            if 'pose_video' in df.columns:
                val = row.get('pose_video')
                if val is not None and pd.notna(val) and str(val).strip():
                    pose_video_path = (str(val) if os.path.isabs(str(val))
                                       else os.path.join(args.data_base_path, str(val)))

            sample_name = Path(audio_rel).stem
            save_path = os.path.join(args.output_dir, f"{sample_name}.mp4")

            if os.path.exists(save_path):
                print(f"[{idx+1}/{total}] Skip (exists): {sample_name}")
                continue

            print(f"\n[{idx+1}/{total}] Processing: {sample_name}")
            person_image = Image.open(image_path).convert("RGB").resize((args.width, args.height))

            product_image = None
            if product_ref_path and os.path.exists(product_ref_path):
                product_image = resize_and_pad(
                    Image.open(product_ref_path).convert("RGB"),
                    target_height=args.height, target_width=args.width,
                    pad_color=(0, 0, 0)
                )

            speech_to_video(
                pipe=pipe,
                prompts=prompts,
                person_image=person_image,
                product_image=product_image,
                product_image_scale=product_image_scale,
                audio_path=audio_path,
                pose_video_path=pose_video_path,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_clips=args.num_clips,
                num_inference_steps=args.num_inference_steps,
                save_path=save_path,
                sigma_shift=args.sigma_shift,
                cfg_scale=args.cfg_scale,
            )
            print(f"  Saved: {save_path}")

        except Exception as e:
            print(f"[{idx+1}/{total}] Error: {e}")
            continue

    print(f"\nDone! Output: {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    print("Loading models...")
    pipe = load_model(args)
    print("\nStarting batch inference...")
    process_batch(args, pipe)
