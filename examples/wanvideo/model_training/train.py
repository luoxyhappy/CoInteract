"""
CoInteract Training Script

Trains the Human-Aware MoE model with:
- Spatially-Structured Co-Generation (RGB + HOI depth dual-stream)
- Human-Aware Mixture-of-Experts (hand_expert + face_expert + router)
- Audio Face Mask (audio cross-attention controls face region only)
- Joint Encoding strategy (motion + input concatenated before VAE to avoid flickering)
- Router supervision via spatial GT bounding boxes

Usage:
    deepspeed --num_gpus=8 examples/wanvideo/model_training/train.py \
        --model_paths '<json_list_of_model_paths>' \
        --use_moe --expert_hidden_dim 1280 \
        --use_audio_face_mask --audio_mask_train_source gt \
        --use_hoi_branch --depth_mutual_visible \
        --dataset_metadata_path data.csv \
        --extra_inputs person_image,product_image \
        --output_path ./output
"""
import torch
import os
import json
import math
import random

from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import (
    DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
)
from diffsynth.trainers.unified_dataset import (
    UnifiedDataset, LoadVideo, LoadVideoFramesOnly, LoadAudio, LoadImage,
    ImageCropAndResize, ImageResizeAndPad, ToAbsolutePath
)
from diffsynth.utils.bbox_utils import create_masks_from_metadata

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    """
    Training module for CoInteract with MoE, Audio Face Mask, and Depth Branch.
    """

    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=32,
        lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        height=480,
        width=832,
        use_moe=False,
        expert_hidden_dim=None,
        use_depth_branch=False,
        depth_mutual_visible=True,
        use_deepspeed_activation_checkpointing=False,
        train_shift=None,
        use_audio_face_mask=False,
        audio_mask_train_source="gt",
    ):
        super().__init__()

        # Parse and load model weights
        model_configs = self.parse_model_configs(
            model_paths, model_id_with_origin_paths, enable_fp8_training=False
        )
        audio_model_configs = ModelConfig(
            path=os.environ.get("AUDIO_ENCODER_DIR", ""),
            offload_dtype=None
        )
        tokenizer_config = ModelConfig(
            path=os.environ.get("TOKENIZER_DIR", ""),
            offload_dtype=None
        )
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_model_configs,
        )

        # --- Enable Human-Aware MoE FFN ---
        # Must be done after loading weights but before adding LoRA
        if use_moe:
            if hasattr(self.pipe, 'dit') and hasattr(self.pipe.dit, 'enable_moe_ffn'):
                self.pipe.dit.enable_moe_ffn(expert_hidden_dim=expert_hidden_dim)
                print(f"[MoE] Enabled with expert_hidden_dim={expert_hidden_dim}")
                # Update LoRA target paths for MoE structure
                if 'ffn.0' in lora_target_modules or 'ffn.2' in lora_target_modules:
                    lora_target_modules = lora_target_modules.replace(
                        'ffn.0', 'ffn.ffn_base.0'
                    ).replace('ffn.2', 'ffn.ffn_base.2')
                    print(f"[MoE] Updated lora_target_modules: {lora_target_modules}")

        # --- Enable Audio Face Mask ---
        # Constrains audio cross-attention residual to face region only
        if use_audio_face_mask:
            if hasattr(self.pipe, 'dit') and hasattr(self.pipe.dit, 'set_audio_face_mask_config'):
                self.pipe.dit.set_audio_face_mask_config(
                    use_audio_face_mask=True,
                    audio_mask_train_source=audio_mask_train_source,
                )
                print(f"[Audio Face Mask] Enabled (source={audio_mask_train_source})")

        # --- Enable Depth Video Branch (Spatially-Structured Co-Generation) ---
        self.use_depth_branch = use_depth_branch
        if use_depth_branch:
            if hasattr(self.pipe, 'dit') and hasattr(self.pipe.dit, 'enable_depth_branch'):
                self.pipe.dit.enable_depth_branch()
                print("[Depth Branch] Enabled")

        # Switch to training mode: apply LoRA / freeze layers
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank,
            lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
            use_moe=use_moe,
            use_depth_branch=use_depth_branch,
            train_shift=train_shift,
        )

        # --- Set MoE experts + router to full training ---
        if use_moe and hasattr(self.pipe, 'dit'):
            moe_param_count = 0
            for block in self.pipe.dit.blocks:
                if hasattr(block, 'ffn') and hasattr(block.ffn, 'router'):
                    for name, param in block.ffn.named_parameters():
                        if 'hand_expert' in name or 'face_expert' in name:
                            param.requires_grad = True
                            moe_param_count += param.numel()
                    for param in block.ffn.router.parameters():
                        param.requires_grad = True
                        moe_param_count += param.numel()
            print(f"[MoE] {moe_param_count:,} params set to full training")

        # --- Set Depth Branch parameters to full training ---
        if use_depth_branch and hasattr(self.pipe, 'dit') and self.pipe.dit.is_depth_branch_enabled():
            depth_param_count = 0
            for attr in ['depth_patch_embedding', 'depth_head', 'depth_cond_mask']:
                module = getattr(self.pipe.dit, attr, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = True
                        depth_param_count += param.numel()
            for block in self.pipe.dit.blocks:
                if hasattr(block, 'depth_modulation'):
                    block.depth_modulation.requires_grad = True
                    depth_param_count += block.depth_modulation.numel()
            print(f"[Depth Branch] {depth_param_count:,} params set to full training")

        # Store training configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.use_deepspeed_activation_checkpointing = use_deepspeed_activation_checkpointing
        self.extra_inputs = extra_inputs.split(",") if extra_inputs else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.height = height
        self.width = width
        self.depth_mutual_visible = depth_mutual_visible

        # Router supervision config
        self.enable_router_supervision = True
        self.vae_scale_factor = 8
        self.patch_size = (2, 2)

    def forward_preprocess(self, data):
        """Preprocess a training sample into pipeline inputs."""
        # Prompt with 10% CFG dropout
        inputs_posi = {"prompt": data["prompt"] if random.random() > 0.1 else ""}
        inputs_nega = {}

        num_frames = 81
        motion_video = data.get("motion_video", [])
        input_video = data.get("input_video", None)
        person_image = data["person_image"]

        # Motion dropout (40% probability)
        motion_dropped = random.random() < 0.4
        if motion_dropped:
            motion_video = []

        if input_video is None:
            raise ValueError(
                f"input_video is None for sample: {data.get('prompt', 'N/A')}. "
                "Check your dataset CSV."
            )

        # --- Joint Encoding Strategy ---
        # Concatenate motion_video + input_video before VAE encode to maintain
        # temporal continuity (Causal Conv3d) and avoid flickering artifacts.
        has_motion_video = len(motion_video) > 0
        if has_motion_video:
            # Combine: motion (73 frames) + input[:80] = 153 frames total
            # VAE encodes jointly -> 39 latents (19 motion + 20 target)
            input_video = motion_video + input_video[:80]
            motion_video = []
        else:
            # No motion context: use person_image as first frame
            input_video[0] = person_image

        # HOI video for Spatially-Structured Co-Generation
        # CSV column is `hoi_video` (user-facing name); internally we still route
        # it through the pipeline under the legacy `depth_video` key.
        depth_video = data.get("hoi_video", None)
        if depth_video is not None and not isinstance(depth_video, float) and len(depth_video) > 0:
            pass  # Use as-is
        else:
            depth_video = None

        # Pose video for pose-driven conditioning (optional).
        # CSV empty cells are parsed as NaN (float), so explicitly filter them out.
        pose_video = data.get("pose_video", None)
        if pose_video is not None and not isinstance(pose_video, float) and len(pose_video) > 0:
            # 50% dropout to support CFG-style pose guidance at inference time
            if random.random() < 0.5:
                pose_video = None
        else:
            pose_video = None

        inputs_shared = {
            "input_video": input_video,
            "input_audio": data["audio"],
            "height": self.height,
            "width": self.width,
            "num_frames": num_frames,
            "audio_embeds": None,
            "s2v_pose_latents": None,
            "s2v_pose_video": pose_video,
            "motion_video": motion_video,
            "depth_video": depth_video,
            "start_idx": 0,
            # Pipeline config (do not modify)
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }

        # Handle extra inputs (person_image, product_image, etc.)
        for extra_input in self.extra_inputs:
            if extra_input == "person_image":
                # Pipeline I2V units internally use the "input_image" key.
                inputs_shared["input_image"] = data["person_image"]
            elif extra_input == "product_image":
                inputs_shared["product_image"] = data.get("product_image", None)
                product_image_scale = data.get("scale", 1.0)
                if product_image_scale is None or (isinstance(product_image_scale, float) and math.isnan(product_image_scale)):
                    product_image_scale = 1.0
                inputs_shared["product_image_scale"] = product_image_scale
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input in ("reference_image", "vace_reference_image"):
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]

        # Run pipeline units (VAE encode, text encode, audio encode, etc.)
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )

        # Audio CFG dropout (10%)
        if random.random() < 0.1:
            inputs_posi["audio_embeds"] = inputs_nega["audio_embeds"].clone()

        # --- Build spatial masks for MoE Router Supervision ---
        if self.enable_router_supervision:
            face_bbox_str = data.get("face", "")
            hand_bbox_str = data.get("hand_object", "")
            orig_height = data.get("height", None)
            orig_width = data.get("width", None)

            latent_h = self.height // self.vae_scale_factor
            latent_w = self.width // self.vae_scale_factor
            num_frames_latent = 20  # (81 - 1) // 4

            if orig_height is None or orig_width is None:
                raise ValueError(
                    "CSV missing 'height'/'width' fields required for bbox conversion."
                )

            face_mask, hand_mask = create_masks_from_metadata(
                face_bbox_str=face_bbox_str,
                hand_bbox_str=hand_bbox_str,
                num_frames=num_frames_latent,
                latent_h=latent_h,
                latent_w=latent_w,
                patch_size=self.patch_size,
                original_size=(orig_height, orig_width),
                target_size=(self.height, self.width),
                vae_scale_factor=self.vae_scale_factor,
                device=self.pipe.device,
            )
            inputs_shared["face_mask"] = face_mask
            inputs_shared["hand_mask"] = hand_mask

        inputs_shared["depth_mutual_visible"] = self.depth_mutual_visible
        return {**inputs_shared, **inputs_posi}

    def forward(self, data, inputs=None):
        """Compute training loss."""
        if inputs is None:
            inputs = self.forward_preprocess(data)

        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}

        face_mask = inputs.pop("face_mask", None)
        hand_mask = inputs.pop("hand_mask", None)
        depth_context = inputs.pop("depth_context", None)

        loss = self.pipe.training_loss(
            **models, **inputs,
            face_mask=face_mask,
            hand_mask=hand_mask,
            depth_context=depth_context,
            use_deepspeed_activation_checkpointing=self.use_deepspeed_activation_checkpointing,
        )
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()

    # Define data fields to load from CSV
    data_file_keys = (
        "motion_video,input_video,audio,person_image,"
        "product_image,pose_video,face,hand_object,height,width,scale"
    ).split(",")
    if args.use_depth_branch:
        data_file_keys.append("hoi_video")

    # Data loading operators for each CSV field
    special_operator_map = {
        "motion_video": (
            ToAbsolutePath(args.dataset_base_path)
            >> LoadVideoFramesOnly(73, frame_processor=ImageCropAndResize(
                args.height, args.width, None, 16, 16))
        ),
        "input_video": (
            ToAbsolutePath(args.dataset_base_path)
            >> LoadVideoFramesOnly(args.num_frames, frame_processor=ImageCropAndResize(
                args.height, args.width, None, 16, 16))
        ),
        "person_image": (
            ToAbsolutePath(args.dataset_base_path)
            >> LoadImage()
            >> ImageCropAndResize(args.height, args.width, None, 16, 16)
        ),
        "audio": (
            ToAbsolutePath(args.dataset_base_path)
            >> LoadAudio(sample_rate=16000)
        ),
        "product_image": (
            ToAbsolutePath(args.dataset_base_path)
            >> LoadImage()
            >> ImageResizeAndPad(args.height, args.width, args.max_pixels, 16, 16)
        ),
        # Optional pose skeleton video (DWPose); leave the CSV cell empty to skip.
        "pose_video": (
            ToAbsolutePath(args.dataset_base_path)
            >> LoadVideoFramesOnly(args.num_frames, frame_processor=ImageCropAndResize(
                args.height, args.width, None, 16, 16))
        ),
        # Metadata fields (pass-through from CSV)
        "face": lambda x: x,
        "hand_object": lambda x: x,
        "height": lambda x: int(x) if x else None,
        "width": lambda x: int(x) if x else None,
        "scale": lambda x: float(x) if x and str(x) != 'nan' else 1.0,
    }

    if args.use_depth_branch:
        special_operator_map["hoi_video"] = (
            ToAbsolutePath(args.dataset_base_path)
            >> LoadVideoFramesOnly(args.num_frames, frame_processor=ImageCropAndResize(
                args.height, args.width, None, 16, 16))
        )

    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=data_file_keys,
        main_data_operator=None,
        special_operator_map=special_operator_map,
    )

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        height=args.height,
        width=args.width,
        use_moe=args.use_moe,
        expert_hidden_dim=args.expert_hidden_dim,
        use_depth_branch=args.use_depth_branch,
        depth_mutual_visible=args.depth_mutual_visible,
        use_deepspeed_activation_checkpointing=args.use_deepspeed_activation_checkpointing,
        train_shift=args.train_shift,
        use_audio_face_mask=args.use_audio_face_mask,
        audio_mask_train_source=args.audio_mask_train_source,
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )

    launch_training_task(dataset, model, model_logger, args=args)
