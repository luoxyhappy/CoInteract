import torch, warnings, glob, os, types
import numpy as np
from PIL import Image
from einops import repeat, reduce
from typing import Optional, Union
from dataclasses import dataclass
from modelscope import snapshot_download
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal

from ..utils import BasePipeline, ModelConfig, PipelineUnit, PipelineUnitRunner
from ..models import ModelManager, load_state_dict
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_dit_s2v import rope_precompute
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_animate_adapter import WanAnimateAdapter
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
from ..lora import GeneralLoRALoader
import gc


class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.vace2: VaceWanModel = None
        self.animate_adapter: WanAnimateAdapter = None
        self.in_iteration_models = ("dit", "motion_controller", "vace", "animate_adapter")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace2", "animate_adapter")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_S2V(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            ###Product reference image embedder
            WanVideoUnit_ProductImageEmbedder(),
            WanVideoUnit_FunControl(),
            WanVideoUnit_FunReference(),
            WanVideoUnit_FunCameraControl(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_VACE(),
            WanVideoPostUnit_AnimateVideoSplit(),
            WanVideoPostUnit_AnimatePoseLatents(),
            WanVideoPostUnit_AnimateFacePixelValues(),
            WanVideoPostUnit_AnimateInpaint(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
        ]
        self.post_units = [
            WanVideoPostUnit_S2V(),
        ]
        self.model_fn = model_fn_wan_video
    
    def cleanup(self):
        """Explicitly clean up all cached resources."""
        # Clean VAE cache
        if hasattr(self.vae, 'cache') and self.vae.cache is not None:
            del self.vae.cache
            self.vae.cache = None
            
        # Clean video encoder cache
        if hasattr(self.video_encoder, 'cache') and self.video_encoder.cache is not None:
            del self.video_encoder.cache
            self.video_encoder.cache = None
            
        # Clean audio processor cache
        if hasattr(self.audio_processor, 'cache') and self.audio_processor.cache is not None:
            del self.audio_processor.cache
            self.audio_processor.cache = None
            
        # Release GPU memory
        torch.cuda.empty_cache()
        gc.collect()

 


    def load_lora(
        self,
        module: torch.nn.Module,
        lora_config: Union[ModelConfig, str] = None,
        alpha=1,
        hotload=False,
        state_dict=None,
    ):
        if state_dict is None:
            if isinstance(lora_config, str):
                lora = load_state_dict(lora_config, torch_dtype=self.torch_dtype, device=self.device)
            else:
                lora_config.download_if_necessary()
                lora = load_state_dict(lora_config.path, torch_dtype=self.torch_dtype, device=self.device)
        else:
            lora = state_dict
        if hotload:
            for name, module in module.named_modules():
                if isinstance(module, AutoWrappedLinear):
                    lora_a_name = f'{name}.lora_A.default.weight'
                    lora_b_name = f'{name}.lora_B.default.weight'
                    if lora_a_name in lora and lora_b_name in lora:
                        module.lora_A_weights.append(lora[lora_a_name] * alpha)
                        module.lora_B_weights.append(lora[lora_b_name])
        else:
            loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
            loader.load(module, lora, alpha=alpha)
    
    def _sample_depth_timestep(self, video_timestep, offset_ratio_min=0.1, offset_ratio_max=0.3):
        """
        Sample depth branch timestep (with randomness).
        
        Strategy: depth_timestep > video_timestep (higher noise level)
        so depth branch needs to rely more on video information for reconstruction.
        
        Args:
            video_timestep: Main video branch timestep (range: [0, num_train_timesteps))
            offset_ratio_min: Minimum offset ratio
            offset_ratio_max: Maximum offset ratio
        
        Returns:
            depth_timestep: Depth branch timestep (range: [0, num_train_timesteps))
        
        Note: In flow matching, larger timestep = more noise.
        depth_timestep = video_timestep + random_offset
        """
        num_train_timesteps = self.scheduler.num_train_timesteps
        
        # Randomly sample offset ratio
        offset_ratio = torch.rand(1).item() * (offset_ratio_max - offset_ratio_min) + offset_ratio_min
        
        # Compute offset
        offset = int(num_train_timesteps * offset_ratio)
        
        # depth_timestep should be larger than video_timestep (higher noise)
        depth_timestep_value = min(num_train_timesteps - 1, video_timestep.item() + offset)
        
        return torch.tensor([depth_timestep_value], dtype=video_timestep.dtype, device=video_timestep.device)

    #s2v training loss - fixed: reference frame should stay clean, only denoise target frames
    def training_loss(self, **inputs):
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        
        # Extract MoE Router supervision masks
        face_mask = inputs.pop("face_mask", None)
        hand_mask = inputs.pop("hand_mask", None)
        
        # Extract Depth Video latents (if present)
        depth_latents = inputs.pop("depth_latents", None)
        has_depth_latents = depth_latents is not None
        has_depth_method = hasattr(self.dit, 'is_depth_branch_enabled')
        depth_enabled = has_depth_method and self.dit.is_depth_branch_enabled() if has_depth_method else False
        enable_depth_branch = has_depth_latents and depth_enabled
        
        # Debug info
        if has_depth_latents:
            print(f"[Depth Debug] depth_latents shape: {depth_latents.shape}, has_method: {has_depth_method}, depth_enabled: {depth_enabled}, enable_depth_branch: {enable_depth_branch}")
        
        # S2V: Reference frame (frame 0) stays clean, only add noise to target frames
        # input_latents shape: [B, C, T, H, W], T = (num_frames-1)//4 + 1
        # Frame 0 is the reference image latent, frame 1+ are video frames to generate
        ref_latent = inputs["input_latents"][:, :, 0:1].clone()  # [B, C, 1, H, W] clean reference
        target_latents = inputs["input_latents"][:, :, 1:]       # [B, C, T-1, H, W] GT target frames
        target_noise = inputs["noise"][:, :, 1:]                 # [B, C, T-1, H, W] target noise only
        
        # Add noise to target frames only
        noisy_target = self.scheduler.add_noise(target_latents, target_noise, timestep)
        
        # Concat: clean reference + noisy target
        inputs["latents"] = torch.cat([ref_latent, noisy_target], dim=2)
        
        # training_target: only compute for target frames (flow = noise - sample)
        training_target = self.scheduler.training_target(target_latents, target_noise, timestep)
        
        # ========== Depth Video Branch Processing ==========
        depth_training_target = None
        depth_timestep = None
        if enable_depth_branch:
            # Depth Video: 81 frames -> 21 latents, use last 20 (aligned with target)
            # depth_latents shape: [B, C, 21, H, W]
            depth_target_latents = depth_latents[:, :, 1:]  # [B, C, 20, H, W] last 20 frames only
            
            # Sample depth timestep (simplified: same as video timestep)
            depth_timestep = timestep.clone()
            
            # Generate depth noise and add noise
            depth_noise = torch.randn_like(depth_target_latents)
            noisy_depth = self.scheduler.add_noise(depth_target_latents, depth_noise, depth_timestep)
            
            # Pass to model_fn
            inputs["depth_latents"] = noisy_depth
            inputs["depth_timestep"] = depth_timestep
            
            # Depth training target
            depth_training_target = self.scheduler.training_target(depth_target_latents, depth_noise, depth_timestep)
        
        # Model prediction: return router logits if MoE supervision is enabled
        # enable_router_supervision requires: 1) masks present 2) MoE enabled
        moe_enabled = hasattr(self.dit, 'is_moe_enabled') and self.dit.is_moe_enabled()
        enable_router_supervision = moe_enabled and face_mask is not None and hand_mask is not None
        
        # Extract depth_mutual_visible parameter
        depth_mutual_visible = inputs.pop("depth_mutual_visible", True)
        
        if enable_router_supervision:
            # MoE: pass face_mask and hand_mask for hard expert selection during training
            model_output = self.model_fn(
                **inputs, 
                timestep=timestep, 
                return_router_logits=True,
                is_conditional=True,
                face_mask=face_mask,
                hand_mask=hand_mask,
                depth_mutual_visible=depth_mutual_visible,
            )
            if isinstance(model_output, tuple):
                if len(model_output) == 3:
                    # Returns (video_pred, depth_pred, router_logits)
                    noise_pred, depth_pred, router_logits_list = model_output
                elif len(model_output) == 2:
                    # Could be (video_pred, router_logits) or (video_pred, depth_pred)
                    # Check second element type to distinguish
                    if isinstance(model_output[1], list):
                        noise_pred, router_logits_list = model_output
                        depth_pred = None
                    else:
                        noise_pred, depth_pred = model_output
                        router_logits_list = []
                else:
                    noise_pred = model_output[0]
                    router_logits_list = []
                    depth_pred = None
            else:
                noise_pred = model_output
                router_logits_list = []
                depth_pred = None
        else:
            # No MoE router supervision
            model_output = self.model_fn(**inputs, timestep=timestep, is_conditional=True, depth_mutual_visible=depth_mutual_visible)
            if isinstance(model_output, tuple):
                if len(model_output) == 2:
                    # Returns (video_pred, depth_pred)
                    noise_pred, depth_pred = model_output
                else:
                    noise_pred = model_output[0]
                    depth_pred = None
            else:
                noise_pred = model_output
                depth_pred = None
            router_logits_list = []
        
        # noise_pred shape: [B, C, T, H, W]
        # Only compute loss on target frames (from frame 1+), reference frame prediction excluded
        noise_pred_target = noise_pred[:, :, 1:]
        
        # Compute diffusion loss
        # if enable_router_supervision and face_mask is not None and hand_mask is not None:
        #     # 计算 per-pixel loss
        #     mse_loss_per_pixel = (noise_pred_target.float() - training_target.float()) ** 2  # [B, C, T, H, W]
            
        #     # 创建 spatial 权重 mask
        #     # face_mask, hand_mask shape: [N] where N = T * H_token * W_token
        #     # 需要 reshape 到 [T, H_token, W_token] 然后 upsample 到 [T, H, W]
        #     B, C, T, H, W = noise_pred_target.shape
        #     device = noise_pred_target.device
            
        #     # 从 token mask 恢复 spatial dimensions
        #     # latent: H = H_token * patch_h, W = W_token * patch_w
        #     # patch_size = 2 (默认)
        #     patch_h, patch_w = 2, 2
        #     H_token = H // patch_h
        #     W_token = W // patch_w
            
        #     # Reshape mask: [N] -> [T, H_token, W_token]
        #     seq_len_x = face_mask.shape[0]
        #     face_mask_3d = face_mask.reshape(T, H_token, W_token).float()  # [T, H_token, W_token]
        #     hand_mask_3d = hand_mask.reshape(T, H_token, W_token).float()  # [T, H_token, W_token]
            
        #     # Upsample 到 latent 分辨率: [T, H_token, W_token] -> [T, H, W]
        #     face_mask_spatial = face_mask_3d.repeat_interleave(patch_h, dim=1).repeat_interleave(patch_w, dim=2)  # [T, H, W]
        #     hand_mask_spatial = hand_mask_3d.repeat_interleave(patch_h, dim=1).repeat_interleave(patch_w, dim=2)  # [T, H, W]
            
        #     # 创建权重 map (手部优先)
        #     # 注意：MoE 1.2 中使用 1.5 权重可能导致训练不平衡，改为 1.0
        #     diffusion_face_weight = 2.0  # 面部区域 diffusion loss 权重
        #     diffusion_hand_weight = 2.0  # 手部区域 diffusion loss 权重
            
        #     weight_map = torch.ones(T, H, W, device=device)  # [T, H, W]
        #     weight_map = weight_map * (1.0 + (diffusion_face_weight - 1.0) * face_mask_spatial)  # 面部加权
        #     weight_map = weight_map * (1.0 + (diffusion_hand_weight - 1.0) * hand_mask_spatial)  # 手部额外加权
            
        #     # 应用权重: [B, C, T, H, W] * [1, 1, T, H, W]
        #     weighted_mse = mse_loss_per_pixel * weight_map.unsqueeze(0).unsqueeze(0)  # [B, C, T, H, W]
        #     diffusion_loss = weighted_mse.mean()
        #     diffusion_loss = diffusion_loss * self.scheduler.training_weight(timestep)
        # else:
        # Standard uniform-weight diffusion loss
        diffusion_loss = torch.nn.functional.mse_loss(noise_pred_target.float(), training_target.float())
        diffusion_loss = diffusion_loss * self.scheduler.training_weight(timestep)
        
        # ========== Depth Video Loss ==========
        depth_loss = torch.tensor(0.0, device=self.device)
        # 调试信息
        print(f"[Depth Loss Debug] enable_depth_branch: {enable_depth_branch}, depth_pred is None: {depth_pred is None}, depth_training_target is None: {depth_training_target is None}")
        if enable_depth_branch and depth_pred is not None and depth_training_target is not None:
            # depth_pred shape: [B, C, T, H, W]
            print(f"[Depth Loss Debug] depth_pred shape: {depth_pred.shape}, depth_training_target shape: {depth_training_target.shape}")
            # 检查 depth_pred 和 depth_training_target 的统计信息
            print(f"[Depth Loss Debug] depth_pred stats: min={depth_pred.min().item():.4f}, max={depth_pred.max().item():.4f}, mean={depth_pred.mean().item():.4f}")
            print(f"[Depth Loss Debug] depth_target stats: min={depth_training_target.min().item():.4f}, max={depth_training_target.max().item():.4f}, mean={depth_training_target.mean().item():.4f}")
            # 计算 MSE loss
            depth_mse_loss = torch.nn.functional.mse_loss(depth_pred.float(), depth_training_target.float())
            # 使用主 timestep 的 training_weight，因为 depth_timestep 可能超出范围
            depth_weight_value = self.scheduler.training_weight(timestep)
            depth_loss = depth_mse_loss * depth_weight_value
            print(f"[Depth Loss Debug] depth_mse_loss: {depth_mse_loss.item():.6f}, weight: {depth_weight_value.item():.6f}, depth_loss: {depth_loss.item():.6f}")
        
        # 计算 router loss（如果启用 MoE 监督）
        total_loss = diffusion_loss
        if enable_router_supervision and len(router_logits_list) > 0:
            from diffsynth.utils.bbox_utils import create_router_targets, create_loss_weights
            
            # 创建 router target distribution
            router_targets = create_router_targets(face_mask, hand_mask)  # [seq_len_x, 3]
            
            # 创建 loss 权重（给面部/手部 token 更高的权重）
            loss_weights = create_loss_weights(
                face_mask, hand_mask,
                base_weight=1.0,
                face_weight=10.0,
                hand_weight=10.0
            )  # [seq_len_x]
            
            # seq_len_x: 目标帧的 token 数量（不包含参考帧、额外参考帧、motion tokens）
            # mask 的长度就是目标帧的 token 数量
            seq_len_x = face_mask.shape[0]
            
            # 计算所有 block 的 router loss
            router_loss = 0.0
            num_valid_blocks = 0
            for router_logits in router_logits_list:
                # router_logits shape: [B, total_seq_len, 3]
                # total_seq_len = seq_len_x + ref_tokens + product_tokens + motion_tokens
                # 只取目标帧部分 [:, :seq_len_x, :] 计算 loss
                B, total_seq_len, C = router_logits.shape
                
                # 裁剪 router_logits，只保留目标帧部分
                # 参考帧和额外参考帧的预测不参与 loss 计算（和 diffusion loss 一致）
                router_logits_target = router_logits[:, :seq_len_x, :]  # [B, seq_len_x, 3]
                
                # 将 router_logits 展开为 [B*seq_len_x, 3]
                router_logits_flat = router_logits_target.reshape(-1, 3)  # [B*seq_len_x, 3]
                
                # router_targets 需要重复 B 次: [seq_len_x, 3] -> [B*seq_len_x, 3]
                router_targets_batch = router_targets.unsqueeze(0).expand(B, -1, -1).reshape(-1, 3)  # [B*seq_len_x, 3]
                
                # loss_weights 也需要重复 B 次: [seq_len_x] -> [B*seq_len_x]
                loss_weights_batch = loss_weights.unsqueeze(0).expand(B, -1).reshape(-1)  # [B*seq_len_x]
                
                # 计算加权的 cross-entropy loss
                # 使用 log_softmax + nll_loss 来手动计算加权 CE loss
                log_probs = torch.nn.functional.log_softmax(router_logits_flat, dim=-1)  # [B*seq_len_x, 3]
                # 手动计算 NLL: -sum(target * log_prob)
                nll = -(router_targets_batch * log_probs).sum(dim=-1)  # [B*seq_len_x]
                # 应用权重
                weighted_nll = nll * loss_weights_batch  # [B*seq_len_x]
                block_router_loss = weighted_nll.mean()
                
                router_loss += block_router_loss
                num_valid_blocks += 1
            
            if num_valid_blocks > 0:
                router_loss = router_loss / num_valid_blocks
                router_weight = 0.05  # 可以从配置中读取
                total_loss = diffusion_loss + router_weight * router_loss
                
                # 添加 depth loss
                if enable_depth_branch and depth_loss.item() > 0:
                    depth_weight = 0.5  # Depth loss 权重
                    total_loss = total_loss + depth_weight * depth_loss
                    # 输出主分支和 depth 分支的 timestep 和 loss
                    main_timestep_value = timestep.item()
                    depth_timestep_value = depth_timestep.item() if depth_timestep is not None else 0.0
                    print(f'[MoE+Depth Training] main_t: {main_timestep_value:.4f}, depth_t: {depth_timestep_value:.4f} | main_loss: {diffusion_loss.item():.4f}, router_loss: {router_loss.item():.4f}, depth_loss: {depth_loss.item():.4f}, total_loss: {total_loss.item():.4f}')
                    return {
                        "loss": total_loss,
                        "diffusion_loss": diffusion_loss,
                        "router_loss": router_loss,
                        "depth_loss": depth_loss,
                    }
                else:
                    print(f'[MoE Training] diffusion_loss: {diffusion_loss.item():.4f}, router_loss: {router_loss.item():.4f}, total_loss: {total_loss.item():.4f}')
                    return {
                        "loss": total_loss,
                        "diffusion_loss": diffusion_loss,
                        "router_loss": router_loss,
                    }
            else:
                print(f'[MoE Training] diffusion_loss: {diffusion_loss.item():.4f} (no router logits collected)')
                return {
                    "loss": diffusion_loss,
                    "diffusion_loss": diffusion_loss,
                }
        else:
            # 添加 depth loss
            if enable_depth_branch and depth_pred is not None:
                depth_weight = 0.5  # Depth loss 权重
                total_loss = diffusion_loss + depth_weight * depth_loss
                # 输出主分支和 depth 分支的 timestep 和 loss
                main_timestep_value = timestep.item()
                depth_timestep_value = depth_timestep.item() if depth_timestep is not None else 0.0
                print(f'[Depth Training] main_timestep: {main_timestep_value:.4f}, depth_timestep: {depth_timestep_value:.4f} | main_loss: {diffusion_loss.item():.4f}, depth_loss: {depth_loss.item():.4f}, total_loss: {total_loss.item():.4f}')
                return {
                    "loss": total_loss,
                    "diffusion_loss": diffusion_loss,
                    "depth_loss": depth_loss,
                }
            else:
                print('training loss:', total_loss.item())
        
        return {"loss": total_loss, "diffusion_loss": diffusion_loss}


    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                    torch.nn.Embedding: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit2 is not None:
            dtype = next(iter(self.dit2.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit2,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.vace,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.audio_encoder is not None:
            # TODO: need check
            dtype = next(iter(self.audio_encoder.parameters())).dtype
            enable_vram_management(
                self.audio_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
            
            
    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())
            
            
    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True

    def _collect_router_logits_from_dit(self, dit, cfg_merge=False):
        """
        从 DiT 模型的 audio injection blocks 收集 router logits（不平均，按 block 分别返回）
        
        只收集 audio injection 对应 block 的 router weights，用于可视化 audio face mask
        
        Args:
            dit: DiT 模型
            cfg_merge: 是否使用 cfg_merge 模式，如果 True 则 batch[0] 是 conditional，batch[1] 是 unconditional
        
        Returns:
            dict: {block_idx: (1, N, 3) softmax 权重} 或 None
                  block_idx 为 audio injection block 的索引
        """
        import torch.nn.functional as F
        
        # 获取 audio injection block 的索引集合
        audio_block_idxs = set()
        if hasattr(dit, 'audio_injector') and hasattr(dit.audio_injector, 'injected_block_id'):
            audio_block_idxs = set(dit.audio_injector.injected_block_id.keys())
        
        audio_block_weights = {}
        for block_idx, block in enumerate(dit.blocks):
            if block_idx not in audio_block_idxs:
                continue
            if hasattr(block, 'ffn') and hasattr(block.ffn, 'get_cached_router_logits'):
                router_logits = block.ffn.get_cached_router_logits()
                if router_logits is not None:
                    # 如果 cfg_merge=True，batch 维度是 2，只取 batch[0]（conditional branch）
                    if cfg_merge and router_logits.shape[0] > 1:
                        router_logits = router_logits[:1]  # (1, N, 3)
                    # 应用 softmax 得到权重
                    router_weights = F.softmax(router_logits, dim=-1)
                    audio_block_weights[block_idx] = router_weights.detach()
        
        return audio_block_weights if audio_block_weights else None
    
    def get_collected_router_logits_per_step(self):
        """
        获取每个 timestep 收集的 router logits（按 audio block 分别存储）
        
        Returns:
            tuple: (router_logits_list, timesteps_list)
                - router_logits_list: 每个元素为 dict {block_idx: (B, N, 3) softmax 权重}
                - timesteps_list: 对应的归一化 timestep 值 (0-1)
        """
        logits = getattr(self, '_collected_router_logits_per_step', None)
        timesteps = getattr(self, '_collected_timesteps', None)
        return logits, timesteps


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = False,
        use_usp=False,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp: pipe.initialize_usp()
        
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        vace = model_manager.fetch_model("wan_video_vace", index=2)
        if isinstance(vace, list):
            pipe.vace, pipe.vace2 = vace
        else:
            pipe.vace = vace
        pipe.audio_encoder = model_manager.fetch_model("wans2v_audio_encoder")
        pipe.animate_adapter = model_manager.fetch_model("wan_video_animate_adapter")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(use_usp=use_usp)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)

        if audio_processor_config is not None:
            audio_processor_config.download_if_necessary(use_usp=use_usp)
            from transformers import Wav2Vec2FeatureExtractor
            pipe.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_processor_config.path)
            #from infinitetalk  
            #from transformers import Wav2Vec2FeatureExtractor
            #pipe.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_processor_config.path)
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        return pipe

    #这里应该是inference调用的代码逻辑
    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video (person reference)
        person_image: Optional[Image.Image] = None,
        # Product reference image (RoPE position: 31)
        product_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Animate
        animate_pose_video: Optional[list[Image.Image]] = None,
        animate_face_video: Optional[list[Image.Image]] = None,
        animate_inpaint_video: Optional[list[Image.Image]] = None,
        animate_mask_video: Optional[list[Image.Image]] = None,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_scale_audio: Optional[float] = 4.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
        # Router logits 收集（推理时每个 timestep 收集）
        collect_router_logits_per_step: Optional[bool] = False,
        # Depth visibility mode: True=mutual visible (v1), False=only depth sees video (v2)
        depth_mutual_visible: Optional[bool] = True,
        # Product image RoPE spatial scale factor (from CSV scale field)
        product_image_scale: Optional[float] = 1.0,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # If collecting router logits, initialize storage list
        self._collected_router_logits_per_step = [] if collect_router_logits_per_step else None
        self._collected_timesteps = [] if collect_router_logits_per_step else None
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": person_image,
            "product_image": product_image,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale,"cfg_scale_audio": cfg_scale_audio, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio, "audio_sample_rate": audio_sample_rate, "s2v_pose_video": s2v_pose_video, "audio_embeds": audio_embeds, "s2v_pose_latents": s2v_pose_latents, "motion_video": motion_video,
            "animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video,
            "product_image_scale": product_image_scale,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        
        #borrow from LiveAIDiffStudio
        #inputs_null_text = {key:value  for key,value in inputs_nega.items() if value is not None}
        #inputs_null_text['audio_embeds'] = inputs_posi['audio_embeds']

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        
        # 检查是否启用了深度分支（Unity 风格）
        depth_branch_enabled = (
            hasattr(self.dit, 'is_depth_branch_enabled') and 
            self.dit.is_depth_branch_enabled()
        )
        
        # 初始化深度 latents（如果启用深度分支）
        depth_latents = None
        if depth_branch_enabled:
            latents_shape = inputs_shared["latents"].shape
            # 深度 latents 只包含目标帧部分（不包含参考帧）
            depth_latents = torch.randn(
                latents_shape[0], latents_shape[1], latents_shape[2] - 1,
                latents_shape[3], latents_shape[4],
                dtype=self.torch_dtype, device=self.device
            )
        
        # 如果需要收集 router logits，启用缓存
        if collect_router_logits_per_step and hasattr(self.dit, 'is_moe_enabled') and self.dit.is_moe_enabled():
            for block in self.dit.blocks:
                if hasattr(block, 'ffn') and hasattr(block.ffn, 'enable_router_logits_cache'):
                    block.ffn.enable_router_logits_cache(True)
        
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * self.scheduler.num_train_timesteps and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                models["vace"] = self.vace2
                
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # 如果启用深度分支，传递 depth_latents 和 depth_timestep
            if depth_branch_enabled and depth_latents is not None:
                inputs_shared["depth_latents"] = depth_latents
                inputs_shared["depth_timestep"] = timestep
            
            # Inference
            # depth_mutual_visible: True=相互可见(版本1), False=只有depth能看到video(版本2)
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep, is_conditional=True, depth_mutual_visible=depth_mutual_visible)
            
            # 处理返回值：可能是 tensor 或 (video_pred, depth_pred) 元组
            depth_pred_posi = None
            if isinstance(noise_pred_posi, tuple):
                if len(noise_pred_posi) == 2:
                    noise_pred_posi, depth_pred_posi = noise_pred_posi
                elif len(noise_pred_posi) == 3:
                    noise_pred_posi, depth_pred_posi, _ = noise_pred_posi
            
            # 如果启用了 router logits 收集，在 posi 推理后立即收集（只取 conditional branch）
            if collect_router_logits_per_step and hasattr(self.dit, 'is_moe_enabled') and self.dit.is_moe_enabled():
                step_router_logits = self._collect_router_logits_from_dit(self.dit, cfg_merge=cfg_merge)
                if step_router_logits is not None:
                    self._collected_router_logits_per_step.append(step_router_logits)
                    # 记录归一化的 timestep 值 (0-1)
                    t_normalized = timestep.item() / self.scheduler.num_train_timesteps
                    self._collected_timesteps.append(t_normalized)
            
            if cfg_scale != 1.0:
                if cfg_merge:
                    # MoE 模式下暂不支持 cfg_merge=True
                    # 因为 cfg_merge 会将 conditional 和 unconditional 合并成一个 batch
                    # 但 MoE 路由需要对两个分支使用不同的 is_conditional 参数
                    if hasattr(self.dit, 'is_moe_enabled') and self.dit.is_moe_enabled():
                        raise NotImplementedError(
                            "cfg_merge=True is not supported when MoE is enabled. "
                            "Please set cfg_merge=False (default) for MoE inference. "
                            "MoE routing requires different is_conditional flags for conditional and unconditional branches."
                        )
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                    depth_pred_posi_cfg, depth_pred_nega = None, None
                    if depth_pred_posi is not None:
                        depth_pred_posi_cfg, depth_pred_nega = depth_pred_posi.chunk(2, dim=0)
                        depth_pred_posi = depth_pred_posi_cfg
                else:
                    # 推理时 unconditional 分支也启用 MoE 路由（与 conditional 分支一致）
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep, is_conditional=True, depth_mutual_visible=depth_mutual_visible)
                    
                    # 处理 negative 分支返回值
                    depth_pred_nega = None
                    if isinstance(noise_pred_nega, tuple):
                        if len(noise_pred_nega) == 2:
                            noise_pred_nega, depth_pred_nega = noise_pred_nega
                        elif len(noise_pred_nega) == 3:
                            noise_pred_nega, depth_pred_nega, _ = noise_pred_nega
                
                # 主视频 CFG
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                
                # 深度视频 CFG（如果有）
                depth_pred = None
                if depth_pred_posi is not None and depth_pred_nega is not None:
                    depth_pred = depth_pred_nega + cfg_scale * (depth_pred_posi - depth_pred_nega)
                elif depth_pred_posi is not None:
                    depth_pred = depth_pred_posi
            else:
                noise_pred = noise_pred_posi
                depth_pred = depth_pred_posi

            # Scheduler - 主视频
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
            
            # Scheduler - 深度视频（如果有）
            if depth_branch_enabled and depth_pred is not None and depth_latents is not None:
                depth_latents = self.scheduler.step(
                    depth_pred,
                    self.scheduler.timesteps[progress_id],
                    depth_latents
                )
        
        # 如果启用了 router logits 收集，禁用缓存
        if collect_router_logits_per_step and hasattr(self.dit, 'is_moe_enabled') and self.dit.is_moe_enabled():
            for block in self.dit.blocks:
                if hasattr(block, 'ffn') and hasattr(block.ffn, 'enable_router_logits_cache'):
                    block.ffn.enable_router_logits_cache(False)
        
        # VACE
        if vace_reference_image is not None or (animate_pose_video is not None and animate_face_video is not None):
            if vace_reference_image is not None and isinstance(vace_reference_image, list):
                f = len(vace_reference_image)
            else:
                f = 1
            inputs_shared["latents"] = inputs_shared["latents"][:, :, f:]
        # post-denoising, pre-decoding processing logic
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        
        # 解码深度视频（如果启用深度分支）
        depth_video = None
        if depth_branch_enabled and depth_latents is not None:
            # 深度 latents 需要添加参考帧位置（用零填充）
            ref_latent = torch.zeros(
                depth_latents.shape[0], depth_latents.shape[1], 1,
                depth_latents.shape[3], depth_latents.shape[4],
                dtype=depth_latents.dtype, device=depth_latents.device
            )
            depth_latents_full = torch.cat([ref_latent, depth_latents], dim=2)
            
            depth_video_tensor = self.vae.decode(
                depth_latents_full,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride
            )
            depth_video = self.vae_output_to_video(depth_video_tensor)
        self.load_models_to_device([])

        # 如果启用了深度分支，返回 (video, depth_video) 元组
        if depth_branch_enabled and depth_video is not None:
            return video, depth_video
        return video



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        #对于帧数不满足下采样要求的,会被自动补帧在这里
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        #input_params匹配train.py中定义的input_shared中的key
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            f = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
            length += f
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)
        return {"noise": noise}
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "height", "width", "tiled", "tile_size", "tile_stride", "vace_reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, height, width, tiled, tile_size, tile_stride, vace_reference_image):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        
        # 检测是否是联合编码模式（训练时使用）
        # 联合编码：motion_video (73帧) + input_video[:80] = 153 帧
        # 普通模式：input_video 81 帧
        num_input_frames = len(input_video) if isinstance(input_video, list) else input_video.shape[2]
        is_joint_encoding = (num_input_frames == 153)
        
        # encode video
        input_video_tensor = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video_tensor, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        
        if is_joint_encoding:
            # ========== 联合编码模式（训练时使用）==========
            # 153帧 → 39 latents：前 19 个是 motion latent，后 20 个是 target latent
            # 需要把 motion_latents 提取出来，并在 target_latents 前面插入 ref latent 位置
            motion_latents_from_joint = input_latents[:, :, :19, :, :]  # [B, C, 19, H, W]
            target_latents_from_joint = input_latents[:, :, 19:, :, :]  # [B, C, 20, H, W]
            
            # 创建一个占位的 ref latent（后续会被 WanVideoUnit_ImageEmbedderFused 替换）
            # 这里用零填充，实际会被 input_image 的 latent 替换
            ref_latent_placeholder = torch.zeros_like(target_latents_from_joint[:, :, :1, :, :])
            
            # 拼接：ref_placeholder (1) + target_latents (20) = 21 latents
            # 这样和原来的 input_latents 结构一致
            input_latents = torch.cat([ref_latent_placeholder, target_latents_from_joint], dim=2)
            
            print(f"[Joint Encoding] 153 frames -> 39 latents")
            print(f"[Joint Encoding] motion_latents: {motion_latents_from_joint.shape}, target_latents: {target_latents_from_joint.shape}")
            print(f"[Joint Encoding] Reconstructed input_latents (with ref placeholder): {input_latents.shape}")
            
            # 注意：WanVideoUnit_S2V 在本 Unit 之前执行，所以需要在这里设置 motion_latents
            # 覆盖 WanVideoUnit_S2V 设置的空 motion_latents
            result = {
                "motion_latents": motion_latents_from_joint,
                "drop_motion_frames": False,
                "use_joint_encoding": True,
            }
        else:
            result = {}
        
        if vace_reference_image is not None:
            if not isinstance(vace_reference_image, list):
                vace_reference_image = [vace_reference_image]
            vace_reference_image = pipe.preprocess_video(vace_reference_image)
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        
        if pipe.scheduler.training:
            result.update({"latents": noise, "input_latents": input_latents})
            return result
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            result.update({"latents": latents})
            return result



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    # Depth prompt 前缀，与训练时保持一致
    # DEPTH_PROMPT_PREFIX = (
    #     "a video transformed into a depth map. "
    #     "Core (DEPTH MAP): per-pixel distance, near objects brighter (or warmer), "
    #     "far objects darker (or cooler), smooth surfaces, sharp depth edges, "
    #     "and stable values across frames. Following is the RGB description:"
    # )

    DEPTH_PROMPT_PREFIX = (
        "a rendered video on a black background showing a 3D human body mesh "
        "interacting with an object. The scene features the human body represented "
        "as a polygonal mesh with visible surface geometry, alongside a 3D object. "
        "Following is the RGB description:"
    )
    
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        
        result = {"context": prompt_emb}
        
        # 如果启用了 depth branch，同时编码 depth prompt
        # depth prompt 格式: "{DEPTH_PROMPT_PREFIX}{video_prompt}"
        if hasattr(pipe, 'dit') and hasattr(pipe.dit, 'is_depth_branch_enabled') and pipe.dit.is_depth_branch_enabled():
            depth_prompt = f"{self.DEPTH_PROMPT_PREFIX}{prompt}"
            depth_context = pipe.prompter.encode_prompt(depth_prompt, positive=positive, device=pipe.device)
            result["depth_context"] = depth_context
        
        return result



class WanVideoUnit_ImageEmbedder(PipelineUnit):
    """
    Deprecated
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("image_encoder", "vae")
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or pipe.image_encoder is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context, "y": y}



class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}
    


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}



class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    
    在训练模式下，会同时更新 latents 和 input_latents 的第一帧。
    在联合编码模式下，input_latents 的第一帧是 placeholder，需要被替换。
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "input_latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, latents, input_latents, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0: 1] = z
        
        # 训练模式下，同时更新 input_latents 的第一帧
        # 这对于联合编码模式尤其重要，因为 input_latents 的第一帧是 placeholder
        result = {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}
        if input_latents is not None:
            input_latents[:, :, 0:1] = z
            result["input_latents"] = input_latents
        
        return result


class WanVideoUnit_ProductImageEmbedder(PipelineUnit):
    """
    Encode product reference image to latents using VAE.
    Product image uses RoPE position 30-32, mask=1.
    """
    def __init__(self):
        super().__init__(
            input_params=("product_image", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, product_image, height, width, tiled, tile_size, tile_stride):
        if product_image is None:
            return {"product_image_latents": None}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(product_image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"product_image_latents": z}


class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "clip_feature", "y", "latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y, latents):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        y_dim = pipe.dit.in_dim-control_latents.shape[1]-latents.shape[1]
        if clip_feature is None or y is None:
            clip_feature = torch.zeros((1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.zeros((1, y_dim, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            y = y[:, -y_dim:]
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}
    


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, reference_image, height, width):
        if reference_image is None:
            return {}
        pipe.load_models_to_device(["vae"])
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        if pipe.image_encoder is None:
            return {"reference_latents": reference_latents}
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}



class WanVideoUnit_FunCameraControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "camera_control_direction", "camera_control_speed", "camera_control_origin", "latents", "input_image", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, camera_control_direction, camera_control_speed, camera_control_origin, latents, input_image, tiled, tile_size, tile_stride):
        if camera_control_direction is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        camera_control_plucker_embedding = pipe.dit.control_adapter.process_camera_coordinates(
            camera_control_direction, num_frames, height, width, camera_control_speed, camera_control_origin)
        
        control_camera_video = camera_control_plucker_embedding[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        control_camera_latents_input = control_camera_latents.to(device=pipe.device, dtype=pipe.torch_dtype)
        
        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

        if y.shape[1] != pipe.dit.in_dim - latents.shape[1]:
            image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
            y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]
            y = torch.cat([msk,y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}



class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("motion_bucket_id",))

    def process(self, pipe: WanVideoPipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"motion_bucket_id": motion_bucket_id}



class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("vace_video", "vace_video_mask", "vace_reference_image", "vace_scale", "height", "width", "num_frames", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_video, vace_video_mask, vace_reference_image, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride
    ):
        if vace_video is not None or vace_video_mask is not None or vace_reference_image is not None:
            pipe.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
            else:
                vace_video = pipe.preprocess_video(vace_video)
            
            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)
            
            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            
            vace_mask_latents = rearrange(vace_video_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            
            if vace_reference_image is None:
                pass
            else:
                if not isinstance(vace_reference_image,list):
                    vace_reference_image = [vace_reference_image]

                vace_reference_image = pipe.preprocess_video(vace_reference_image)

                bs, c, f, h, w = vace_reference_image.shape
                new_vace_ref_images = []
                for j in range(f):
                    new_vace_ref_images.append(vace_reference_image[0, :, j:j+1])
                vace_reference_image = new_vace_ref_images
                
                vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_reference_latents = [u.unsqueeze(0) for u in vace_reference_latents]

                vace_video_latents = torch.concat((*vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :f]), vace_mask_latents), dim=2)
            
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return {"vace_context": None, "vace_scale": vace_scale}



class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=())

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}



class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
        )

    def process(self, pipe: WanVideoPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}



class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
	    onload_model_names=("audio_encoder", "vae",)
	)
        self.video_rate = 25
   
    def process_audio(self, pipe: WanVideoPipeline, input_audio, audio_sample_rate, num_frames, fps=25, audio_embeds=None, return_all=False, start_index=0):
        if audio_embeds is not None:
            if return_all:
                return [audio_embeds] if not isinstance(audio_embeds, list) else audio_embeds
            return {"audio_embeds": audio_embeds}
        
        pipe.load_models_to_device(["audio_encoder"])
        audio_embeds = pipe.audio_encoder.get_audio_feats_per_inference(input_audio, audio_sample_rate, pipe.audio_processor, fps=fps, batch_frames=num_frames, m=0, dtype=pipe.torch_dtype, device=pipe.device, start_idx=start_index)
        
        if return_all:
            print(f"Audio processed: {len(audio_embeds)} clips, each with {num_frames} frames ({num_frames/fps:.2f}s per clip)")
            return audio_embeds
        
        if len(audio_embeds) < 2:
            print("audio shape < 2", len(audio_embeds))
            return {"audio_embeds": audio_embeds[0]}
        print("audio embeds shape", audio_embeds[1].shape)
        return {"audio_embeds": audio_embeds[1]}
        
    def process_motion_latents(self, pipe: WanVideoPipeline, height, width, tiled, tile_size, tile_stride, motion_video=None):
        pipe.load_models_to_device(["vae"])
        motion_frames = 73
        kwargs = {}
        if motion_video is not None and len(motion_video) > 0:
            assert len(motion_video) == motion_frames, f"motion video must have {motion_frames} frames, but got {len(motion_video)}"
            motion_latents = pipe.preprocess_video(motion_video)  #preprocess in BCTHW
            kwargs["drop_motion_frames"] = False
        else:
            motion_latents = torch.zeros([1, 3, motion_frames, height, width], dtype=pipe.torch_dtype, device=pipe.device)
            kwargs["drop_motion_frames"] = True
        motion_latents = pipe.vae.encode(motion_latents, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        kwargs.update({"motion_latents": motion_latents})
        return kwargs

    def process_pose_cond(self, pipe: WanVideoPipeline, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=None, num_repeats=1, return_all=False):
        if s2v_pose_latents is not None:
            return {"s2v_pose_latents": s2v_pose_latents}
        if s2v_pose_video is None:
            return {"s2v_pose_latents": None}
        pipe.load_models_to_device(["vae"])
        infer_frames = num_frames - 1
        input_video = pipe.preprocess_video(s2v_pose_video)[:, :, :infer_frames * num_repeats]
        # pad if not enough frames
        padding_frames = infer_frames * num_repeats - input_video.shape[2]
        input_video = torch.cat([input_video, -torch.ones(1, 3, padding_frames, height, width, device=input_video.device, dtype=input_video.dtype)], dim=2)
        input_videos = input_video.chunk(num_repeats, dim=2)
        pose_conds = []
        for r in range(num_repeats):
            cond = input_videos[r]
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond], dim=2)
            cond_latents = pipe.vae.encode(cond, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            pose_conds.append(cond_latents[:,:,1:])
        if return_all:
            return pose_conds
        else:
            return {"s2v_pose_latents": pose_conds[0]}

    def process_depth_video(self, pipe: WanVideoPipeline, depth_video, num_frames, height, width, tiled, tile_size, tile_stride):
        """
        处理 Depth Video，将其编码为 latents
        
        Args:
            pipe: WanVideoPipeline 实例
            depth_video: 81帧 depth video 帧列表
            num_frames: 视频帧数 (81)
            height: 视频高度
            width: 视频宽度
            tiled: 是否使用 tiled 编码
            tile_size: tile 大小
            tile_stride: tile 步长
        
        Returns:
            dict: {"depth_latents": depth_latents} 或 {"depth_latents": None}
        """
        # 跳过条件：None、空列表、或者是未处理的字符串路径（use_depth_branch=False 时 CSV 字段未被加载处理）
        if depth_video is None or (isinstance(depth_video, (list, tuple)) and len(depth_video) == 0) or isinstance(depth_video, str):
            return {"depth_latents": None}
        
        pipe.load_models_to_device(["vae"])
        
        # depth_video 应该是 81 帧，与 input_video 相同
        assert len(depth_video) == num_frames, f"depth_video must have {num_frames} frames, but got {len(depth_video)}"
        
        # 预处理视频: [B, C, T, H, W]
        depth_video_tensor = pipe.preprocess_video(depth_video)
        
        # VAE 编码: [B, C, T, H, W] -> [B, 16, T//4+1, H//8, W//8]
        # 81帧 -> 21 latents
        depth_latents = pipe.vae.encode(
            depth_video_tensor, 
            device=pipe.device, 
            tiled=tiled, 
            tile_size=tile_size, 
            tile_stride=tile_stride
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        
        print(f"depth_video encoded: {len(depth_video)} frames -> {depth_latents.shape} latents")
        
        return {"depth_latents": depth_latents}

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if (inputs_shared.get("input_audio") is None and inputs_shared.get("audio_embeds") is None) or pipe.audio_encoder is None or pipe.audio_processor is None:
            return inputs_shared, inputs_posi, inputs_nega
        num_frames, height, width, tiled, tile_size, tile_stride = inputs_shared.get("num_frames"), inputs_shared.get("height"), inputs_shared.get("width"), inputs_shared.get("tiled"), inputs_shared.get("tile_size"), inputs_shared.get("tile_stride")
        input_audio, audio_embeds, audio_sample_rate = inputs_shared.pop("input_audio", None), inputs_shared.pop("audio_embeds", None), inputs_shared.get("audio_sample_rate", 16000)
        s2v_pose_video, s2v_pose_latents, motion_video = inputs_shared.pop("s2v_pose_video", None), inputs_shared.pop("s2v_pose_latents", None), inputs_shared.pop("motion_video", None)
        
        # 提取 depth_video 用于 Depth Video 辅助分支
        depth_video = inputs_shared.pop("depth_video", None)
        
        #get the real start point
        start_idx = inputs_shared.get("start_idx",0)
        audio_input_positive = self.process_audio(pipe, input_audio, audio_sample_rate, num_frames, audio_embeds=audio_embeds,fps=25,start_index=start_idx)
        inputs_posi.update(audio_input_positive)
        inputs_nega.update({"audio_embeds": 0.0 * audio_input_positive["audio_embeds"]})
        
        # 处理 motion_video
        # 注意：在联合编码模式下，motion_video 会被训练脚本清空
        # WanVideoUnit_InputVideoEmbedder may override motion_latents later
        inputs_shared.update(self.process_motion_latents(pipe, height, width, tiled, tile_size, tile_stride, motion_video))
        
        inputs_shared.update(self.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=s2v_pose_latents))
        
        # 处理 depth_video: VAE 编码
        inputs_shared.update(self.process_depth_video(pipe, depth_video, num_frames, height, width, tiled, tile_size, tile_stride))
        
        return inputs_shared, inputs_posi, inputs_nega

    @staticmethod
    def pre_calculate_audio_pose(pipe: WanVideoPipeline, input_audio=None, audio_sample_rate=16000, s2v_pose_video=None, num_frames=81, height=448, width=832, fps=16, tiled=True, tile_size=(30, 52), tile_stride=(15, 26)):
        assert pipe.audio_encoder is not None and pipe.audio_processor is not None, "Please load audio encoder and audio processor first."
        shapes = WanVideoUnit_ShapeChecker().process(pipe, height, width, num_frames)
        height, width, num_frames = shapes["height"], shapes["width"], shapes["num_frames"]
        unit = WanVideoUnit_S2V()
        audio_embeds = unit.process_audio(pipe, input_audio, audio_sample_rate, num_frames, fps, return_all=True)
        pose_latents = unit.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, num_repeats=len(audio_embeds), return_all=True, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        pose_latents = None if s2v_pose_video is None else pose_latents
        return audio_embeds, pose_latents, len(audio_embeds)


class WanVideoPostUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("latents", "motion_latents", "drop_motion_frames"))

    def process(self, pipe: WanVideoPipeline, latents, motion_latents, drop_motion_frames):
        if pipe.audio_encoder is None or motion_latents is None or drop_motion_frames:
            return {}
        #这里拼接起来motion_latent可能不止一帧吧
        latents = torch.cat([motion_latents, latents[:,:,1:]], dim=2)
        return {"latents": latents}


class WanVideoPostUnit_AnimateVideoSplit(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("input_video", "animate_pose_video", "animate_face_video", "animate_inpaint_video", "animate_mask_video"))

    def process(self, pipe: WanVideoPipeline, input_video, animate_pose_video, animate_face_video, animate_inpaint_video, animate_mask_video):
        if input_video is None:
            return {}
        if animate_pose_video is not None:
            animate_pose_video = animate_pose_video[:len(input_video) - 4]
        if animate_face_video is not None:
            animate_face_video = animate_face_video[:len(input_video) - 4]
        if animate_inpaint_video is not None:
            animate_inpaint_video = animate_inpaint_video[:len(input_video) - 4]
        if animate_mask_video is not None:
            animate_mask_video = animate_mask_video[:len(input_video) - 4]
        return {"animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video}


class WanVideoPostUnit_AnimatePoseLatents(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_pose_video", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, animate_pose_video, tiled, tile_size, tile_stride):
        if animate_pose_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        animate_pose_video = pipe.preprocess_video(animate_pose_video)
        pose_latents = pipe.vae.encode(animate_pose_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"pose_latents": pose_latents}


class WanVideoPostUnit_AnimateFacePixelValues(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if inputs_shared.get("animate_face_video", None) is None:
            return inputs_shared, inputs_posi, inputs_nega
        inputs_posi["face_pixel_values"] = pipe.preprocess_video(inputs_shared["animate_face_video"])
        inputs_nega["face_pixel_values"] = torch.zeros_like(inputs_posi["face_pixel_values"]) - 1
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoPostUnit_AnimateInpaint(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_inpaint_video", "animate_mask_video", "input_image", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )
        
    def get_i2v_mask(self, lat_t, lat_h, lat_w, mask_len=1, mask_pixel_values=None, device="cuda"):
        if mask_pixel_values is None:
            msk = torch.zeros(1, (lat_t-1) * 4 + 1, lat_h, lat_w, device=device)
        else:
            msk = mask_pixel_values.clone()
        msk[:, :mask_len] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        return msk

    def process(self, pipe: WanVideoPipeline, animate_inpaint_video, animate_mask_video, input_image, tiled, tile_size, tile_stride):
        if animate_inpaint_video is None or animate_mask_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)

        bg_pixel_values = pipe.preprocess_video(animate_inpaint_video)
        y_reft = pipe.vae.encode(bg_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0].to(dtype=pipe.torch_dtype, device=pipe.device)
        _, lat_t, lat_h, lat_w = y_reft.shape
        
        ref_pixel_values = pipe.preprocess_video([input_image])
        ref_latents = pipe.vae.encode(ref_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        mask_ref = self.get_i2v_mask(1, lat_h, lat_w, 1, device=pipe.device)
        y_ref = torch.concat([mask_ref, ref_latents[0]]).to(dtype=torch.bfloat16, device=pipe.device)
        
        mask_pixel_values = 1 - pipe.preprocess_video(animate_mask_video, max_value=1, min_value=0)
        mask_pixel_values = rearrange(mask_pixel_values, "b c t h w -> (b t) c h w")
        mask_pixel_values = torch.nn.functional.interpolate(mask_pixel_values, size=(lat_h, lat_w), mode='nearest')
        mask_pixel_values = rearrange(mask_pixel_values, "(b t) c h w -> b t c h w", b=1)[:,:,0]
        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, 0, mask_pixel_values=mask_pixel_values, device=pipe.device)
        
        y_reft = torch.concat([msk_reft, y_reft]).to(dtype=torch.bfloat16, device=pipe.device)
        y = torch.concat([y_ref, y_reft], dim=1).unsqueeze(0)
        return {"y": y}


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value



def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    animate_adapter: WanAnimateAdapter = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    product_image_latents: Optional[torch.Tensor] = None,
    drop_motion_frames: bool = True,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    return_router_logits: bool = False,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    # wan2.2 s2v
    if audio_embeds is not None:
        # 推理时：cfg_merge=True 时 batch[0] 是 conditional，batch[1] 是 unconditional
        # 这里需要根据调用方传入的 is_conditional 参数来决定
        # 默认情况下，如果没有传入 is_conditional，则认为是 conditional
        is_conditional_flag = kwargs.get('is_conditional', True)
        # MoE 1.3: 训练时传递 face_mask 和 hand_mask 用于硬选择专家
        face_mask = kwargs.get('face_mask', None)
        hand_mask = kwargs.get('hand_mask', None)
        # 提取 depth 相关参数
        depth_latents = kwargs.get('depth_latents', None)
        depth_timestep = kwargs.get('depth_timestep', None)
        depth_context = kwargs.get('depth_context', None)
        use_deepspeed_activation_checkpointing = kwargs.get('use_deepspeed_activation_checkpointing', False)
        
        # 提取 depth_mutual_visible 参数
        depth_mutual_visible = kwargs.get('depth_mutual_visible', True)
        
        # Extract product_image_scale (product image RoPE spatial scale factor)
        product_image_scale = kwargs.get('product_image_scale', 1.0)
        
        return model_fn_wans2v(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            audio_embeds=audio_embeds,
            motion_latents=motion_latents,
            s2v_pose_latents=s2v_pose_latents,
            product_image_latents=product_image_latents,
            drop_motion_frames=drop_motion_frames,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            return_router_logits=return_router_logits,
            is_conditional=is_conditional_flag,
            face_mask=face_mask,
            hand_mask=hand_mask,
            depth_latents=depth_latents,
            depth_timestep=depth_timestep,
            depth_context=depth_context,
            depth_mutual_visible=depth_mutual_visible,
            use_deepspeed_activation_checkpointing=use_deepspeed_activation_checkpointing,
            product_image_scale=product_image_scale,
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    # Timestep
    # TI2V-5B: seperated_timestep=True + fuse_vae_embedding_in_latents=True
    # 第一帧 tokens 的 timestep=0（已是干净的输入图片），其余帧使用实际 timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        # T2V / 其他模型: 标量 timestep（原始 Wan2.2 逻辑）
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    # Camera control
    x = dit.patchify(x, control_camera_latents_input)
    
    # Animate
    if pose_latents is not None and face_pixel_values is not None:
        x, motion_vec = animate_adapter.after_patch_embedding(x, pose_latents, face_pixel_values)
    
    # Patchify
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    
    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(
            x, vace_context, context, t_mod, freqs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload
        )
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            # Block
            # 
            # x, context, t_mod, freqs, audio_embedding, grid_sizes,
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs)
            
            # VACE
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0)
                x = x + current_vace_hint * vace_scale
            
            # Animate
            if pose_latents is not None and face_pixel_values is not None:
                x = animate_adapter.after_transformer_block(block_id, x, motion_vec)
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    return x


def model_fn_wans2v(
    dit,
    latents,
    timestep,
    context,
    audio_embeds,
    motion_latents,
    s2v_pose_latents,
    product_image_latents=None,
    drop_motion_frames=True,
    use_gradient_checkpointing_offload=False,
    use_gradient_checkpointing=False,
    use_unified_sequence_parallel=False,
    return_router_logits=False,
    is_conditional=True,
    face_mask=None,
    hand_mask=None,
    # Depth Video 分支参数
    depth_latents=None,
    depth_timestep=None,
    depth_context=None,  # 预编码的 "depth video" prompt context
    depth_mutual_visible=True,  # Depth 可见性模式: True=相互可见, False=只有depth能看到video
    # 嵌套 checkpoint 策略参数
    use_deepspeed_activation_checkpointing=False,
    # Product image RoPE spatial scale factor
    product_image_scale=1.0,
    **kwargs,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    
    # 检查是否启用 depth 分支
    has_depth_latents = depth_latents is not None
    has_depth_method = hasattr(dit, 'is_depth_branch_enabled')
    depth_method_result = dit.is_depth_branch_enabled() if has_depth_method else False
    enable_depth = has_depth_latents and has_depth_method and depth_method_result
    
    # 调试信息
    if has_depth_latents:
        pass  # Depth debug info available via depth_latents.shape if needed
    
    # 分离参考帧和目标帧
    # latents[:, :, 0:1] 是干净的参考帧（训练时保持干净，不加噪）
    # latents[:, :, 1:] 是加噪后的目标帧
    origin_ref_latents = latents[:, :, 0:1]  # [B, C, 1, H, W] 干净的参考帧
    x = latents[:, :, 1:]                     # [B, C, T-1, H, W] 加噪的目标帧

    # context embedding
    context = dit.text_embedding(context)
    
    # Depth Video: 准备 depth context (由文本编码器动态编码 depth prompt 得到)
    # depth prompt 格式: "a video transformed into a depth map. ... Following is the RGB description:{video_prompt}"
    depth_context_final = None
    if enable_depth:
        if depth_context is not None:
            # 使用传入的动态编码 depth_context
            # 需要通过 text_embedding 投影到 DiT 的隐藏维度 (4096 -> 5120)
            depth_context_final = dit.text_embedding(depth_context)
            # 确保 batch size 匹配
            if depth_context_final.shape[0] != context.shape[0]:
                depth_context_final = depth_context_final.expand(context.shape[0], -1, -1)
        else:
            # 如果没有传入 depth_context，使用主分支的 context 作为 fallback
            # 注意：正常推理时 WanVideoUnit_PromptEmbedder 会自动编码 depth prompt
            # 这里是 fallback，可能在某些特殊调用场景下触发
            depth_context_final = context.clone()

    # audio encode
    audio_emb_global, merged_audio_emb = dit.cal_audio_emb(audio_embeds)
    # x and s2v_pose_latents
    s2v_pose_latents = torch.zeros_like(x) if s2v_pose_latents is None else s2v_pose_latents
    x, (f, h, w) = dit.patchify(dit.patch_embedding(x) + dit.cond_encoder(s2v_pose_latents))
    seq_len_x = seq_len_x_global = x.shape[1] # global used for unified sequence parallel
    #seq_len_x记录的是不包含ref_latent外的patch后序列长度
    
    # ========== Depth Video Patchify ==========
    depth_tokens = None
    depth_grid_size = None
    seq_len_depth = 0
    if enable_depth:
        # depth_latents shape: [B, C, T, H, W] (已经是后20帧)
        # 使用 depth 专用的 patchify
        depth_tokens, depth_grid_size = dit.depth_patchify(depth_latents)
        seq_len_depth = depth_tokens.shape[1]
        # 添加 depth 类型标识 (类型 = 3)
        depth_type_ids = torch.zeros([1, seq_len_depth], dtype=torch.long, device=depth_tokens.device)
        depth_tokens = depth_tokens + dit.depth_cond_mask(depth_type_ids).to(depth_tokens.dtype)
    
    # reference image - clean reference frame
    ref_latents, (rf, rh, rw) = dit.patchify(dit.patch_embedding(origin_ref_latents))
    seq_len_ref = ref_latents.shape[1]  # person ref tokens length
    
    # Process product image latents (if provided)
    seq_len_extra_ref = 0  # product image tokens length
    if product_image_latents is not None:
        product_latents_patched, (erf, erh, erw) = dit.patchify(dit.patch_embedding(product_image_latents))
        seq_len_extra_ref = product_latents_patched.shape[1]
        grid_sizes = dit.get_grid_sizes((f, h, w), (rf, rh, rw), (erf, erh, erw), extra_ref_scale=product_image_scale)
        # Concat order: target frames + person ref + product image
        x = torch.cat([x, ref_latents, product_latents_patched], dim=1)
        # mask: 0=target, 1=person ref, 1=product image
        mask = torch.cat([
            torch.zeros([1, seq_len_x]),
            torch.ones([1, ref_latents.shape[1]]),
            torch.ones([1, product_latents_patched.shape[1]])
        ], dim=1).to(torch.long).to(x.device)
    else:
        grid_sizes = dit.get_grid_sizes((f, h, w), (rf, rh, rw))
        x = torch.cat([x, ref_latents], dim=1)
        # mask
        mask = torch.cat([torch.zeros([1, seq_len_x]), torch.ones([1, ref_latents.shape[1]])], dim=1).to(torch.long).to(x.device)
    # freqs
    pre_compute_freqs = rope_precompute(x.detach().view(1, x.size(1), dit.num_heads, dit.dim // dit.num_heads), grid_sizes, dit.freqs, start=None)
    # motion
    x, pre_compute_freqs, mask = dit.inject_motion(x, pre_compute_freqs, mask, motion_latents, drop_motion_frames=drop_motion_frames, add_last_motion=2)
    # trainable_cond_mask: nn.Embedding(3, dim) linear projection
    x = x + dit.trainable_cond_mask(mask).to(x.dtype)
    
    # 记录主序列长度（用于 attention mask）
    seq_len_main = x.shape[1]
    
    # ========== Depth Video: 拼接到主序列并计算位置编码 ==========
    depth_freqs = None
    # 记录 seq_len_target (input_video tokens 的长度)，用于分块 attention
    seq_len_target = seq_len_x  # input_video tokens 的长度
    if enable_depth and depth_tokens is not None:
        # 计算 depth 的负数位置编码
        depth_grid_sizes = dit.get_depth_grid_sizes(depth_grid_size)
        depth_freqs = rope_precompute(
            depth_tokens.detach().view(1, depth_tokens.shape[1], dit.num_heads, dit.dim // dit.num_heads),
            depth_grid_sizes,
            dit.freqs,
            start=None
        )
        
        # 拼接 depth tokens 到主序列末尾
        x = torch.cat([x, depth_tokens], dim=1)
        
        # 拼接位置编码 (用于 gradient checkpointing 模式)
        pre_compute_freqs = torch.cat([pre_compute_freqs, depth_freqs], dim=1)
        
        # 注意: 不再创建 attention_mask
        # 改用分块计算方案: 主序列 Self-Attention + Depth 混合 Attention
        # 这样可以完全使用 Flash Attention，避免性能损失

    # tmod
    # 保存 timestep 值用于 MoE 路由控制
    # timestep 是原始值 (0-1000 范围)，直接使用 900 作为阈值
    # 注意：timestep 传入模型时不需要归一化，sinusoidal_embedding_1d 直接使用原始值
    raw_timestep = timestep[0].item() if timestep.numel() > 0 else 0.0
    
    # MoE 路由启用条件: is_conditional=True 且 timestep < 900
    # 阈值 900 对应归一化值 0.9 (900/1000)
    moe_routing_enabled = is_conditional and (raw_timestep < 900)
    if hasattr(dit, 'is_moe_enabled') and dit.is_moe_enabled():
        pass  # print(f"[MoE Debug] t={raw_timestep:.0f}, is_conditional={is_conditional}, moe_routing_enabled={moe_routing_enabled}")
    
    # 传递给 block 的 normalized_timestep 仍然需要归一化到 0-1，因为 MoEFFN 内部使用 0.9 作为阈值
    normalized_timestep = raw_timestep / 1000.0
    
    timestep = torch.cat([timestep, torch.zeros([1], dtype=timestep.dtype, device=timestep.device)])
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim)).unsqueeze(2).transpose(0, 2)
    
    # ========== Depth Video: 计算 depth 的 time embedding ==========
    t_depth = None
    t_mod_depth = None
    if enable_depth and depth_timestep is not None:
        depth_timestep_cat = torch.cat([depth_timestep, torch.zeros([1], dtype=depth_timestep.dtype, device=depth_timestep.device)])
        t_depth = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, depth_timestep_cat))
        t_mod_depth = dit.time_projection(t_depth).unflatten(1, (6, dit.dim)).unsqueeze(2).transpose(0, 2)
    
    # Router logits 收集器（训练时使用）
    router_logits_cache = [] if return_router_logits else None
    
    # 如果需要收集 router_logits 且使用 gradient checkpointing，启用缓存模式
    use_router_cache = return_router_logits and (use_gradient_checkpointing_offload or use_gradient_checkpointing or use_deepspeed_activation_checkpointing)
    if use_router_cache:
        for block in dit.blocks:
            if hasattr(block, 'ffn') and hasattr(block.ffn, 'enable_router_logits_cache'):
                block.ffn.enable_router_logits_cache(True)
    
    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        world_size, sp_rank = get_sequence_parallel_world_size(), get_sequence_parallel_rank()
        assert x.shape[1] % world_size == 0, f"the dimension after chunk must be divisible by world size, but got {x.shape[1]} and {get_sequence_parallel_world_size()}"
        x = torch.chunk(x, world_size, dim=1)[sp_rank]
        seg_idxs = [0] + list(torch.cumsum(torch.tensor([x.shape[1]] * world_size), dim=0).cpu().numpy())
        seq_len_x_list = [min(max(0, seq_len_x - seg_idxs[i]), x.shape[1]) for i in range(len(seg_idxs)-1)]
        seq_len_x = seq_len_x_list[sp_rank]

    def create_custom_forward(module, seq_len_depth=0, seq_len_target=0, depth_freqs=None, depth_mutual_visible=True,
                               timestep=None, is_conditional=True, seq_len_ref=0, seq_len_extra_ref=0,
                               depth_context=None):
        """
        创建 block 的包装函数，支持 depth 分支参数和 MoE 参数
        
        注意：depth_freqs 和 depth_context 作为显式参数传递给 custom_forward，避免闭包捕获 tensor 导致显存爆炸
              timestep 和 is_conditional 是标量，通过闭包捕获不会有显存问题
        """
        def custom_forward(x, ctx, tm, seq_len, freq, _depth_freqs, _depth_ctx):
            # depth_freqs 和 depth_context 作为显式参数传入，而非闭包捕获
            # timestep 和 is_conditional 通过闭包捕获（标量，不会导致显存问题）
            return module(
                x, ctx, tm, seq_len, freq,
                seq_len_depth=seq_len_depth,
                seq_len_target=seq_len_target,
                depth_freqs=_depth_freqs,
                depth_mutual_visible=depth_mutual_visible,
                timestep=timestep,
                is_conditional=is_conditional,
                seq_len_ref=seq_len_ref,
                seq_len_extra_ref=seq_len_extra_ref,
                depth_context=_depth_ctx,
            )
        return custom_forward, depth_freqs, depth_context  # 返回函数、depth_freqs 和 depth_context，调用时需要传递
    
    def create_after_block_forward(dit, block_id, seq_len):
        """创建 after_transformer_block 的包装函数，将所有需要梯度的 tensor 作为参数"""
        def forward_fn(x, audio_emb_g, merged_audio):
            return dit.after_transformer_block(block_id, x, audio_emb_g, merged_audio, seq_len)
        return forward_fn
    
    # 嵌套 checkpoint 策略：
    # - 外层：每 N 个 block 保存边界激活值到 CPU（减少 CPU 内存压力）
    # - 内层：每个 block 单独 checkpoint（不 offload，GPU 峰值低）
    CHECKPOINT_EVERY_N_BLOCKS = 2
    num_blocks = len(dit.blocks)
    
    def create_nested_checkpoint_forward(blocks, block_ids, dit, seq_len, audio_g, audio_m, 
                                          seq_len_depth=0, seq_len_target=0, depth_freqs=None, depth_mutual_visible=True,
                                          timestep=None, is_conditional=True, seq_len_ref=0, seq_len_extra_ref=0,
                                          depth_context=None):
        """
        创建嵌套 checkpoint 前向函数，支持 depth 分支参数和 MoE 参数
        
        注意：depth_freqs 和 depth_context 作为显式参数传递给 forward_fn，避免闭包捕获 tensor 导致显存爆炸
              timestep 和 is_conditional 是标量，通过闭包捕获不会有显存问题
        """
        def forward_fn(x, ctx, tm, freq, _depth_freqs, _depth_ctx):
            # depth_freqs 和 depth_context 作为显式参数传入，而非闭包捕获
            for blk, bid in zip(blocks, block_ids):
                # 内层 checkpoint：不 offload，只做重计算
                # 注意：只捕获标量参数，tensor 参数通过 checkpoint 的显式参数传递
                #       timestep 和 is_conditional 通过闭包捕获（标量，不会导致显存问题）
                def blk_forward(_x, _ctx, _tm, _freq, _df, _dc, _blk=blk, _seq=seq_len, 
                               _seq_depth=seq_len_depth, _seq_target=seq_len_target, 
                               _depth_mutual_visible=depth_mutual_visible,
                               _timestep=timestep, _is_conditional=is_conditional,
                               _seq_ref=seq_len_ref, _seq_extra_ref=seq_len_extra_ref):
                    return _blk(
                        _x, _ctx, _tm, _seq, _freq,
                        seq_len_depth=_seq_depth,
                        seq_len_target=_seq_target,
                        depth_freqs=_df,
                        depth_mutual_visible=_depth_mutual_visible,
                        timestep=_timestep,
                        is_conditional=_is_conditional,
                        seq_len_ref=_seq_ref,
                        seq_len_extra_ref=_seq_extra_ref,
                        depth_context=_dc,
                    )
                
                def after_forward(_x, _ag, _am, _dit=dit, _bid=bid, _seq=seq_len):
                    # depth 序列跳过 audio injection：after_transformer_block 只处理前 _seq 个 tokens
                    # depth tokens（如果存在）在 clone 后原样返回，不参与 audio 注入
                    return _dit.after_transformer_block(_bid, _x, _ag, _am, _seq)
                
                x = torch.utils.checkpoint.checkpoint(
                    blk_forward, x, ctx, tm, freq, _depth_freqs, _depth_ctx,
                    use_reentrant=False,
                )
                x = torch.utils.checkpoint.checkpoint(
                    after_forward, x, audio_g, audio_m,
                    use_reentrant=False,
                )
            return x
        return forward_fn, depth_freqs, depth_context  # 返回 forward_fn、depth_freqs 和 depth_context，调用时需要传递
    
    # ========== Debug: 确认 activation checkpoint 模式 ==========
    # print(f"[Checkpoint Debug] use_deepspeed_activation_checkpointing={use_deepspeed_activation_checkpointing}, "
    #       f"use_gradient_checkpointing_offload={use_gradient_checkpointing_offload}, "
    #       f"use_gradient_checkpointing={use_gradient_checkpointing}, "
    #       f"CHECKPOINT_EVERY_N_BLOCKS={CHECKPOINT_EVERY_N_BLOCKS}, "
    #       f"num_blocks={num_blocks}, "
    #       f"seq_len_x={seq_len_x}, seq_len_depth={seq_len_depth}, total_seq={x.shape[1]}")
    
    # ========== Audio Face Mask: 在 block loop 前设置 GT mask ==========
    # 如果 use_audio_face_mask 开启且训练时使用 GT source，将 face_mask 存到 dit 上
    # after_transformer_block 内部会自动读取
    # 注意：必须始终调用 set（即使 face_mask 为 None），确保不使用上一步的 stale mask
    if hasattr(dit, 'use_audio_face_mask') and dit.use_audio_face_mask:
        if dit.audio_mask_train_source == "gt":
            dit.set_audio_gt_mask(face_mask)  # face_mask 可以是 None
    
    block_idx = 0
    while block_idx < num_blocks:
        end_idx = min(block_idx + CHECKPOINT_EVERY_N_BLOCKS, num_blocks)
        current_blocks = [dit.blocks[i] for i in range(block_idx, end_idx)]
        current_block_ids = list(range(block_idx, end_idx))
        
        if use_deepspeed_activation_checkpointing:
            # 嵌套 checkpoint 策略：外层 checkpoint + save_on_cpu，内层 checkpoint 不 offload
            # 适用于 depth 分支等大上下文场景，减少 CPU 内存压力同时降低 GPU 峰值显存
            # 注意：depth_freqs 和 depth_context 作为显式参数传递，避免闭包捕获 tensor 导致显存爆炸
            #       timestep 和 is_conditional 是标量，通过闭包捕获不会有显存问题
            forward_fn, _depth_freqs, _depth_ctx = create_nested_checkpoint_forward(
                current_blocks, current_block_ids, dit, seq_len_x,
                audio_emb_global, merged_audio_emb,
                seq_len_depth=seq_len_depth,
                seq_len_target=seq_len_target,
                depth_freqs=depth_freqs,
                depth_mutual_visible=depth_mutual_visible,
                timestep=normalized_timestep,
                is_conditional=is_conditional,
                seq_len_ref=seq_len_ref,
                seq_len_extra_ref=seq_len_extra_ref,
                depth_context=depth_context_final,
            )
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    forward_fn,
                    x, context, t_mod, pre_compute_freqs[0], _depth_freqs, _depth_ctx,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing_offload:
            # 原有逻辑：每个 block 单独 checkpoint + save_on_cpu
            # 注意：depth_freqs 和 depth_context 作为显式参数传递，避免闭包捕获 tensor 导致显存爆炸
            #       timestep 和 is_conditional 是标量，通过闭包捕获不会有显存问题
            for blk, bid in zip(current_blocks, current_block_ids):
                custom_fn, _depth_freqs, _depth_ctx = create_custom_forward(
                    blk, seq_len_depth=seq_len_depth, seq_len_target=seq_len_target, 
                    depth_freqs=depth_freqs, depth_mutual_visible=depth_mutual_visible,
                    timestep=normalized_timestep, is_conditional=is_conditional,
                    seq_len_ref=seq_len_ref, seq_len_extra_ref=seq_len_extra_ref,
                    depth_context=depth_context_final,
                )
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        custom_fn,
                        x, context, t_mod, seq_len_x, pre_compute_freqs[0], _depth_freqs, _depth_ctx,
                        use_reentrant=False,
                    )
                    x = torch.utils.checkpoint.checkpoint(
                        create_after_block_forward(dit, bid, seq_len_x),
                        x, audio_emb_global, merged_audio_emb,
                        use_reentrant=False,
                    )
        elif use_gradient_checkpointing:
            # 每个 block 单独 checkpoint
            # 注意：depth_freqs 和 depth_context 作为显式参数传递，避免闭包捕获 tensor 导致显存爆炸
            #       timestep 和 is_conditional 是标量，通过闭包捕获不会有显存问题
            for blk, bid in zip(current_blocks, current_block_ids):
                custom_fn, _depth_freqs, _depth_ctx = create_custom_forward(
                    blk, seq_len_depth=seq_len_depth, seq_len_target=seq_len_target,
                    depth_freqs=depth_freqs, depth_mutual_visible=depth_mutual_visible,
                    timestep=normalized_timestep, is_conditional=is_conditional,
                    seq_len_ref=seq_len_ref, seq_len_extra_ref=seq_len_extra_ref,
                    depth_context=depth_context_final,
                )
                x = torch.utils.checkpoint.checkpoint(
                    custom_fn,
                    x, context, t_mod, seq_len_x, pre_compute_freqs[0], _depth_freqs, _depth_ctx,
                    use_reentrant=False,
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_after_block_forward(dit, bid, seq_len_x),
                    x, audio_emb_global, merged_audio_emb,
                    use_reentrant=False,
                )
        else:
            # 不使用 checkpoint，直接前向传播
            for blk, bid in zip(current_blocks, current_block_ids):
                if return_router_logits:
                    # MoE 1.3: 训练时传递 face_mask 和 hand_mask 用于硬选择专家
                    # depth_mutual_visible: True=相互可见(版本1), False=只有depth能看到video(版本2)
                    x_out = blk(
                        x, context, t_mod, seq_len_x, pre_compute_freqs[0], 
                        return_router_logits=True,
                        timestep=normalized_timestep,
                        is_conditional=is_conditional,
                        face_mask=face_mask,
                        hand_mask=hand_mask,
                        skip_moe_for_depth=enable_depth,
                        seq_len_depth=seq_len_depth,
                        seq_len_target=seq_len_target,
                        depth_freqs=depth_freqs,
                        depth_mutual_visible=depth_mutual_visible,
                        seq_len_ref=seq_len_ref,
                        seq_len_extra_ref=seq_len_extra_ref,
                        depth_context=depth_context_final,
                    )
                    if isinstance(x_out, tuple):
                        x, router_logits = x_out
                        if router_logits is not None:
                            router_logits_cache.append(router_logits)
                    else:
                        x = x_out
                else:
                    # depth_mutual_visible: True=相互可见(版本1), False=只有depth能看到video(版本2)
                    x = blk(
                        x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                        timestep=normalized_timestep,
                        is_conditional=is_conditional,
                        face_mask=face_mask,
                        hand_mask=hand_mask,
                        skip_moe_for_depth=enable_depth,
                        seq_len_depth=seq_len_depth,
                        seq_len_target=seq_len_target,
                        depth_freqs=depth_freqs,
                        depth_mutual_visible=depth_mutual_visible,
                        seq_len_ref=seq_len_ref,
                        seq_len_extra_ref=seq_len_extra_ref,
                        depth_context=depth_context_final,
                    )
                # Audio injection: 只对 RGB 主序列注入 audio，depth 序列跳过
                # after_transformer_block 内部只处理前 seq_len_x_global 个 tokens
                # depth tokens（如果存在，在序列末尾）会被 clone 后原样返回
                x = dit.after_transformer_block(bid, x, audio_emb_global, merged_audio_emb, seq_len_x_global, use_unified_sequence_parallel)
        
        block_idx = end_idx

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        x = get_sp_group().all_gather(x, dim=1)
    
    # ========== Audio Face Mask: 不在此处 clear ==========
    # _audio_gt_mask 必须保留到 backward 完成（checkpoint recomputation 需要）
    # 下一步的 set_audio_gt_mask 会自动覆盖
    
    # 如果使用了 router_logits 缓存模式，收集缓存的 logits
    if use_router_cache:
        for block in dit.blocks:
            if hasattr(block, 'ffn') and hasattr(block.ffn, 'get_cached_router_logits'):
                cached_logits = block.ffn.get_cached_router_logits()
                if cached_logits is not None:
                    router_logits_cache.append(cached_logits)
                # 清理缓存
                block.ffn.clear_router_logits_cache()
                block.ffn.enable_router_logits_cache(False)

    # ========== 分离主序列和 depth 序列，分别进行 unpatchify ==========
    depth_output = None
    if enable_depth and seq_len_depth > 0:
        # 分离主序列和 depth 序列
        x_main = x[:, :seq_len_main]
        x_depth = x[:, seq_len_main:]
        
        # 主序列: 只取目标帧部分
        x_main = x_main[:, :seq_len_x_global]
        x_main = dit.head(x_main, t[:-1])
        x_main = dit.unpatchify(x_main, (f, h, w))
        
        # Depth 序列: 使用 depth_unpatchify 进行处理
        # depth_unpatchify 内部会调用 depth_head，所以直接传入 x_depth
        depth_t = t_depth[:-1] if t_depth is not None else t[:-1]
        depth_output = dit.depth_unpatchify(x_depth, depth_grid_size, depth_t)
        
        x = x_main
    else:
        x = x[:, :seq_len_x_global]
        x = dit.head(x, t[:-1])
        x = dit.unpatchify(x, (f, h, w))
    
    # 返回时拼接：参考帧位置填充零（因为参考帧的flow目标应该是0，不参与loss）
    # 对于干净的参考帧：flow = noise - sample，但参考帧不加噪，所以这里填0
    # ref_flow_placeholder = torch.zeros_like(origin_ref_latents)
    x = torch.cat([origin_ref_latents, x], dim=2)
    #print("final_x",x.shape)
    
    # 返回结果
    if enable_depth and depth_output is not None:
        if return_router_logits:
            return x, depth_output, router_logits_cache
        return x, depth_output
    else:
        if return_router_logits:
            return x, router_logits_cache
        return x
