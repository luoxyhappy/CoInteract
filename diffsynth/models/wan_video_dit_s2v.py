import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .utils import hash_state_dict_keys
from .wan_video_dit import rearrange, precompute_freqs_cis_3d, DiTBlock, Head, CrossAttention, modulate, sinusoidal_embedding_1d, flash_attention, rope_apply, RMSNorm
from .moe_ffn import MoEFFN


def create_depth_attention_mask(
    seq_len_main: int,
    seq_len_depth: int,
    seq_len_target: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    创建 Depth Video 的 Self-Attention Mask
    
    Attention 可见性规则:
    - 主序列 (target + ref + extra_ref + motion) 内部互相可见
    - 主序列看不到 depth
    - depth 能看到自己和 target (input video)
    - depth 看不到 ref, extra_ref, motion
    
    Args:
        seq_len_main: 主序列长度 (target + ref + extra_ref + motion)
        seq_len_depth: depth 序列长度
        seq_len_target: target 序列长度 (用于确定 depth 能看到的范围)
        device: 设备
        dtype: 数据类型
    
    Returns:
        mask: [1, 1, total_seq_len, total_seq_len]
              True/1.0 = 可见, False/0.0 = 不可见
              用于 attention_mask，不可见位置会被设为 -inf
    """
    total_len = seq_len_main + seq_len_depth
    
    # 初始化 mask 为全 False (不可见)
    mask = torch.zeros(1, 1, total_len, total_len, device=device, dtype=dtype)
    
    # 1. 主序列内部互相可见 (左上角 seq_len_main x seq_len_main)
    mask[:, :, :seq_len_main, :seq_len_main] = 1.0
    
    # 2. depth 能看到自己 (右下角 seq_len_depth x seq_len_depth)
    mask[:, :, seq_len_main:, seq_len_main:] = 1.0
    
    # 3. depth 能看到 target (右下角的左边部分，只看 target，不看 ref/motion)
    # depth tokens 在 [seq_len_main:] 位置
    # target tokens 在 [0:seq_len_target] 位置
    mask[:, :, seq_len_main:, :seq_len_target] = 1.0
    
    # 4. 主序列看不到 depth (已经是 0，不需要额外设置)
    
    return mask


def apply_attention_mask_to_scores(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    将 attention mask 应用到 attention scores
    
    Args:
        scores: attention scores, shape [B, H, S, S]
        mask: attention mask, shape [1, 1, S, S], 1.0=可见, 0.0=不可见
    
    Returns:
        masked_scores: 不可见位置被设为 -inf
    """
    # 将 mask 中的 0 转换为 -inf
    mask_value = torch.finfo(scores.dtype).min
    masked_scores = scores + (1.0 - mask) * mask_value
    return masked_scores


def torch_dfs(model: nn.Module, parent_name='root'):
    module_names, modules = [], []
    current_name = parent_name if parent_name else 'root'
    module_names.append(current_name)
    modules.append(model)

    for name, child in model.named_children():
        if parent_name:
            child_name = f'{parent_name}.{name}'
        else:
            child_name = name
        child_modules, child_names = torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names


def rope_precompute(x, grid_sizes, freqs, start=None):
    b, s, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2

    # split freqs
    if type(freqs) is list:
        trainable_freqs = freqs[1]
        freqs = freqs[0]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = torch.view_as_complex(x.detach().reshape(b, s, n, -1, 2).to(torch.float64))
    seq_bucket = [0]
    if not type(grid_sizes) is list:
        grid_sizes = [grid_sizes]
    for g in grid_sizes:
        if not type(g) is list:
            g = [torch.zeros_like(g), g]
        batch_size = g[0].shape[0]
        for i in range(batch_size):
            if start is None:
                f_o, h_o, w_o = g[0][i]
            else:
                f_o, h_o, w_o = start[i]

            f, h, w = g[1][i]
            t_f, t_h, t_w = g[2][i]
            seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
            seq_len = int(seq_f * seq_h * seq_w)
            if seq_len > 0:
                if t_f > 0:
                    factor_f, factor_h, factor_w = (t_f / seq_f).item(), (t_h / seq_h).item(), (t_w / seq_w).item()
                    # Generate a list of seq_f integers starting from f_o and ending at math.ceil(factor_f * seq_f.item() + f_o.item())
                    if f_o >= 0:
                        f_sam = np.linspace(f_o.item(), (t_f + f_o).item() - 1, seq_f).astype(int).tolist()
                    else:
                        f_sam = np.linspace(-f_o.item(), (-t_f - f_o).item() + 1, seq_f).astype(int).tolist()
                    h_sam = np.linspace(h_o.item(), (t_h + h_o).item() - 1, seq_h).astype(int).tolist()
                    w_sam = np.linspace(w_o.item(), (t_w + w_o).item() - 1, seq_w).astype(int).tolist()

                    assert f_o * f >= 0 and h_o * h >= 0 and w_o * w >= 0
                    freqs_0 = freqs[0][f_sam] if f_o >= 0 else freqs[0][f_sam].conj()
                    freqs_0 = freqs_0.view(seq_f, 1, 1, -1)

                    freqs_i = torch.cat(
                        [
                            freqs_0.expand(seq_f, seq_h, seq_w, -1),
                            freqs[1][h_sam].view(1, seq_h, 1, -1).expand(seq_f, seq_h, seq_w, -1),
                            freqs[2][w_sam].view(1, 1, seq_w, -1).expand(seq_f, seq_h, seq_w, -1),
                        ],
                        dim=-1
                    ).reshape(seq_len, 1, -1)
                elif t_f < 0:
                    freqs_i = trainable_freqs.unsqueeze(1)
                # apply rotary embedding
                output[i, seq_bucket[-1]:seq_bucket[-1] + seq_len] = freqs_i
        seq_bucket.append(seq_bucket[-1] + seq_len)
    return output


class CausalConv1d(nn.Module):

    def __init__(self, chan_in, chan_out, kernel_size=3, stride=1, dilation=1, pad_mode='replicate', **kwargs):
        super().__init__()

        self.pad_mode = pad_mode
        padding = (kernel_size - 1, 0)  # T
        self.time_causal_padding = padding

        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class MotionEncoder_tc(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, num_heads=int, need_global=True, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.num_heads = num_heads
        self.need_global = need_global
        self.conv1_local = CausalConv1d(in_dim, hidden_dim // 4 * num_heads, 3, stride=1)
        if need_global:
            self.conv1_global = CausalConv1d(in_dim, hidden_dim // 4, 3, stride=1)
        self.norm1 = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.act = nn.SiLU()
        self.conv2 = CausalConv1d(hidden_dim // 4, hidden_dim // 2, 3, stride=2)
        self.conv3 = CausalConv1d(hidden_dim // 2, hidden_dim, 3, stride=2)

        if need_global:
            self.final_linear = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs)

        self.norm1 = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.norm2 = nn.LayerNorm(hidden_dim // 2, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x):
        #print("x.shape",x.shape)
        #b means batch_size, t means audio_frames, c means dim, 这里各层已经加权
        x = rearrange(x, 'b t c -> b c t')
        x_ori = x.clone()
        b, c, t = x.shape
        x = self.conv1_local(x)
        x = rearrange(x, 'b (n c) t -> (b n) t c', n=self.num_heads)
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv2(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv3(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm3(x)
        x = self.act(x)
        x = rearrange(x, '(b n) t c -> b t n c', b=b)
        padding = self.padding_tokens.repeat(b, x.shape[1], 1, 1).to(device=x.device, dtype=x.dtype)
        x = torch.cat([x, padding], dim=-2)
        x_local = x.clone()

        if not self.need_global:
            return x_local

        x = self.conv1_global(x_ori)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv2(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv3(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm3(x)
        x = self.act(x)
        x = self.final_linear(x)
        x = rearrange(x, '(b n) t c -> b t n c', b=b)

        return x, x_local


class FramePackMotioner(nn.Module):

    def __init__(self, inner_dim=1024, num_heads=16, zip_frame_buckets=[1, 2, 16], drop_mode="drop", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
        self.zip_frame_buckets = torch.tensor(zip_frame_buckets, dtype=torch.long)

        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.freqs = torch.cat(precompute_freqs_cis_3d(inner_dim // num_heads), dim=1)
        self.drop_mode = drop_mode

    def forward(self, motion_latents, add_last_motion=2):
        motion_frames = motion_latents[0].shape[1]
        mot = []
        mot_remb = []
        for m in motion_latents:
            lat_height, lat_width = m.shape[2], m.shape[3]
            padd_lat = torch.zeros(16, self.zip_frame_buckets.sum(), lat_height, lat_width).to(device=m.device, dtype=m.dtype)
            overlap_frame = min(padd_lat.shape[1], m.shape[1])
            if overlap_frame > 0:
                padd_lat[:, -overlap_frame:] = m[:, -overlap_frame:]

            if add_last_motion < 2 and self.drop_mode != "drop":
                zero_end_frame = self.zip_frame_buckets[:self.zip_frame_buckets.__len__() - add_last_motion - 1].sum()
                padd_lat[:, -zero_end_frame:] = 0

            padd_lat = padd_lat.unsqueeze(0)
            clean_latents_4x, clean_latents_2x, clean_latents_post = padd_lat[:, :, -self.zip_frame_buckets.sum():, :, :].split(
                list(self.zip_frame_buckets)[::-1], dim=2
            )  # 16, 2 ,1

            # patchfy
            clean_latents_post = self.proj(clean_latents_post).flatten(2).transpose(1, 2)
            clean_latents_2x = self.proj_2x(clean_latents_2x).flatten(2).transpose(1, 2)
            clean_latents_4x = self.proj_4x(clean_latents_4x).flatten(2).transpose(1, 2)

            if add_last_motion < 2 and self.drop_mode == "drop":
                clean_latents_post = clean_latents_post[:, :0] if add_last_motion < 2 else clean_latents_post
                clean_latents_2x = clean_latents_2x[:, :0] if add_last_motion < 1 else clean_latents_2x

            motion_lat = torch.cat([clean_latents_post, clean_latents_2x, clean_latents_4x], dim=1)

            # rope
            start_time_id = -(self.zip_frame_buckets[:1].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[0]
            grid_sizes = [] if add_last_motion < 2 and self.drop_mode == "drop" else \
                        [
                            [torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                            torch.tensor([end_time_id, lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1),
                            torch.tensor([self.zip_frame_buckets[0], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1), ]
                        ]

            start_time_id = -(self.zip_frame_buckets[:2].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[1] // 2
            grid_sizes_2x = [] if add_last_motion < 1 and self.drop_mode == "drop" else \
            [
                [torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                torch.tensor([end_time_id, lat_height // 4, lat_width // 4]).unsqueeze(0).repeat(1, 1),
                torch.tensor([self.zip_frame_buckets[1], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1), ]
            ]

            start_time_id = -(self.zip_frame_buckets[:3].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[2] // 4
            grid_sizes_4x = [
                [
                    torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([end_time_id, lat_height // 8, lat_width // 8]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([self.zip_frame_buckets[2], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1),
                ]
            ]

            grid_sizes = grid_sizes + grid_sizes_2x + grid_sizes_4x

            motion_rope_emb = rope_precompute(
                motion_lat.detach().view(1, motion_lat.shape[1], self.num_heads, self.inner_dim // self.num_heads),
                grid_sizes,
                self.freqs,
                start=None
            )

            mot.append(motion_lat)
            mot_remb.append(motion_rope_emb)
        return mot, mot_remb


class AdaLayerNorm(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, elementwise_affine=False)

    def forward(self, x, temb):
        temb = self.linear(F.silu(temb))
        shift, scale = temb.chunk(2, dim=1)
        shift = shift[:, None, :]
        scale = scale[:, None, :]
        x = self.norm(x) * (1 + scale) + shift
        return x


class AudioInjector_WAN(nn.Module):

    def __init__(
        self,
        all_modules,
        all_modules_names,
        dim=2048,
        num_heads=32,
        inject_layer=[0, 27],
        enable_adain=False,
        adain_dim=2048,
    ):
        super().__init__()
        self.injected_block_id = {}
        audio_injector_id = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, DiTBlock):
                for inject_id in inject_layer:
                    if f'transformer_blocks.{inject_id}' in mod_name:
                        self.injected_block_id[inject_id] = audio_injector_id
                        audio_injector_id += 1

        self.injector = nn.ModuleList([CrossAttention(
            dim=dim,
            num_heads=num_heads,
        ) for _ in range(audio_injector_id)])
        self.injector_pre_norm_feat = nn.ModuleList([nn.LayerNorm(
            dim,
            elementwise_affine=False,
            eps=1e-6,
        ) for _ in range(audio_injector_id)])
        self.injector_pre_norm_vec = nn.ModuleList([nn.LayerNorm(
            dim,
            elementwise_affine=False,
            eps=1e-6,
        ) for _ in range(audio_injector_id)])
        if enable_adain:
            self.injector_adain_layers = nn.ModuleList([AdaLayerNorm(output_dim=dim * 2, embedding_dim=adain_dim) for _ in range(audio_injector_id)])


class CausalAudioEncoder(nn.Module):

    def __init__(self, dim=5120, num_layers=25, out_dim=2048, num_token=4, need_global=False):
        super().__init__()
        #print("initial CausalAudio",num_layers)
        #print("initial CausalAudio",num_token)
        self.encoder = MotionEncoder_tc(in_dim=dim, hidden_dim=out_dim, num_heads=num_token, need_global=need_global)
        weight = torch.ones((1, num_layers, 1, 1)) * 0.01

        self.weights = torch.nn.Parameter(weight)
        self.act = torch.nn.SiLU()

    def forward(self, features):
        # features B * num_layers * dim * video_length, 对每一层音频特征加权求和
        weights = self.act(self.weights.to(device=features.device, dtype=features.dtype))
        weights_sum = weights.sum(dim=1, keepdims=True)
        weighted_feat = ((features * weights) / weights_sum).sum(dim=1)  # b dim f
        weighted_feat = weighted_feat.permute(0, 2, 1)  # b f dim
        res = self.encoder(weighted_feat)  # b f n dim
        return res  # b f n dim


class WanS2VDiTBlock(DiTBlock):
    def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps, use_moe=False, lora_rank=64, use_depth_branch=False):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps)
        # 如果启用 MoE，则替换原始 FFN 为 MoE FFN
        # 注意：这里传入原始的 self.ffn 以便复用预训练权重
        if use_moe:
            original_ffn = self.ffn  # 保存原始 FFN 的引用
            self.ffn = MoEFFN(original_ffn, dim, lora_rank)
        # 否则保持父类的 FFN
        
        # Depth Video 辅助分支的独立 modulation 参数
        # 参考 UnityVideo 设计：不同模态使用独立的可学习 modulation 参数
        # 这样 depth 模态可以学习到与主序列不同的 AdaLN 调制参数
        self.use_depth_branch = use_depth_branch
        if use_depth_branch:
            # depth_modulation: 独立的可学习参数，shape 与 self.modulation 相同
            # 6 个参数: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
            self.depth_modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def _chunked_self_attention(self, input_x, freqs, seq_len_main, seq_len_depth, seq_len_target, depth_freqs, seq_len_ref=0, seq_len_extra_ref=0):
        """
        分块计算 Self-Attention (版本2: depth_mutual_visible=False)
        完全避开 attention mask，使用高效的 Flash Attention
        
        分块策略:
        - Block 1 (主序列 Self-Attention): [input_video, ref, extra_ref, motion] 互相可见
        - Block 2 (Depth 混合 Attention): depth 作为 Q，[depth, input_video, extra_ref] 作为 K,V
        
        注意: 这种模式下，video 看不到 depth，只有 depth 能看到 video (input_video + extra_ref)
        
        Args:
            input_x: 输入 tensor [B, S_total, D]，S_total = S_main + S_depth
            freqs: 完整序列的位置编码频率，可能是:
                   - 3维 [S_total, num_heads, head_dim//2]（从 pre_compute_freqs[0] 取出）
                   - 4维 [1, S_total, num_heads, head_dim//2]
            seq_len_main: 主序列长度 (input_video + ref + extra_ref + motion)
            seq_len_depth: depth 序列长度
            seq_len_target: input_video tokens 的长度
            depth_freqs: depth 的位置编码频率，可能是:
                   - 3维 [S_depth, num_heads, head_dim//2]
                   - 4维 [1, S_depth, num_heads, head_dim//2]
            seq_len_ref: ref (主参考帧) tokens 的长度
            seq_len_extra_ref: extra_ref (额外参考帧) tokens 的长度
        
        Returns:
            output: 分块计算后的输出 [B, S_total, D]
        """
        # 分离主序列和 depth 序列
        x_main = input_x[:, :seq_len_main]  # [B, S_main, D]
        x_depth = input_x[:, seq_len_main:]  # [B, S_depth, D]
        
        # ========== 处理 freqs 维度 ==========
        # freqs 可能是 3 维或 4 维，统一处理
        if freqs.dim() == 4:
            # 4维: [1, S_total, num_heads, head_dim//2]
            main_freqs = freqs[:, :seq_len_main]  # [1, S_main, ...]
            target_freqs = freqs[:, :seq_len_target]  # [1, S_target, ...]
        else:
            # 3维: [S_total, num_heads, head_dim//2]
            main_freqs = freqs[:seq_len_main]  # [S_main, ...]
            target_freqs = freqs[:seq_len_target]  # [S_target, ...]
        
        # 提取 extra_ref 的 freqs (如果存在)
        # 主序列结构: [input_video(seq_len_target), ref(seq_len_ref), extra_ref(seq_len_extra_ref), motion(...)]
        extra_ref_freqs = None
        if seq_len_extra_ref > 0 and seq_len_ref > 0:
            extra_ref_start = seq_len_target + seq_len_ref
            extra_ref_end = extra_ref_start + seq_len_extra_ref
            if freqs.dim() == 4:
                extra_ref_freqs = freqs[:, extra_ref_start:extra_ref_end]
            else:
                extra_ref_freqs = freqs[extra_ref_start:extra_ref_end]
        
        # ========== 处理 depth_freqs 维度 ==========
        # depth_freqs 可能是 3 维或 4 维，统一处理
        if depth_freqs.dim() == 4:
            # 4维: [1, S_depth, num_heads, head_dim//2]，需要 squeeze 成 3 维
            depth_freqs_3d = depth_freqs.squeeze(0)  # [S_depth, ...]
        else:
            # 3维: [S_depth, num_heads, head_dim//2]
            depth_freqs_3d = depth_freqs
        
        # 确保 target_freqs 也是 3 维
        if target_freqs.dim() == 4:
            target_freqs_3d = target_freqs.squeeze(0)  # [S_target, ...]
        else:
            target_freqs_3d = target_freqs
        
        # ========== Block 1: 主序列 Self-Attention (Flash Attention, 无 mask) ==========
        # [input_video, ref, extra_ref, motion] 互相可见
        main_out = self.self_attn(x_main, main_freqs)  # 无 attention_mask，使用 Flash Attention
        
        # ========== Block 2: Depth 混合 Attention ==========
        # depth 作为 Q，[depth, input_video, extra_ref] 作为 K,V
        # 这样 depth 能看到自己、input_video 和 extra_ref，但主序列看不到 depth
        
        # 提取 input_video tokens (主序列的前 seq_len_target 个)
        x_target = x_main[:, :seq_len_target]  # [B, S_target, D]
        
        # 构建 K,V 输入和对应的 freqs
        if seq_len_extra_ref > 0 and extra_ref_freqs is not None:
            # 提取 extra_ref tokens
            extra_ref_start = seq_len_target + seq_len_ref
            extra_ref_end = extra_ref_start + seq_len_extra_ref
            x_extra_ref = x_main[:, extra_ref_start:extra_ref_end]  # [B, S_extra_ref, D]
            
            # 确保 extra_ref_freqs 也是 3 维
            if extra_ref_freqs.dim() == 4:
                extra_ref_freqs_3d = extra_ref_freqs.squeeze(0)
            else:
                extra_ref_freqs_3d = extra_ref_freqs
            
            # 拼接 K,V 的输入: [depth, input_video, extra_ref]
            kv_input = torch.cat([x_depth, x_target, x_extra_ref], dim=1)  # [B, S_depth + S_target + S_extra_ref, D]
            kv_freqs = torch.cat([depth_freqs_3d, target_freqs_3d, extra_ref_freqs_3d], dim=0)  # [S_depth + S_target + S_extra_ref, ...]
        else:
            # 没有 extra_ref，保持原始逻辑: [depth, input_video]
            kv_input = torch.cat([x_depth, x_target], dim=1)  # [B, S_depth + S_target, D]
            kv_freqs = torch.cat([depth_freqs_3d, target_freqs_3d], dim=0)  # [S_depth + S_target, ...]
        
        # 计算 Q, K, V
        q_depth = self.self_attn.norm_q(self.self_attn.q(x_depth))
        k_kv = self.self_attn.norm_k(self.self_attn.k(kv_input))
        v_kv = self.self_attn.v(kv_input)
        
        # 应用 RoPE (使用 3 维的 freqs)
        q_depth = rope_apply(q_depth, depth_freqs_3d, self.self_attn.num_heads)
        k_kv = rope_apply(k_kv, kv_freqs, self.self_attn.num_heads)
        
        # Flash Attention (无 mask)
        depth_out = self.self_attn.attn(q_depth, k_kv, v_kv)
        depth_out = self.self_attn.o(depth_out)
        
        # 合并输出
        output = torch.cat([main_out, depth_out], dim=1)
        return output

    def forward(self, x, context, t_mod, seq_len_x, freqs, return_router_logits=False, timestep=None, is_conditional=True, face_mask=None, hand_mask=None, skip_moe_for_depth=False, seq_len_depth=0, seq_len_target=0, depth_freqs=None, depth_mutual_visible=True, seq_len_ref=0, seq_len_extra_ref=0, depth_context=None):
        """
        WanS2VDiTBlock 前向传播
        
        Args:
            x: 输入 tensor [B, S, D]
            context: 上下文 tensor (主视频 prompt embedding)
            t_mod: 时间调制 tensor
            seq_len_x: 目标帧序列长度 (用于 t_mod 分割)
            freqs: 位置编码频率
            return_router_logits: 是否返回 router logits
            timestep: 归一化的 timestep (0-1)，用于 MoE 路由控制
            is_conditional: 是否是 conditional 分支，用于 MoE 路由控制
            face_mask: 面部区域 mask (训练时使用)，shape (N,)
            hand_mask: 手部区域 mask (训练时使用)，shape (N,)
            skip_moe_for_depth: 是否跳过 depth tokens 的 MoE 处理
            seq_len_depth: depth tokens 的长度
            seq_len_target: input_video tokens 的长度 (用于 depth 的 K,V 拼接)
            depth_freqs: depth 的位置编码频率 (用于分块 attention)
            depth_mutual_visible: Depth 可见性模式
                - True (版本1): depth 和 video 相互可见，使用普通 Self-Attention
                - False (版本2): 只有 depth 能看到 video，video 看不到 depth，使用分块 Attention
            seq_len_ref: ref (主参考帧) tokens 的长度 (用于定位 extra_ref)
            seq_len_extra_ref: extra_ref (额外参考帧) tokens 的长度 (用于 depth 的 K,V 拼接)
            depth_context: depth 专用的上下文 tensor (depth prompt embedding)，用于 depth tokens 的 cross-attention
        """
        # 计算主序列长度
        seq_len_main = x.shape[1] - seq_len_depth if seq_len_depth > 0 else x.shape[1]
        
        # ========== AdaLN Modulation ==========
        # 主序列使用 self.modulation，depth 使用独立的 self.depth_modulation
        if seq_len_depth > 0 and self.use_depth_branch and hasattr(self, 'depth_modulation'):
            # 分离主序列和 depth 序列
            x_main = x[:, :seq_len_main]
            x_depth = x[:, seq_len_main:]
            
            # 主序列的 modulation (使用原始 self.modulation)
            t_mod_main = (self.modulation.unsqueeze(2).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
            t_mod_main = [
                torch.cat([element[:, :, 0].expand(1, seq_len_x, x.shape[-1]), element[:, :, 1].expand(1, seq_len_main - seq_len_x, x.shape[-1])], dim=1)
                for element in t_mod_main
            ]
            shift_msa_main, scale_msa_main, gate_msa_main, shift_mlp_main, scale_mlp_main, gate_mlp_main = t_mod_main
            
            # Depth 的 modulation (使用独立的 self.depth_modulation)
            # depth 使用 t_mod[:, :, 1] (与 ref/motion 相同的 timestep embedding)
            t_mod_depth = (self.depth_modulation.unsqueeze(2).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
            t_mod_depth = [element[:, :, 1].expand(1, seq_len_depth, x.shape[-1]) for element in t_mod_depth]
            shift_msa_depth, scale_msa_depth, gate_msa_depth, shift_mlp_depth, scale_mlp_depth, gate_mlp_depth = t_mod_depth
            
            # 拼接主序列和 depth 的 modulation 参数
            shift_msa = torch.cat([shift_msa_main, shift_msa_depth], dim=1)
            scale_msa = torch.cat([scale_msa_main, scale_msa_depth], dim=1)
            gate_msa = torch.cat([gate_msa_main, gate_msa_depth], dim=1)
            shift_mlp = torch.cat([shift_mlp_main, shift_mlp_depth], dim=1)
            scale_mlp = torch.cat([scale_mlp_main, scale_mlp_depth], dim=1)
            gate_mlp = torch.cat([gate_mlp_main, gate_mlp_depth], dim=1)

            # 释放中间 modulation tensor，避免在 checkpoint 重计算时占用 ~4.2 GB 显存
            del shift_msa_main, scale_msa_main, gate_msa_main, shift_mlp_main, scale_mlp_main, gate_mlp_main
            del shift_msa_depth, scale_msa_depth, gate_msa_depth, shift_mlp_depth, scale_mlp_depth, gate_mlp_depth
            del t_mod_main, t_mod_depth
        else:
            # 无 depth 或未启用 depth_modulation，使用原始逻辑
            t_mod = (self.modulation.unsqueeze(2).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
            # t_mod[:, :, 0] for x, t_mod[:, :, 1] for other like ref, motion, etc.
            t_mod = [
                torch.cat([element[:, :, 0].expand(1, seq_len_x, x.shape[-1]), element[:, :, 1].expand(1, x.shape[1] - seq_len_x, x.shape[-1])], dim=1)
                for element in t_mod
            ]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = t_mod
        
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        
        # Self-Attention: 根据 depth_mutual_visible 选择计算方式
        if seq_len_depth > 0 and depth_freqs is not None:
            if depth_mutual_visible:
                # print('path 1')
                # 版本1 (depth_mutual_visible=True): depth 和 video 相互可见
                # 使用普通 Self-Attention，所有 tokens 互相可见
                attn_out = self.self_attn(input_x, freqs)
            else:
                # print('path 2')
                # 版本2 (depth_mutual_visible=False): 只有 depth 能看到 video
                # 分块计算: 主序列 Self-Attention + Depth 混合 Attention
                # 完全避开 attention mask，使用高效的 Flash Attention
                attn_out = self._chunked_self_attention(
                    input_x, freqs, seq_len_main, seq_len_depth, seq_len_target, depth_freqs,
                    seq_len_ref=seq_len_ref, seq_len_extra_ref=seq_len_extra_ref
                )
        else:
            # 无 depth 时，使用普通 Self-Attention
            attn_out = self.self_attn(input_x, freqs)
        
        x = self.gate(x, gate_msa, attn_out)
        # Cross-Attention: 主序列使用 context (视频 prompt)，depth 使用 depth_context (depth prompt)
        if seq_len_depth > 0 and depth_context is not None:
            x_main_ca = x[:, :seq_len_main]
            x_depth_ca = x[:, seq_len_main:]
            # RMSNorm 是 per-token 的，拆分不影响结果
            x_main_ca = x_main_ca + self.cross_attn(self.norm3(x_main_ca), context)
            x_depth_ca = x_depth_ca + self.cross_attn(self.norm3(x_depth_ca), depth_context)
            x = torch.cat([x_main_ca, x_depth_ca], dim=1)
        else:
            x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        
        # MoE FFN 处理
        if hasattr(self.ffn, 'router'):
            # MoE FFN: 只对 input_video tokens (前 seq_len_x 个) 使用 MoE
            # 对 ref_latents、motion_latents 和 depth_latents 使用基础 FFN (ffn_base)
            # 这样可以避免多 chunk 推理时因 motion_latents 的 MoE 路由不一致导致的闪烁问题
            
            # 序列结构: [input_video (seq_len_x), ref, motion, depth (seq_len_depth)]
            # 需要分离出三部分:
            # 1. input_video: 前 seq_len_x 个 tokens -> 使用 MoE
            # 2. ref + motion: 中间部分 -> 使用 ffn_base
            # 3. depth: 最后 seq_len_depth 个 tokens (如果有) -> 使用 ffn_base
            
            total_len = input_x.shape[1]
            
            # 计算各部分的长度
            # seq_len_depth 可能为 0 (无 depth 分支)
            seq_len_ref_motion = total_len - seq_len_x - seq_len_depth
            
            # 分离各部分
            video_input = input_x[:, :seq_len_x]  # input_video tokens
            
            # 处理 input_video: 使用 MoE FFN
            if return_router_logits:
                video_ffn_out, router_logits = self.ffn(
                    video_input, 
                    timestep=timestep, 
                    is_conditional=is_conditional, 
                    return_router_logits=True,
                    face_mask=face_mask,
                    hand_mask=hand_mask,
                )
            else:
                video_ffn_out = self.ffn(
                    video_input, 
                    timestep=timestep, 
                    is_conditional=is_conditional, 
                    return_router_logits=False,
                    face_mask=face_mask,
                    hand_mask=hand_mask,
                )
                router_logits = None
            
            # 处理 ref + motion + depth: 使用 ffn_base
            if seq_len_ref_motion + seq_len_depth > 0:
                other_input = input_x[:, seq_len_x:]  # ref + motion + depth tokens
                other_ffn_out = self.ffn.ffn_base(other_input)
                
                # 合并输出
                ffn_out = torch.cat([video_ffn_out, other_ffn_out], dim=1)
            else:
                # 只有 input_video tokens
                ffn_out = video_ffn_out
            
            x = self.gate(x, gate_mlp, ffn_out)
            
            if return_router_logits:
                return x, router_logits
            return x
        else:
            # 普通 FFN
            x = self.gate(x, gate_mlp, self.ffn(input_x))
            if return_router_logits:
                return x, None
            return x


class WanS2VModel(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        cond_dim: int,
        audio_dim: int,
        num_audio_token: int,
        enable_adain: bool = True,
        audio_inject_layers: list = [0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39],
        zero_timestep: bool = True,
        add_last_motion: bool = True,
        framepack_drop_mode: str = "padd",
        fuse_vae_embedding_in_latents: bool = True,
        require_vae_embedding: bool = False,
        seperated_timestep: bool = False,
        require_clip_embedding: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.enable_adain = enable_adain
        self.add_last_motion = add_last_motion
        self.zero_timestep = zero_timestep
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.require_vae_embedding = require_vae_embedding
        self.seperated_timestep = seperated_timestep
        self.require_clip_embedding = require_clip_embedding

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # 支持 MoE FFN (默认不启用)
        self.blocks = nn.ModuleList([WanS2VDiTBlock(False, dim, num_heads, ffn_dim, eps, use_moe=False, lora_rank=64) for _ in range(num_layers)])
        self.head = Head(dim, out_dim, patch_size, eps)
        self.freqs = torch.cat(precompute_freqs_cis_3d(dim // num_heads), dim=1)

        self.cond_encoder = nn.Conv3d(cond_dim, dim, kernel_size=patch_size, stride=patch_size)
        #这里有可能需要修改CasualAudioEncoder初始化层数(试了无法修改,因为有预加载的模型参数不匹配)
        self.casual_audio_encoder = CausalAudioEncoder(dim=audio_dim, out_dim=dim, num_token=num_audio_token, need_global=enable_adain)
        all_modules, all_modules_names = torch_dfs(self.blocks, parent_name="root.transformer_blocks")
        self.audio_injector = AudioInjector_WAN(
            all_modules,
            all_modules_names,
            dim=dim,
            num_heads=num_heads,
            inject_layer=audio_inject_layers,
            enable_adain=enable_adain,
            adain_dim=dim,
        )
        self.trainable_cond_mask = nn.Embedding(3, dim)
        self.frame_packer = FramePackMotioner(inner_dim=dim, num_heads=num_heads, zip_frame_buckets=[1, 2, 16], drop_mode=framepack_drop_mode)
        
        # MoE 相关配置
        self._moe_enabled = False
        self._expert_hidden_dim = None  # MoE 1.4: 专家 FFN 的隐藏层维度
        
        # Audio Face Mask 配置
        # use_audio_face_mask: 是否对 audio injection 的 residual 施加 face mask
        #   False: 不使用 mask，audio 影响所有空间位置（当前默认行为）
        #   True: 使用 face mask，audio 主要影响面部区域
        # audio_mask_train_source: 训练时 mask 的来源
        #   "gt": 使用 GT bbox face_mask（硬 mask）
        #   "router": 使用当前 block MoE router 的 w_face（软 mask，detached）
        # 推理时始终使用 router 的 w_face（无 GT 可用）
        self.use_audio_face_mask = False
        self.audio_mask_train_source = "gt"  # "gt" or "router"
        self._audio_gt_mask = None  # 临时存储 GT face_mask，由 pipeline 在 block loop 前设置
        
        # Depth Video 分支相关配置
        self._depth_branch_enabled = False
        # Depth Video 专用模块 (初始化时为 None，调用 enable_depth_branch 后创建)
        self.depth_patch_embedding = None  # Conv3d: 将 depth latents 转换为 tokens
        self.depth_head = None  # Head: 将 depth tokens 转换回 latents
        self.depth_cond_mask = None  # Embedding: depth token 类型标识 (类型 = 3)

    def enable_moe_ffn(self, expert_hidden_dim: int = None):
        """
        动态启用 MoE FFN，将所有 DiT Block 的 FFN 替换为 MoE FFN
        
        MoE 1.4 版本：使用 LightweightFFN 作为专家网络
        
        注意: 应在加载预训练权重之后调用此方法，这样：
        1. 原始 FFN 权重已经加载到 self.ffn 中
        2. MoE 包装后，原始权重会保留在 ffn.ffn_base 中
        3. 新增的 router、hand_expert、face_expert 使用随机初始化
        
        Args:
            expert_hidden_dim: 专家 FFN 的隐藏层维度 (默认为 dim)
        """
        if self._moe_enabled:
            print("[MoE] MoE FFN already enabled, skipping.")
            return
        
        self._expert_hidden_dim = expert_hidden_dim if expert_hidden_dim is not None else self.dim
        converted_count = 0
        
        for block_idx, block in enumerate(self.blocks):
            if hasattr(block, 'ffn') and not hasattr(block.ffn, 'router'):
                # 保存原始 FFN
                original_ffn = block.ffn
                # 创建 MoE FFN 包装 (MoE 1.4: 使用 expert_hidden_dim)
                block.ffn = MoEFFN(original_ffn, self.dim, self._expert_hidden_dim)
                converted_count += 1
        
        self._moe_enabled = True
        print(f"[MoE] Enabled MoE FFN for {converted_count} blocks with expert_hidden_dim={self._expert_hidden_dim}")
        print(f"[MoE] Original FFN weights are preserved in ffn.ffn_base")
        print(f"[MoE] New modules (router, hand_expert, face_expert) require training")
    
    def is_moe_enabled(self) -> bool:
        """检查 MoE FFN 是否已启用"""
        return self._moe_enabled

    def set_audio_face_mask_config(self, use_audio_face_mask: bool = False, audio_mask_train_source: str = "gt"):
        """
        配置 Audio Face Mask
        
        Args:
            use_audio_face_mask: 是否对 audio injection 使用 face mask
                False: audio 影响所有空间位置（默认行为）
                True: audio 主要影响面部区域
            audio_mask_train_source: 训练时 mask 来源
                "gt": 使用 GT bbox face_mask（二值硬 mask）
                "router": 使用当前 block MoE router 的 w_face（软 mask，detached）
        """
        assert audio_mask_train_source in ("gt", "router"), \
            f"audio_mask_train_source must be 'gt' or 'router', got '{audio_mask_train_source}'"
        self.use_audio_face_mask = use_audio_face_mask
        self.audio_mask_train_source = audio_mask_train_source
        print(f"[Audio Face Mask] use_audio_face_mask={use_audio_face_mask}, "
              f"audio_mask_train_source={audio_mask_train_source}")

    def set_audio_gt_mask(self, face_mask):
        """设置当前 step 的 GT face_mask，由 pipeline 在 block loop 前调用"""
        self._audio_gt_mask = face_mask
    
    def clear_audio_gt_mask(self):
        """清理临时 GT face_mask，由 pipeline 在 block loop 后调用"""
        self._audio_gt_mask = None

    def _get_audio_face_mask_for_block(self, block_idx):
        """
        获取指定 block 的 audio face mask
        
        优先级:
        1. 如果有 GT mask 且 audio_mask_train_source == "gt": 使用 GT（训练时）
        2. 否则尝试从当前 block 的 MoE router 获取 w_face（推理时/router模式训练时）
        3. 都没有则返回 None
        
        注意: 返回的 mask 不做 dtype 转换，由调用方统一 .to(residual_out.dtype)
              避免在 checkpointed 函数内产生不确定的 dtype 转换操作
        
        Returns:
            mask: (1, N, 1) 或 (B, N, 1) 或 None
        """
        # 训练时使用 GT mask
        if self._audio_gt_mask is not None and self.audio_mask_train_source == "gt":
            # GT face_mask shape: (N,) → (1, N, 1) for broadcasting
            # 不用 .float()，调用方会统一 dtype
            return self._audio_gt_mask.unsqueeze(0).unsqueeze(-1)
        
        # 从 router 获取 w_face
        block = self.blocks[block_idx]
        if hasattr(block.ffn, 'get_cached_w_face'):
            w_face = block.ffn.get_cached_w_face()
            if w_face is not None:
                return w_face  # (B, N, 1), already detached
        
        return None

    def enable_depth_branch(self):
        """
        启用 Depth Video 辅助分支
        
        创建 depth 专用的 patchify/unpatchify 模块，并从主模块初始化权重。
        这些模块用于将 depth video latents 转换为 tokens 并转换回 latents。
        
        同时为每个 DiT Block 创建独立的 depth_modulation 参数，
        参考 UnityVideo 设计：不同模态使用独立的可学习 modulation 参数。
        
        注意: 应在加载预训练权重之后调用此方法，这样：
        1. 主模块的权重已经加载
        2. Depth 模块从主模块复制权重作为初始化
        3. Depth 模块需要单独训练
        """
        if self._depth_branch_enabled:
            print("[Depth Branch] Depth branch already enabled, skipping.")
            return
        
        # 创建 depth_patch_embedding: 与主 patch_embedding 结构相同
        self.depth_patch_embedding = nn.Conv3d(
            self.in_dim, self.dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        # 从主 patch_embedding 复制权重
        self.depth_patch_embedding.load_state_dict(self.patch_embedding.state_dict())
        
        # 创建 depth_head: 与主 head 结构相同
        self.depth_head = Head(self.dim, self.in_dim, self.patch_size, eps=1e-6)
        # 从主 head 复制权重
        self.depth_head.load_state_dict(self.head.state_dict())
        
        # 创建 depth_cond_mask: depth token 类型标识 (类型 = 3)
        # 注意: 主分支使用 0=目标帧, 1=参考帧, 2=motion帧
        # depth 分支使用 3=depth帧
        self.depth_cond_mask = nn.Embedding(1, self.dim)
        # 初始化为零，让模型学习 depth 的特殊表示
        nn.init.zeros_(self.depth_cond_mask.weight)
        
        # 注意: depth_context 由训练脚本动态编码，不在模型中存储
        # depth prompt 格式: "{depth_map_description}{video_prompt}"
        
        # 为每个 DiT Block 创建独立的 depth_modulation 参数
        # 参考 UnityVideo 设计：不同模态使用独立的可学习 modulation 参数
        # 从原本 video 的 modulation 复制权重作为初始化
        depth_modulation_count = 0
        for block in self.blocks:
            if isinstance(block, WanS2VDiTBlock):
                # 设置 use_depth_branch 标志
                block.use_depth_branch = True
                # 创建独立的 depth_modulation 参数
                # shape 与 self.modulation 相同: (1, 6, dim)
                # 6 个参数: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
                # 从原本 video 的 modulation 复制权重作为初始化
                block.depth_modulation = nn.Parameter(
                    block.modulation.data.clone()
                )
                depth_modulation_count += 1
        
        self._depth_branch_enabled = True
        print(f"[Depth Branch] Enabled Depth Video branch")
        print(f"[Depth Branch] Created depth_patch_embedding (initialized from patch_embedding)")
        print(f"[Depth Branch] Created depth_head (initialized from head)")
        print(f"[Depth Branch] Created depth_cond_mask (type=3, zero initialized)")
        print(f"[Depth Branch] Note: depth_context will be dynamically encoded with video prompt in training")
        print(f"[Depth Branch] Created depth_modulation for {depth_modulation_count} blocks (initialized from modulation)")
    
    def is_depth_branch_enabled(self) -> bool:
        """检查 Depth Video 分支是否已启用"""
        return self._depth_branch_enabled
    
    def depth_patchify(self, x: torch.Tensor):
        """
        将 depth video latents 转换为 tokens
        
        Args:
            x: depth latents, shape [B, C, F, H, W]
        
        Returns:
            tokens: shape [B, F*H*W, dim]
            grid_size: (F, H, W)
        """
        if self.depth_patch_embedding is None:
            raise RuntimeError("Depth branch not enabled. Call enable_depth_branch() first.")
        
        x = self.depth_patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size
    
    def depth_unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor, t_mod: torch.Tensor):
        """
        将 depth tokens 转换回 latents
        
        Args:
            x: depth tokens, shape [B, N, dim]
            grid_size: (F, H, W)
            t_mod: time modulation for head
        
        Returns:
            latents: shape [B, C, F*patch_f, H*patch_h, W*patch_w]
        """
        if self.depth_head is None:
            raise RuntimeError("Depth branch not enabled. Call enable_depth_branch() first.")
        
        # 通过 depth_head 处理
        x = self.depth_head(x, t_mod)
        # unpatchify
        return rearrange(
            x,
            'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2]
        )
    
    def get_depth_grid_sizes(self, grid_size_depth: Tuple[int, int, int]):
        """
        计算 depth video 的位置编码 grid sizes
        
        Depth video 使用负数宽度位置编码，避免与主视频位置编码冲突。
        这样训练 480p、推理 720p 时不会出现问题。
        
        Args:
            grid_size_depth: (f, h, w) - depth video 的 grid size
        
        Returns:
            grid_sizes: 位置编码 grid sizes 列表
            - 时间位置: [0, f) (与主视频相同)
            - 高度位置: [0, h) (与主视频相同)
            - 宽度位置: [-w, 0) (负数，与主视频不重叠)
        """
        f, h, w = grid_size_depth
        # 使用负数宽度位置编码
        # start: (0, 0, -w), end: (f, h, 0), target: (f, h, w)
        grid_sizes_depth = [[
            torch.tensor([0, 0, -w]).unsqueeze(0),  # start position
            torch.tensor([f, h, 0]).unsqueeze(0),   # end position (exclusive)
            torch.tensor([f, h, w]).unsqueeze(0),   # target size for interpolation
        ]]
        return grid_sizes_depth

    def patchify(self, x: torch.Tensor):
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2]
        )

    def process_motion_frame_pack(self, motion_latents, drop_motion_frames=False, add_last_motion=2):
        flattern_mot, mot_remb = self.frame_packer(motion_latents, add_last_motion)
        if drop_motion_frames:
            return [m[:, :0] for m in flattern_mot], [m[:, :0] for m in mot_remb]
        else:
            return flattern_mot, mot_remb

    def inject_motion(self, x, rope_embs, mask_input, motion_latents, drop_motion_frames=True, add_last_motion=2):
        # inject the motion frames token to the hidden states
        mot, mot_remb = self.process_motion_frame_pack(motion_latents, drop_motion_frames=drop_motion_frames, add_last_motion=add_last_motion)
        if len(mot) > 0:
            x = torch.cat([x, mot[0]], dim=1)
            rope_embs = torch.cat([rope_embs, mot_remb[0]], dim=1)
            mask_input = torch.cat(
                [mask_input, 2 * torch.ones([1, x.shape[1] - mask_input.shape[1]], device=mask_input.device, dtype=mask_input.dtype)], dim=1
            )
        return x, rope_embs, mask_input


    #音频条件如何注入的核心函数,用全局特征来做AdaIn调制,用局部特征来做Cross_attetion
    def after_transformer_block(self, block_idx, hidden_states, audio_emb_global, audio_emb, original_seq_len, use_unified_sequence_parallel=False):
        # 非注入 block 直接返回，避免无意义的 clone (~703 MB/次)
        if block_idx not in self.audio_injector.injected_block_id.keys():
            return hidden_states
        new_hidden_states = hidden_states.clone()
        if True:  # block_idx 已确认在 injected_block_id 中
            audio_attn_id = self.audio_injector.injected_block_id[block_idx]
            num_frames = audio_emb.shape[1]
            if use_unified_sequence_parallel:
                from xfuser.core.distributed import get_sp_group
                hidden_states = get_sp_group().all_gather(hidden_states, dim=1)

            input_hidden_states = hidden_states[:, :original_seq_len].clone()  # b (f h w) c
            input_hidden_states = rearrange(input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames)

            audio_emb_global = rearrange(audio_emb_global, "b t n c -> (b t) n c")
            #print("inner param",input_hidden_states.shape)   [20,1560,5120]
            #print("inner param",audio_emb_global.shape)      [20,1,5120]
            adain_hidden_states = self.audio_injector.injector_adain_layers[audio_attn_id](input_hidden_states, temb=audio_emb_global[:, 0])
            attn_hidden_states = adain_hidden_states

            audio_emb = rearrange(audio_emb, "b t n c -> (b t) n c", t=num_frames)
            attn_audio_emb = audio_emb
            residual_out = self.audio_injector.injector[audio_attn_id](attn_hidden_states, attn_audio_emb)
            residual_out = rearrange(residual_out, "(b t) n c -> b (t n) c", t=num_frames)
            
            # Audio Face Mask: 让 audio cross-attention 的 residual 主要影响面部区域
            if self.use_audio_face_mask:
                # breakpoint()
                audio_face_mask = self._get_audio_face_mask_for_block(block_idx)
                if audio_face_mask is not None:
                    # audio_face_mask: (1, N, 1) for GT 或 (B, N, 1) for router
                    # residual_out: (B, N, C)
                    # 面部区域保持全量 audio residual，非面部区域被压制
                    residual_out = residual_out * audio_face_mask.to(residual_out.dtype)
            
            new_hidden_states[:, :original_seq_len] = new_hidden_states[:, :original_seq_len] + residual_out
            #hidden_states[:, :original_seq_len] = hidden_states[:, :original_seq_len] + residual_out
            if use_unified_sequence_parallel:
                from xfuser.core.distributed import get_sequence_parallel_world_size, get_sequence_parallel_rank
                new_hidden_states = torch.chunk(new_hidden_states, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
        return new_hidden_states
    #audio_input is
    #73 motion_frames, 19 is motion latent frames
    def cal_audio_emb(self, audio_input, motion_frames=[73, 19]):
        #首帧重复73次,模拟motion_frames片段
        audio_input = torch.cat([audio_input[..., 0:1].repeat(1, 1, 1, motion_frames[0]), audio_input], dim=-1)
        #print("cal_audio_emb audio_input",audio_input.shape)
        audio_emb_global, audio_emb = self.casual_audio_encoder(audio_input)
        audio_emb_global = audio_emb_global[:, motion_frames[1]:].clone()
        merged_audio_emb = audio_emb[:, motion_frames[1]:, :]
        return audio_emb_global, merged_audio_emb

    def get_grid_sizes(self, grid_size_x, grid_size_ref, grid_size_extra_ref=None, extra_ref_scale=1.0):
        """计算位置编码的grid sizes
        
        注意: grid_sizes使用左闭右开区间 [start, end)
        
        Args:
            grid_size_x: 目标帧的grid size (f, h, w)
            grid_size_ref: 主参考帧的grid size (rf, rh, rw)
            grid_size_extra_ref: 额外参考帧的grid size (erf, erh, erw)，可选
            extra_ref_scale: 额外参考帧的RoPE空间缩放因子 (0~1)，
                             对h/w维度的RoPE位置索引做压缩，
                             使模型认为商品占据更小的空间范围。
                             默认 1.0 表示不缩放。
        
        Returns:
            grid_sizes: 位置编码grid sizes列表
            - 目标帧: 时间位置 [0, f)
            - 主参考帧: 时间位置 30 (区间[30, 31))
            - 额外参考帧: 时间位置 31 (区间[31, 32), 如果提供)
        """
        f, h, w = grid_size_x
        rf, rh, rw = grid_size_ref
        grid_sizes_x = torch.tensor([f, h, w], dtype=torch.long).unsqueeze(0)
        grid_sizes_x = [[torch.zeros_like(grid_sizes_x), grid_sizes_x, grid_sizes_x]]
        # 主参考帧: 位置编码30 (区间[30, 31))
        grid_sizes_ref = [[
            torch.tensor([30, 0, 0]).unsqueeze(0),
            torch.tensor([31, rh, rw]).unsqueeze(0),
            torch.tensor([1, rh, rw]).unsqueeze(0),
        ]]
        
        result = grid_sizes_x + grid_sizes_ref
        
        # 额外参考帧: 位置编码31 (区间[31, 32), 紧跟主参考帧)
        if grid_size_extra_ref is not None:
            erf, erh, erw = grid_size_extra_ref
            # 对 h/w 维度的 RoPE 目标尺寸做 scale，压缩位置索引范围
            # 例如 scale=0.7 时，h 位置从 [0, erh) 压缩到 [0, erh*0.7)
            # 模型会认为图像内容占据更小的空间范围
            scaled_erh = max(1, round(erh * extra_ref_scale))
            scaled_erw = max(1, round(erw * extra_ref_scale))
            if extra_ref_scale != 1.0:
                print(f"[RoPE Scale] extra_ref_scale={extra_ref_scale:.4f}, "
                      f"RoPE target: ({erh},{erw}) -> ({scaled_erh},{scaled_erw})")
            grid_sizes_extra_ref = [[
                torch.tensor([31, 0, 0]).unsqueeze(0),
                torch.tensor([32, erh, erw]).unsqueeze(0),
                torch.tensor([1, scaled_erh, scaled_erw]).unsqueeze(0),
            ]]
            result = result + grid_sizes_extra_ref
        
        return result

    def forward(
        self,
        latents,
        timestep,
        context,
        audio_input,
        motion_latents,
        pose_cond,
        use_gradient_checkpointing_offload=False,
        use_gradient_checkpointing=False,
        return_router_logits=False,
        is_conditional=True
    ):
        origin_ref_latents = latents[:, :, 0:1]
        x = latents[:, :, 1:]

        # context embedding
        context = self.text_embedding(context)

        # audio encode
        audio_emb_global, merged_audio_emb = self.cal_audio_emb(audio_input)

        # x and pose_cond
        pose_cond = torch.zeros_like(x) if pose_cond is None else pose_cond
        x, (f, h, w) = self.patchify(self.patch_embedding(x) + self.cond_encoder(pose_cond))  # torch.Size([1, 29120, 5120])
        seq_len_x = x.shape[1]

        # reference image
        ref_latents, (rf, rh, rw) = self.patchify(self.patch_embedding(origin_ref_latents))  # torch.Size([1, 1456, 5120])
        grid_sizes = self.get_grid_sizes((f, h, w), (rf, rh, rw))
        x = torch.cat([x, ref_latents], dim=1)
        # mask
        mask = torch.cat([torch.zeros([1, seq_len_x]), torch.ones([1, ref_latents.shape[1]])], dim=1).to(torch.long).to(x.device)
        # freqs
        pre_compute_freqs = rope_precompute(
            x.detach().view(1, x.size(1), self.num_heads, self.dim // self.num_heads), grid_sizes, self.freqs, start=None
        )
        # motion
        #print("motion process in here")
        x, pre_compute_freqs, mask = self.inject_motion(x, pre_compute_freqs, mask, motion_latents, add_last_motion=2)

        x = x + self.trainable_cond_mask(mask).to(x.dtype)

        # t_mod
        # 保存原始的归一化 timestep 值用于 MoE 路由控制
        # timestep 是归一化的值 (0-1)，取第一个元素作为标量
        normalized_timestep = timestep[0].item() if timestep.numel() > 0 else 0.0
        
        timestep = torch.cat([timestep, torch.zeros([1], dtype=timestep.dtype, device=timestep.device)])
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim)).unsqueeze(2).transpose(0, 2)

        # Router logits 收集器（训练时使用）
        router_logits_cache = [] if return_router_logits else None

        def create_custom_forward(module, timestep=None, is_conditional=True):
            """创建 block 的包装函数，支持 MoE 参数传递
            
            timestep 和 is_conditional 是标量，通过闭包捕获不会有显存问题
            """
            def custom_forward(x, ctx, tm, seq_len, freq):
                return module(x, ctx, tm, seq_len, freq, timestep=timestep, is_conditional=is_conditional)
            return custom_forward

        for block_id, block in enumerate(self.blocks):
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    # timestep 和 is_conditional 通过闭包传递（标量，不会导致显存问题）
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block, timestep=normalized_timestep, is_conditional=is_conditional),
                        x,
                        context,
                        t_mod,
                        seq_len_x,
                        pre_compute_freqs[0],
                        use_reentrant=False,
                    )
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(lambda x: self.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                        x,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                # timestep 和 is_conditional 通过闭包传递（标量，不会导致显存问题）
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block, timestep=normalized_timestep, is_conditional=is_conditional),
                    x,
                    context,
                    t_mod,
                    seq_len_x,
                    pre_compute_freqs[0],
                    use_reentrant=False,
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(lambda x: self.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                    x,
                    use_reentrant=False,
                )
            else:
                if return_router_logits:
                    x_out = block(
                        x, context, t_mod, seq_len_x, pre_compute_freqs[0], 
                        return_router_logits=True,
                        timestep=normalized_timestep,
                        is_conditional=is_conditional
                    )
                    if isinstance(x_out, tuple):
                        x, router_logits = x_out
                        if router_logits is not None:
                            router_logits_cache.append(router_logits)
                    else:
                        x = x_out
                else:
                    x = block(
                        x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                        timestep=normalized_timestep,
                        is_conditional=is_conditional
                    )
                x = self.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)

        x = x[:, :seq_len_x]
        x = self.head(x, t[:-1])
        x = self.unpatchify(x, (f, h, w))
        # make compatible with wan video
        x = torch.cat([origin_ref_latents, x], dim=2)
        
        if return_router_logits:
            return x, router_logits_cache
        return x

    @staticmethod
    def state_dict_converter():
        return WanS2VModelStateDictConverter()


class WanS2VModelStateDictConverter:

    def __init__(self):
        pass

    def from_civitai(self, state_dict):
        config = {}
        if hash_state_dict_keys(state_dict) == "966cffdcc52f9c46c391768b27637614":
            config = {
                "dim": 5120,
                "in_dim": 16,
                "ffn_dim": 13824,
                "out_dim": 16,
                "text_dim": 4096,
                "freq_dim": 256,
                "eps": 1e-06,
                "patch_size": (1, 2, 2),
                "num_heads": 40,
                "num_layers": 40,
                "cond_dim": 16,
                "audio_dim": 1024,
                "num_audio_token": 4,
            }
        return state_dict, config
