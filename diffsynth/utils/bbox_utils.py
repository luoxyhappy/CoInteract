"""
Bbox 到 Token Mask 转换工具

将原始视频坐标系中的 bbox 转换为 DiT token 序列的 mask
"""

import json
import torch
from typing import Dict, List, Tuple, Optional, Union


def parse_bbox_from_csv_field(bbox_json_str: str) -> Dict:
    """
    解析 CSV 中的 bbox JSON 字符串
    
    Args:
        bbox_json_str: JSON 格式的 bbox 字符串
        
    Returns:
        解析后的 bbox 字典
    """
    if not bbox_json_str or bbox_json_str == '':
        return {}
    try:
        return json.loads(bbox_json_str)
    except json.JSONDecodeError:
        return {}


def bbox_to_token_mask(
    bbox_data: Dict,
    num_frames: int,
    latent_h: int,
    latent_w: int,
    patch_size: Tuple[int, int] = (2, 2),
    original_size: Optional[Tuple[int, int]] = None,
    target_size: Optional[Tuple[int, int]] = None,
    vae_scale_factor: int = 8,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    将 bbox 坐标转换为 token mask
    
    坐标转换链（和 ImageCropAndResize 一致）:
    原始坐标 -> resize (scale=max) -> center_crop -> VAE latent -> patch token
    
    Args:
        bbox_data: bbox 数据字典, 格式:
            - face: {"frame_2": [x1, y1, x2, y2], ...}
            - hand_object: {"frame_2": {"l_h": [x1,y1,x2,y2], "r_h": [...], "obj": [...]}, ...}
        num_frames: latent 的帧数 (不包含第0帧参考图)
        latent_h: latent 空间的高度
        latent_w: latent 空间的宽度
        patch_size: patch embedding 的尺寸 (patch_h, patch_w)
        original_size: 原始视频尺寸 (H, W), 如果为 None 则跳过坐标转换
        target_size: resize 目标尺寸 (H, W), 如果为 None 则跳过坐标转换
        vae_scale_factor: VAE 下采样率
        device: tensor 所在设备
        
    Returns:
        mask: bool tensor, shape (N,) 
              N = num_frames * (latent_h // patch_h) * (latent_w // patch_w)
    """
    patch_h, patch_w = patch_size
    H_token = latent_h // patch_h
    W_token = latent_w // patch_w
    N = num_frames * H_token * W_token
    
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    
    if not bbox_data:
        return mask
    
    for frame_key, bbox_value in bbox_data.items():
        # 解析帧索引: "frame_2" -> latent_frame_idx
        if not frame_key.startswith('frame_'):
            continue
            
        try:
            frame_idx = int(frame_key.split('_')[1])
        except (IndexError, ValueError):
            continue
        
        # frame_2, frame_6, frame_10, ... 对应 latent frame 0, 1, 2, ...
        # 因为 csv 中 frame_2 是 latent 除第0帧后的第一帧
        latent_frame_idx = (frame_idx - 2) // 4
        
        # 跳过超出范围的帧
        if latent_frame_idx < 0 or latent_frame_idx >= num_frames:
            continue
        
        # 处理不同的 bbox 格式
        bboxes_to_process = []
        
        if isinstance(bbox_value, list) and len(bbox_value) == 4:
            # face 格式: [x1, y1, x2, y2]
            bboxes_to_process.append(bbox_value)
        elif isinstance(bbox_value, dict):
            # hand_object 格式: {"l_h": [...], "r_h": [...], "obj": [...]}
            # 注意: 只考虑手部 bbox (l_h, r_h)，不考虑物体 bbox (obj)
            # 这样手部 expert 只关注手部区域，不会被物体区域干扰
            for key in ['l_h', 'r_h']:
                if key in bbox_value and bbox_value[key]:
                    if isinstance(bbox_value[key], list) and len(bbox_value[key]) == 4:
                        bboxes_to_process.append(bbox_value[key])
        
        # 处理每个 bbox
        for bbox in bboxes_to_process:
            x1, y1, x2, y2 = bbox
            
            # 坐标转换链（和 ImageCropAndResize 一致）:
            # 原始坐标 -> resize (scale=max) -> center_crop -> 最终坐标
            if original_size is not None and target_size is not None:
                orig_h, orig_w = original_size
                target_h, target_w = target_size
                
                # 1. 计算 resize scale（和 ImageCropAndResize.crop_and_resize 一致）
                scale = max(target_w / orig_w, target_h / orig_h)
                
                # 2. resize 后的尺寸
                resized_h = round(orig_h * scale)
                resized_w = round(orig_w * scale)
                
                # 3. 应用 resize 到 bbox 坐标
                x1_scaled = x1 * scale
                y1_scaled = y1 * scale
                x2_scaled = x2 * scale
                y2_scaled = y2 * scale
                
                # 4. 计算 center_crop 的偏移量
                crop_offset_x = (resized_w - target_w) / 2
                crop_offset_y = (resized_h - target_h) / 2
                
                # 5. 应用 center_crop 偏移
                x1_cropped = x1_scaled - crop_offset_x
                y1_cropped = y1_scaled - crop_offset_y
                x2_cropped = x2_scaled - crop_offset_x
                y2_cropped = y2_scaled - crop_offset_y
                
                # 6. 检查是否在 crop 后的有效范围内
                if x2_cropped < 0 or y2_cropped < 0 or x1_cropped > target_w or y1_cropped > target_h:
                    continue
                
                # 7. 裁剪到有效边界
                x1_resized = max(0, x1_cropped)
                y1_resized = max(0, y1_cropped)
                x2_resized = min(target_w, x2_cropped)
                y2_resized = min(target_h, y2_cropped)
            else:
                # 跳过坐标转换，假设 bbox 已经是目标分辨率的坐标
                x1_resized, y1_resized = x1, y1
                x2_resized, y2_resized = x2, y2
            
            # 3. 转换到 latent 坐标
            x1_latent = x1_resized / vae_scale_factor
            y1_latent = y1_resized / vae_scale_factor
            x2_latent = x2_resized / vae_scale_factor
            y2_latent = y2_resized / vae_scale_factor
            
            # 4. 转换到 patch 坐标
            h1_patch = int(y1_latent / patch_h)
            w1_patch = int(x1_latent / patch_w)
            h2_patch = int(y2_latent / patch_h)
            w2_patch = int(x2_latent / patch_w)
            
            # 确保在有效范围内
            h1_patch = max(0, min(h1_patch, H_token - 1))
            w1_patch = max(0, min(w1_patch, W_token - 1))
            h2_patch = max(0, min(h2_patch, H_token - 1))
            w2_patch = max(0, min(w2_patch, W_token - 1))
            
            # 5. 标记 token mask
            for h in range(h1_patch, h2_patch + 1):
                for w in range(w1_patch, w2_patch + 1):
                    token_idx = latent_frame_idx * H_token * W_token + h * W_token + w
                    if 0 <= token_idx < N:
                        mask[token_idx] = True
    
    return mask


def create_masks_from_metadata(
    face_bbox_str: str,
    hand_bbox_str: str,
    num_frames: int,
    latent_h: int,
    latent_w: int,
    patch_size: Tuple[int, int] = (2, 2),
    original_size: Optional[Tuple[int, int]] = None,
    target_size: Optional[Tuple[int, int]] = None,
    vae_scale_factor: int = 8,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 CSV metadata 创建 face 和 hand mask
    
    Args:
        face_bbox_str: face 字段的 JSON 字符串
        hand_bbox_str: hand_object 字段的 JSON 字符串
        original_size: 原始视频尺寸 (H, W)
        target_size: 目标分辨率 (H, W)，即训练时的 height/width
        其他参数同 bbox_to_token_mask
        
    Returns:
        (face_mask, hand_mask): 两个 bool tensor, shape (N,)
    """
    face_data = parse_bbox_from_csv_field(face_bbox_str)
    hand_data = parse_bbox_from_csv_field(hand_bbox_str)
    
    face_mask = bbox_to_token_mask(
        face_data, num_frames, latent_h, latent_w, patch_size,
        original_size, target_size, vae_scale_factor, device
    )
    
    hand_mask = bbox_to_token_mask(
        hand_data, num_frames, latent_h, latent_w, patch_size,
        original_size, target_size, vae_scale_factor, device
    )
    
    return face_mask, hand_mask


def create_router_targets(
    face_mask: torch.Tensor,
    hand_mask: torch.Tensor
) -> torch.Tensor:
    """
    根据 face 和 hand mask 创建 router 的 target distribution
    
    采用手部优先策略:
    - 手部 tokens -> [0, 1, 0] (hand expert)
    - 面部 tokens (非手部) -> [0, 0, 1] (face expert)
    - 普通 tokens -> [1, 0, 0] (base expert)
    
    Args:
        face_mask: bool tensor, shape (N,)
        hand_mask: bool tensor, shape (N,)
        
    Returns:
        targets: float tensor, shape (N, 3)
    """
    N = face_mask.shape[0]
    device = face_mask.device
    
    # 初始化为普通 tokens [1, 0, 0]
    targets = torch.zeros(N, 3, device=device)
    targets[:, 0] = 1.0  # base expert
    
    # 标记面部 tokens [0, 0, 1]
    targets[face_mask, :] = torch.tensor([0.0, 0.0, 1.0], device=device)
    
    # 标记手部 tokens [0, 1, 0] - 会覆盖面部和手部重叠的部分
    targets[hand_mask, :] = torch.tensor([0.0, 1.0, 0.0], device=device)
    
    return targets


def create_loss_weights(
    face_mask: torch.Tensor,
    hand_mask: torch.Tensor,
    base_weight: float = 1.0,
    face_weight: float = 10.0,
    hand_weight: float = 10.0
) -> torch.Tensor:
    """
    创建 router loss 的权重
    
    Args:
        face_mask: bool tensor, shape (N,)
        hand_mask: bool tensor, shape (N,)
        base_weight: 普通 tokens 的权重
        face_weight: 面部 tokens 的权重
        hand_weight: 手部 tokens 的权重
        
    Returns:
        weights: float tensor, shape (N,)
    """
    N = face_mask.shape[0]
    device = face_mask.device
    
    weights = torch.ones(N, device=device) * base_weight
    
    # 仅面部 (非手部)
    weights[face_mask & ~hand_mask] = face_weight
    
    # 手部 (包含与面部重叠的部分)
    weights[hand_mask] = hand_weight
    
    return weights
