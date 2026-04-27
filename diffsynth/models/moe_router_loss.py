"""
Router Loss 计算模块

用于训练时计算 MoE Router 的监督 loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


def compute_router_loss(
    router_logits_list: List[torch.Tensor],
    face_mask: torch.Tensor,
    hand_mask: torch.Tensor,
    face_weight: float = 10.0,
    hand_weight: float = 10.0,
    base_weight: float = 1.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    计算所有 DiT blocks 的 Router 监督 loss
    
    采用手部优先策略:
    - 手部 tokens 优先级最高 -> target [0, 1, 0]
    - 面部 tokens (非手部) 优先级中等 -> target [0, 0, 1]
    - 普通 tokens 优先级最低 -> target [1, 0, 0]
    
    Args:
        router_logits_list: 所有 blocks 的 router logits, 每个 shape (B, N, 3)
        face_mask: 面部 token mask, shape (N,) 或 (B, N)
        hand_mask: 手部 token mask, shape (N,) 或 (B, N)
        face_weight: 面部 tokens 的 loss 权重
        hand_weight: 手部 tokens 的 loss 权重
        base_weight: 普通 tokens 的 loss 权重
        reduction: loss 归约方式 ('mean', 'sum', 'none')
        
    Returns:
        router_loss: 累加所有 blocks 的加权 cross entropy loss
    """
    if not router_logits_list or router_logits_list[0] is None:
        return torch.tensor(0.0, device=face_mask.device)
    
    device = face_mask.device
    
    # 支持 1D 和 2D mask
    if face_mask.dim() == 1:
        N = face_mask.shape[0]
        B = 1
        face_mask = face_mask.unsqueeze(0)  # (1, N)
        hand_mask = hand_mask.unsqueeze(0)  # (1, N)
    else:
        B, N = face_mask.shape
    
    # 创建 router targets (手部优先策略)
    targets = torch.zeros(B, N, 3, device=device)
    
    # 1. 先标记普通区域 [1, 0, 0]
    targets[:, :, 0] = 1.0
    
    # 2. 再标记面部区域 [0, 0, 1]
    targets[face_mask, :] = torch.tensor([0.0, 0.0, 1.0], device=device)
    
    # 3. 最后标记手部区域 [0, 1, 0] - 会覆盖与面部重叠的部分
    targets[hand_mask, :] = torch.tensor([0.0, 1.0, 0.0], device=device)
    
    # 创建权重 (手部优先)
    weights = torch.ones(B, N, device=device) * base_weight
    weights[face_mask & ~hand_mask] = face_weight  # 仅面部
    weights[hand_mask] = hand_weight  # 手部 (包含重叠部分)
    
    # 计算所有 blocks 的累加 loss
    total_loss = 0.0
    num_blocks = 0
    
    for router_logits in router_logits_list:
        if router_logits is None:
            continue
        
        # Cross entropy loss (手动计算以支持 per-token weights)
        # router_logits: (B, N, 3)
        # targets: (B, N, 3)
        log_probs = F.log_softmax(router_logits, dim=-1)  # (B, N, 3)
        
        # Per-token cross entropy: -sum(target * log_prob)
        loss_per_token = -(targets * log_probs).sum(dim=-1)  # (B, N)
        
        # 应用权重
        weighted_loss = loss_per_token * weights  # (B, N)
        
        if reduction == 'mean':
            block_loss = weighted_loss.mean()
        elif reduction == 'sum':
            block_loss = weighted_loss.sum()
        else:  # 'none'
            block_loss = weighted_loss
        
        total_loss = total_loss + block_loss
        num_blocks += 1
    
    # 取所有 blocks 的平均
    if num_blocks > 0:
        total_loss = total_loss / num_blocks
    
    return total_loss


def compute_router_accuracy(
    router_logits_list: List[torch.Tensor],
    face_mask: torch.Tensor,
    hand_mask: torch.Tensor
) -> Tuple[float, float, float]:
    """
    计算 Router 预测的准确率
    
    Args:
        router_logits_list: 所有 blocks 的 router logits
        face_mask: 面部 token mask, shape (N,) 或 (B, N)
        hand_mask: 手部 token mask, shape (N,) 或 (B, N)
        
    Returns:
        (base_acc, hand_acc, face_acc): 三个专家的平均权重
    """
    if not router_logits_list or router_logits_list[0] is None:
        return 0.0, 0.0, 0.0
    
    device = face_mask.device
    
    # 支持 1D 和 2D mask
    if face_mask.dim() == 1:
        N = face_mask.shape[0]
        B = 1
        face_mask = face_mask.unsqueeze(0)  # (1, N)
        hand_mask = hand_mask.unsqueeze(0)  # (1, N)
    else:
        B, N = face_mask.shape
    
    # 累加所有 blocks 的 router weights
    router_weights_sum = None
    num_blocks = 0
    
    for router_logits in router_logits_list:
        if router_logits is None:
            continue
        
        router_weights = F.softmax(router_logits, dim=-1)  # (B, N, 3)
        
        if router_weights_sum is None:
            router_weights_sum = router_weights
        else:
            router_weights_sum = router_weights_sum + router_weights
        
        num_blocks += 1
    
    if num_blocks == 0:
        return 0.0, 0.0, 0.0
    
    # 平均权重
    avg_weights = router_weights_sum / num_blocks  # (B, N, 3)
    w_base, w_hand, w_face = avg_weights.split(1, dim=-1)  # 每个 (B, N, 1)
    
    # 计算不同区域的专家激活强度
    base_mask = ~face_mask & ~hand_mask  # 普通区域
    face_only_mask = face_mask & ~hand_mask  # 仅面部
    hand_mask_all = hand_mask  # 手部 (包含重叠)
    
    # 普通区域的 base expert 平均权重
    base_acc = w_base[base_mask].mean().item() if base_mask.sum() > 0 else 0.0
    
    # 手部区域的 hand expert 平均权重
    hand_acc = w_hand[hand_mask_all].mean().item() if hand_mask_all.sum() > 0 else 0.0
    
    # 面部区域的 face expert 平均权重
    face_acc = w_face[face_only_mask].mean().item() if face_only_mask.sum() > 0 else 0.0
    
    return base_acc, hand_acc, face_acc


class RouterLossCalculator(nn.Module):
    """
    Router Loss 计算器，可以作为模块集成到训练流程中
    """
    
    def __init__(
        self,
        face_weight: float = 10.0,
        hand_weight: float = 10.0,
        base_weight: float = 1.0,
        router_loss_weight: float = 0.05
    ):
        """
        Args:
            face_weight: 面部 tokens 的 loss 权重
            hand_weight: 手部 tokens 的 loss 权重
            base_weight: 普通 tokens 的 loss 权重
            router_loss_weight: Router loss 在总 loss 中的权重 α
        """
        super().__init__()
        self.face_weight = face_weight
        self.hand_weight = hand_weight
        self.base_weight = base_weight
        self.router_loss_weight = router_loss_weight
    
    def forward(
        self,
        router_logits_list: List[torch.Tensor],
        face_mask: torch.Tensor,
        hand_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Router loss
        
        Args:
            router_logits_list: 所有 blocks 的 router logits
            face_mask: 面部 token mask
            hand_mask: 手部 token mask
            
        Returns:
            router_loss: 加权后的 router loss (已乘以 router_loss_weight)
        """
        loss = compute_router_loss(
            router_logits_list,
            face_mask,
            hand_mask,
            face_weight=self.face_weight,
            hand_weight=self.hand_weight,
            base_weight=self.base_weight,
            reduction='mean'
        )
        
        return loss * self.router_loss_weight
