"""
MoE (Mixture of Experts) FFN 模块

MoE Softmax 加权策略：
- Shared Expert: 原始 FFN，处理所有 tokens（始终激活）
- Hand Expert: 独立的轻量 FFN，专门处理手部 tokens
- Face Expert: 独立的轻量 FFN，专门处理面部 tokens

结构设计：
- Router 输出 3 维 logits [base, hand, face]
- 使用 softmax 加权平均（soft routing）
- base: 只使用 shared_output（没有对应 expert，权重不使用）
- hand/face: 使用 softmax 权重加权残差

公式: output = shared_expert(x) + w_hand * hand_expert(x) + w_face * face_expert(x)

优势：
1. 残差连接：新增的 expert 相当于残差，不会破坏原始模型性能
2. 独立 FFN：比 LoRA 更有表达能力
3. 软选择：平滑的路由决策，梯度更稳定
4. 轻量设计：expert FFN 维度可以比 shared expert 小
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightFFN(nn.Module):
    """
    轻量级 FFN 模块
    
    与标准 FFN 结构相同，但维度更小，用于专家网络
    结构: Linear(dim, hidden_dim) -> GELU -> Linear(hidden_dim, dim)
    
    Args:
        dim: 输入/输出维度
        hidden_dim: 隐藏层维度 (比标准 FFN 小)
    """
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_dim, dim)
        
        # 初始化：fc2 初始化为零，使初始输出为零（残差连接友好）
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重
        
        fc1 使用标准初始化，fc2 初始化为零
        这样初始时 expert 输出为零，不干扰原始模型
        """
        nn.init.kaiming_uniform_(self.fc1.weight, a=5**0.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入 tensor, shape (B, N, D)
            
        Returns:
            输出 tensor, shape (B, N, D)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MoEFFN(nn.Module):
    """
    Mixture of Experts FFN 模块 (Softmax 加权版本)
    
    参考 DiT-MoE 结构设计：
    - Shared Expert: 原始 FFN，处理所有 tokens（始终激活）
    - Hand Expert: 独立的轻量 FFN，专门处理手部 tokens
    - Face Expert: 独立的轻量 FFN，专门处理面部 tokens
    
    Router 输出 3 维 [base, hand, face]:
    - base: 只使用 shared expert (没有对应 expert，权重不使用)
    - hand/face: 使用 softmax 权重加权残差
    
    公式: output = shared_expert(x) + w_hand * hand_expert(x) + w_face * face_expert(x)
    
    优势：
    1. 残差连接：新增的 expert 相当于残差，不会破坏原始模型性能
    2. 独立 FFN：比 LoRA 更有表达能力
    3. 软选择：平滑的路由决策，梯度更稳定
    4. 轻量设计：expert FFN 维度可以比 shared expert 小
    
    Args:
        original_ffn: 原始的 FFN 模块 (nn.Sequential)，将作为 Shared Expert
        dim: 输入/输出维度
        expert_hidden_dim: 专家 FFN 的隐藏层维度 (默认为 dim，可以设置更小的值)
    """
    
    def __init__(self, original_ffn: nn.Module, dim: int, expert_hidden_dim: int = None):
        super().__init__()
        self.dim = dim
        
        # Shared Expert - 直接复用原始 FFN，确保可以加载预训练权重
        # FFN 结构: [0] Linear(dim, ffn_dim), [1] GELU, [2] Linear(ffn_dim, dim)
        self.ffn_base = original_ffn
        
        # 从 ffn_base 推断 ffn_dim
        self.ffn_dim = self.ffn_base[0].out_features
        
        # 专家 FFN 的隐藏层维度，默认为 dim（比 shared expert 的 ffn_dim 小很多）
        # 例如: dim=5120, ffn_dim=13824, expert_hidden_dim=5120
        if expert_hidden_dim is None:
            expert_hidden_dim = dim
        self.expert_hidden_dim = expert_hidden_dim
        
        # Router 网络 - 预测三个专家的权重 [base, hand, face]
        # 使用 SiLU 激活函数（比 GELU 更平滑，适合路由任务）
        # base: 只使用 shared expert (skip 专家网络)
        # hand/face: 使用 shared expert + 对应专家网络的残差
        self.router = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.LayerNorm(dim // 2),
            nn.Linear(dim // 2, 3)  # 输出 3 维: [base, hand, face]
        )
        
        # Hand Expert - 独立的轻量 FFN
        self.hand_expert = LightweightFFN(dim, expert_hidden_dim)
        
        # Face Expert - 独立的轻量 FFN
        self.face_expert = LightweightFFN(dim, expert_hidden_dim)
        
        # Router logits 缓存 (用于 gradient checkpointing 模式下的训练)
        self._cached_router_logits = None
        self._cache_router_logits = False  # 是否缓存 router logits
        
        # w_face 缓存 (用于 audio face mask，推理/router阶段训练时使用)
        self._cached_w_face = None
        
        # 路由控制参数
        self.routing_timestep_threshold = 0.9  # timestep >= 此值时禁用路由
    
    def enable_router_logits_cache(self, enable: bool = True):
        """
        启用/禁用 router logits 缓存
        
        在训练模式下应该启用，这样即使在 gradient checkpointing 时
        也可以通过 get_cached_router_logits() 获取 router logits
        """
        self._cache_router_logits = enable
        if not enable:
            self._cached_router_logits = None
    
    def get_cached_router_logits(self):
        """获取缓存的 router logits"""
        return self._cached_router_logits
    
    def clear_router_logits_cache(self):
        """清理缓存的 router logits"""
        self._cached_router_logits = None
    
    def get_cached_w_face(self):
        """获取缓存的 w_face 权重，用于 audio face mask
        
        Returns:
            w_face: (B, N, 1) detached tensor，或 None
        """
        return self._cached_w_face
    
    def clear_w_face_cache(self):
        """清理缓存的 w_face"""
        self._cached_w_face = None
    
    def _should_enable_routing(self, timestep: float = None, is_conditional: bool = True) -> bool:
        """
        判断是否应该启用路由
        
        Args:
            timestep: 归一化的 timestep (0-1)，None 表示启用路由
            is_conditional: 是否是 conditional 分支，False 表示 unconditional
            
        Returns:
            bool: 是否启用路由
        """
        # 条件1: unconditional 分支不启用路由
        if not is_conditional:
            return False
        
        # 条件2: timestep >= threshold 时不启用路由（增强前期布局稳定性）
        if timestep is not None and timestep >= self.routing_timestep_threshold:
            return False
        
        return True

    def forward(
        self, 
        x: torch.Tensor, 
        timestep: float = None,
        is_conditional: bool = True,
        return_router_logits: bool = False,
        face_mask: torch.Tensor = None,
        hand_mask: torch.Tensor = None,
    ):
        """
        前向传播
        
        MoE Softmax 加权策略:
        - Shared Expert 始终激活，处理所有 tokens
        - Router 输出 3 维 logits [base, hand, face]
        - 使用 softmax 加权平均（soft routing）
        - base: 只使用 shared_output（没有对应 expert，权重不使用）
        - hand/face: 使用 softmax 权重加权残差
        
        公式: output = shared_expert(x) + w_hand * hand_expert(x) + w_face * face_expert(x)
        
        Args:
            x: 输入 tensor, shape (B, N, D)
            timestep: 归一化的 timestep (0-1)，用于控制是否启用路由
                      当 timestep >= threshold 时禁用路由，增强前期布局稳定性
            is_conditional: 是否是 conditional 分支
                           unconditional 分支禁用路由
            return_router_logits: 是否返回 router logits (训练时需要)
            face_mask: 面部区域 mask, shape (N,) 或 None (保留参数，当前未使用)
            hand_mask: 手部区域 mask, shape (N,) 或 None (保留参数，当前未使用)
            
        Returns:
            如果 return_router_logits=False: 输出 tensor, shape (B, N, D)
            如果 return_router_logits=True: (输出 tensor, router_logits)
                - 输出: shape (B, N, D)
                - router_logits: shape (B, N, 3) 用于计算监督 loss [base, hand, face]
                  如果路由被禁用，router_logits 为 None
        """
        # Shared Expert 始终激活
        shared_output = self.ffn_base(x)  # (B, N, D)
        
        # 判断是否启用路由
        enable_routing = self._should_enable_routing(timestep, is_conditional)
        
        # ========== 统一计算图结构，确保 DDP 训练时所有 GPU 操作序列一致 ==========
        # 无论 enable_routing 是 True 还是 False，都执行完全相同的操作序列
        # 通过 routing_scale (1.0 或 0.0) 来控制结果，而不是通过 if-else 分支
        # 这样可以避免不同 GPU 上因 timestep 不同导致的计算图结构不一致，引发 NCCL 超时
        
        # 1. Router 预测权重
        router_logits = self.router(x.detach())  # (B, N, 3) [base, hand, face]
        
        # 缓存 router_logits (用于 gradient checkpointing 模式)
        if self._cache_router_logits:
            self._cached_router_logits = router_logits
        
        # 2. 计算专家输出（始终计算）
        hand_output = self.hand_expert(x)  # (B, N, D)
        face_output = self.face_expert(x)  # (B, N, D)
        
        # 3. Softmax 加权（替代 Top-1 硬选择）
        # 显式转回 router_logits 的 dtype，避免 F.softmax 内部提升到 float32
        # 导致 gradient checkpointing recomputation 时 dtype 不一致
        router_weights = F.softmax(router_logits, dim=-1).to(router_logits.dtype)  # (B, N, 3)
        # 分离权重：base 没有对应的 expert，只使用 hand 和 face 权重
        _, w_hand, w_face = router_weights.chunk(3, dim=-1)  # 每个 (B, N, 1)
        
        # 缓存 w_face (用于 audio face mask，detach 避免影响梯度)
        self._cached_w_face = w_face.detach()
        
        # 4. 通过 routing_scale 控制是否启用路由（保持计算图结构一致）
        # enable_routing=True 时 routing_scale=1.0，专家网络正常工作
        # enable_routing=False 时 routing_scale=0.0，专家网络输出被屏蔽
        routing_scale = 1.0 if enable_routing else 0.0
        
        # 5. 统一的输出计算（无论 enable_routing 是什么，操作序列完全一致）
        # output = shared_expert(x) + w_hand * hand_expert(x) + w_face * face_expert(x)
        expert_contribution = w_hand * hand_output + w_face * face_output
        output = shared_output + routing_scale * expert_contribution
        
        if return_router_logits:
            return output, router_logits
        return output


class MoEFFNConfig:
    """MoE FFN 配置类"""
    
    def __init__(
        self,
        enable_moe: bool = True,
        expert_hidden_dim: int = None,
        router_loss_weight: float = 0.05,
        face_loss_weight: float = 10.0,
        hand_loss_weight: float = 10.0,
    ):
        """
        Args:
            enable_moe: 是否启用 MoE
            expert_hidden_dim: 专家 FFN 的隐藏层维度 (默认为 dim)
            router_loss_weight: Router loss 权重系数 α
            face_loss_weight: 面部 token 的 loss 权重 λ_face
            hand_loss_weight: 手部 token 的 loss 权重 λ_hand
        """
        self.enable_moe = enable_moe
        self.expert_hidden_dim = expert_hidden_dim
        self.router_loss_weight = router_loss_weight
        self.face_loss_weight = face_loss_weight
        self.hand_loss_weight = hand_loss_weight
