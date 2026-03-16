
from transformers import PretrainedConfig


class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-8):
        super().__init__()
        self.dim=dim
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))
    
    def norm(self,x):
        return torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
    
    def forward(self,x):
        return self.weight*self.norm(x)

# rope/yarn
import math
import numpy as np

# ===================== 核心函数：YaRN 基础 =====================
def compute_yarn_d_fit(d_model: int, beta: float) -> float:
    """计算YaRN高频/低频分界值d_fit"""
    if beta == 1.0:
        return d_model / 2
    log_beta = math.log(beta)
    log_d_half = math.log(d_model / 2)
    exponent = log_beta / log_d_half
    d_fit = beta * ((d_model / 2) ** exponent)
    return d_fit

def get_yarn_theta(base_theta: np.ndarray, d_model: int, beta: float) -> np.ndarray:
    """生成YaRN修正后的频率（高频不缩，低频缩放）"""
    d_fit = compute_yarn_d_fit(d_model, beta)
    print(f"d_fit分界值: {d_fit:.2f}")
    
    i = np.arange(len(base_theta))
    yarn_theta = np.where(i < d_fit, base_theta, base_theta / beta)
    return yarn_theta

def compute_yarn_tau(beta: float) -> float:
    """计算YaRN温度系数τ（平滑注意力分布）"""
    tau = math.sqrt((math.log(beta) + 1) / math.log(beta)) if beta > 1 else 1.0
    print(f"YaRN温度系数τ: {tau:.4f}")
    return tau

# ===================== 核心函数：RoPE/YaRN 旋转 Q/K =====================
def rotate_half(x: np.ndarray) -> np.ndarray:
    """RoPE核心：把向量两两分组，后半部分取负（旋转前置操作）"""
    # x.shape = [seq_len, d_model]
    x1 = x[..., ::2]  # 取偶数维度（0,2,4...）
    x2 = x[..., 1::2] # 取奇数维度（1,3,5...）
    return np.concatenate([-x2, x1], axis=-1)

def apply_rope(q: np.ndarray, k: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    对Q/K应用RoPE/YaRN旋转（共用此函数，仅theta不同）
    :param q: 查询向量，shape=[seq_len, d_model]
    :param k: 键向量，shape=[seq_len, d_model]
    :param theta: 频率数组（RoPE原始/YaRN修正），shape=[d_model//2]
    :return: 旋转后的q_rot, k_rot
    """
    seq_len = q.shape[0]
    d_model = q.shape[1]
    
    # 1. 生成位置序列（0,1,2...seq_len-1）
    positions = np.arange(seq_len)[:, None]  # shape=[seq_len, 1]
    
    # 2. 计算每个位置+维度的旋转角度：phi = m * theta_i
    phi = positions * theta[None, :]  # shape=[seq_len, d_model//2]
    
    # 3. 扩展phi到和Q/K同维度（两两分组复用角度）
    phi = np.repeat(phi, 2, axis=-1)  # shape=[seq_len, d_model]
    
    # 4. 计算cos/ sin
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    # 5. 旋转Q/K（RoPE核心公式）
    q_rot = q * cos_phi + rotate_half(q) * sin_phi
    k_rot = k * cos_phi + rotate_half(k) * sin_phi
    
    return q_rot, k_rot

# ===================== 核心函数：计算注意力得分 =====================
def compute_attention_scores(
    q_rot: np.ndarray, 
    k_rot: np.ndarray, 
    d_model: int, 
    tau: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算注意力得分：
    1. 原始得分 = (Q_rot @ K_rot.T) / sqrt(d_model) * τ
    2. 归一化得分 = Softmax(原始得分)
    """
    # 1. 计算QK^T（注意力原始相似度）
    qk_t = np.matmul(q_rot, k_rot.T)  # shape=[seq_len, seq_len]
    
    # 2. 缩放（除以sqrt(d_model)）+ YaRN温度修正
    scale = 1.0 / math.sqrt(d_model)
    raw_scores = qk_t * scale * tau  # YaRN核心：多乘了τ
    
    # 3. Softmax归一化（得到最终注意力权重）
    # 防止数值溢出：先减每行最大值
    raw_scores -= np.max(raw_scores, axis=-1, keepdims=True)
    exp_scores = np.exp(raw_scores)
    norm_scores = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    return raw_scores, norm_scores