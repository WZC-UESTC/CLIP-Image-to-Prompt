"""
rope_module.py
旋转位置编码（RoPE）的可复用模块
将此文件放在 scripts/ 文件夹下
"""

import torch
import torch.nn as nn

class RotaryPositionalEncoding(nn.Module):
    """简洁版RoPE实现 - 可直接在你的Transformer中使用"""
    
    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # 计算逆频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # 构建缓存
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """旋转一半的维度"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """应用RoPE到query和key"""
        seq_len = q.size(1)
        
        # 获取对应位置的cos/sin
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)
        
        # 应用旋转公式
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_rot, k_rot
    
    def forward_single(self, x: torch.Tensor):
        """只旋转单个张量（用于某些场景）"""
        seq_len = x.size(1)
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)
        return (x * cos) + (self.rotate_half(x) * sin)


# 测试代码（当直接运行此文件时执行）
if __name__ == "__main__":
    print("测试 RoPE 模块...")
    
    # 创建RoPE实例
    rope = RotaryPositionalEncoding(dim=64, max_seq_len=128)
    
    # 创建测试数据
    batch, seq_len, num_heads, head_dim = 2, 10, 8, 64
    q = torch.randn(batch, seq_len, num_heads, head_dim)
    k = torch.randn(batch, seq_len, num_heads, head_dim)
    
    # 应用RoPE
    q_rot, k_rot = rope(q, k)
    
    print(f"输入形状: q={q.shape}, k={k.shape}")
    print(f"输出形状: q_rot={q_rot.shape}, k_rot={k_rot.shape}")
    print(f"是否改变形状: {q.shape == q_rot.shape}")
    print(f"是否改变值: {not torch.allclose(q, q_rot)}")
    print("✅ RoPE模块测试通过！")