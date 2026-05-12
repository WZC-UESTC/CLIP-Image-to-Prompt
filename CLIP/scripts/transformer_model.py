"""
transformer_model.py
用于提示词语义重构的Transformer模型
支持是否使用RoPE的配置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from rope_module import RotaryPositionalEncoding

class SemanticReconstructor(nn.Module):
    """
    语义重构Transformer
    输入：BLIP描述 + CLIP标签的组合
    输出：优化后的提示词
    """
    
    def __init__(
        self, 
        vocab_size=30000,      # 词汇表大小
        d_model=512,           # 模型维度
        nhead=8,               # 注意力头数
        num_layers=3,          # Transformer层数
        max_len=128,           # 最大序列长度
        use_rope=True          # 是否使用RoPE
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_rope = use_rope
        self.max_len = max_len
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 位置编码（如果不用RoPE，就用传统的位置编码）
        if not use_rope:
            self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # RoPE模块（如果启用）
        if use_rope:
            # 注意：d_model必须是偶数，且能被nhead整除
            assert d_model % 2 == 0, "d_model必须是偶数才能使用RoPE"
            self.rope = RotaryPositionalEncoding(
                dim=d_model // nhead,  # 每个头的维度
                max_seq_len=max_len
            )
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: 输入token IDs [batch, seq_len]
            attention_mask: 注意力掩码 [batch, seq_len]
        
        Returns:
            logits: 输出logits [batch, seq_len, vocab_size]
        """
        batch, seq_len = x.shape
        
        # 1. 词嵌入
        x = self.embedding(x)  # [batch, seq_len, d_model]
        
        # 2. 添加位置信息
        if self.use_rope:
            # RoPE是在注意力计算时应用的，这里不需要显式添加位置嵌入
            # 但为了保持相同的输入维度，x保持不变
            pass
        else:
            # 传统位置编码
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            pos_embed = self.pos_embedding(positions)  # [1, seq_len, d_model]
            x = x + pos_embed
        
        # 3. 准备自定义注意力层（因为标准Transformer不直接支持RoPE）
        # 这里为了简化，我们假设标准Transformer会使用自定义的注意力
        # 在实际应用中，你可能需要自定义TransformerEncoderLayer
        
        # 简化版：直接使用标准Transformer（不应用RoPE的旋转效果）
        # 在实际项目中，你需要修改注意力机制来应用RoPE
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        # 4. 输出投影
        logits = self.output_proj(x)  # [batch, seq_len, vocab_size]
        
        return logits
    
    def forward_with_rope_attention(self, x, attention_mask=None):
        """
        带RoPE的自定义注意力前向传播
        这展示了如何在注意力计算中应用RoPE
        """
        batch, seq_len = x.shape
        
        # 词嵌入
        x = self.embedding(x)
        
        # 这里应该实现自定义的多头注意力
        # 由于代码较长，这里只给出核心思路：
        """
        # 1. 线性投影得到q, k, v
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        
        # 2. 应用RoPE
        if self.use_rope:
            q, k = self.rope(q, k)
        
        # 3. 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 4. 应用注意力掩码
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # 5. softmax和加权求和
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        # 6. 输出投影
        output = output.view(batch, seq_len, -1)
        """
        
        # 这里返回简化结果
        return x


class SimplifiedTransformer(nn.Module):
    """
    简化的Transformer，用于快速对比实验
    直接展示有无RoPE的效果差异
    """
    
    def __init__(self, d_model=64, nhead=4, use_rope=False):
        super().__init__()
        
        self.d_model = d_model
        self.use_rope = use_rope
        self.num_heads = nhead
        self.head_dim = d_model // nhead
        
        # 线性投影层
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # RoPE模块
        if use_rope:
            self.rope = RotaryPositionalEncoding(dim=self.head_dim, max_seq_len=50)
        
        # 输出层（用于模拟生成）
        self.output_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x, return_attention=False):
        """
        x: [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape
        
        # 1. 线性投影并重塑
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        
        # 2. 应用RoPE（如果启用）
        if self.use_rope:
            q, k = self.rope(q, k)
        
        # 3. 计算注意力分数
        # [batch, num_heads, seq_len, seq_len]
        scores = torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        
        # 4. Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 5. 加权求和
        context = torch.matmul(attn_weights, v.transpose(1, 2))
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        
        # 6. 输出投影
        output = self.out_proj(context)
        
        if return_attention:
            return output, attn_weights
        return output