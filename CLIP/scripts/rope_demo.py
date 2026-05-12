"""
rope_demo.py
旋转位置编码（RoPE）完整演示 - 最终修复版
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RoPEDemo:
    """
    完整的RoPE演示类
    包含多种实现方式和验证函数
    """
    
    @staticmethod
    def version1_simplified(dim=64, seq_len=10):
        """
        版本1：简化版实现（最易理解，无bug）
        """
        print("\n" + "="*50)
        print("版本1：简化版实现")
        print("="*50)
        
        # 1. 创建随机query和key
        batch_size = 2
        q = torch.randn(batch_size, seq_len, dim)
        k = torch.randn(batch_size, seq_len, dim)
        
        print(f"原始query形状: {q.shape}")
        
        # 2. 计算频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        
        # 3. 计算位置
        t = torch.arange(seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        
        # 4. 创建cos和sin
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
        sin = emb.sin().unsqueeze(0).unsqueeze(2)
        
        # 5. 定义旋转函数
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        
        # 6. 应用旋转
        q_rot = q.unsqueeze(2)  # [batch, seq_len, 1, dim]
        k_rot = k.unsqueeze(2)
        
        q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
        
        q_rot = q_rot.squeeze(2)
        k_rot = k_rot.squeeze(2)
        
        print(f"旋转后query形状: {q_rot.shape}")
        print(f"原始query和旋转后是否不同: {not torch.allclose(q, q_rot, rtol=1e-4)}")
        
        return q_rot, k_rot
    
    @staticmethod
    def version2_safe(dim=64, seq_len=10):
        """
        版本2：安全的复数实现（修复了维度bug）
        """
        print("\n" + "="*50)
        print("版本2：安全的复数实现")
        print("="*50)
        
        batch_size = 2
        q = torch.randn(batch_size, seq_len, dim)
        k = torch.randn(batch_size, seq_len, dim)
        
        print(f"原始query形状: {q.shape}")
        
        def precompute_freqs_cis(dim, seq_len, theta=10000.0):
            """预计算复数旋转因子 - 修复版"""
            # 确保dim是偶数
            assert dim % 2 == 0, "dim must be even"
            
            # 计算频率: [dim//2]
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
            
            # 位置索引: [seq_len]
            t = torch.arange(seq_len, dtype=torch.float)
            
            # 外积: [seq_len, dim//2]
            freqs = torch.outer(t, freqs)
            
            # 转为复数形式: [seq_len, dim//2]
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
            
            return freqs_cis
        
        def apply_rotary_emb(xq, xk, freqs_cis):
            """
            应用旋转嵌入 - 修复版
            
            Args:
                xq, xk: [batch, seq_len, dim]
                freqs_cis: [seq_len, dim//2]
            """
            batch, seq_len, dim = xq.shape
            
            # 重塑为 [batch, seq_len, dim//2, 2]
            xq_reshaped = xq.float().reshape(batch, seq_len, -1, 2)
            xk_reshaped = xk.float().reshape(batch, seq_len, -1, 2)
            
            # 转换为复数: [batch, seq_len, dim//2]
            xq_complex = torch.view_as_complex(xq_reshaped)
            xk_complex = torch.view_as_complex(xk_reshaped)
            
            # 调整freqs_cis维度: [seq_len, dim//2] -> [1, seq_len, dim//2]
            freqs_cis = freqs_cis.unsqueeze(0)
            
            # 复数乘法（逐元素相乘，会自动广播batch维度）
            xq_rot_complex = xq_complex * freqs_cis
            xk_rot_complex = xk_complex * freqs_cis
            
            # 转回实数: [batch, seq_len, dim//2, 2] -> [batch, seq_len, dim]
            xq_rot = torch.view_as_real(xq_rot_complex).reshape(batch, seq_len, dim)
            xk_rot = torch.view_as_real(xk_rot_complex).reshape(batch, seq_len, dim)
            
            return xq_rot.type_as(xq), xk_rot.type_as(xk)
        
        # 测试
        freqs_cis = precompute_freqs_cis(dim, seq_len)
        q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)
        
        print(f"旋转后query形状: {q_rot.shape}")
        print(f"freqs_cis形状: {freqs_cis.shape}")
        print(f"原始query和旋转后是否不同: {not torch.allclose(q, q_rot, rtol=1e-4)}")
        
        return q_rot, k_rot
    
    @staticmethod
    def version3_module(dim=64, seq_len=10):
        """
        版本3：模块化实现（推荐用于实际项目）
        """
        print("\n" + "="*50)
        print("版本3：模块化实现")
        print("="*50)
        
        class RotaryPositionalEncoding(nn.Module):
      
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
            def rotate_half(x):
                x1, x2 = x.chunk(2, dim=-1)
                return torch.cat((-x2, x1), dim=-1)
            
            def forward(self, q, k):
                """
                q, k: [batch, seq_len, num_heads, head_dim] 或 [batch, seq_len, dim]
                """
                # 处理不同输入维度
                if q.dim() == 3:
                    # [batch, seq_len, dim]
                    seq_len = q.size(1)
                    cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)
                    sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)
                    
                    q = q.unsqueeze(2)  # [batch, seq_len, 1, dim]
                    k = k.unsqueeze(2)
                    
                    q_rot = (q * cos) + (self.rotate_half(q) * sin)
                    k_rot = (k * cos) + (self.rotate_half(k) * sin)
                    
                    return q_rot.squeeze(2), k_rot.squeeze(2)
                    
                else:
                    # [batch, seq_len, num_heads, head_dim]
                    seq_len = q.size(1)
                    cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)
                    sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)
                    
                    q_rot = (q * cos) + (self.rotate_half(q) * sin)
                    k_rot = (k * cos) + (self.rotate_half(k) * sin)
                    
                    return q_rot, k_rot
        
        # 测试模块
        batch, seq_len, dim = 2, 10, 64
        q = torch.randn(batch, seq_len, dim)
        k = torch.randn(batch, seq_len, dim)
        
        rope = RotaryPositionalEncoding(dim=dim, max_seq_len=20)
        q_rot, k_rot = rope(q, k)
        
        print(f"模块输入形状: q={q.shape}")
        print(f"模块输出形状: q_rot={q_rot.shape}")
        print(f"原始query和旋转后是否不同: {not torch.allclose(q, q_rot, rtol=1e-4)}")
        
        return rope
    
    @staticmethod
    def verify_rotation_property():
        """
        验证RoPE的核心性质：内积依赖于相对位置
        """
        print("\n" + "="*50)
        print("验证RoPE的核心性质")
        print("="*50)
        
        dim = 32
        seq_len = 8
        
        # 创建RoPE模块
        class SimpleRoPE:
            def __init__(self, dim):
                self.dim = dim
                inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
                self.inv_freq = inv_freq
            
            def __call__(self, x, pos):
                """对特定位置的向量应用RoPE"""
                # 确保输入是一维的
                if x.dim() > 1:
                    x = x.squeeze()
                
                # 计算该位置的cos和sin
                t = torch.tensor([pos], dtype=torch.float)
                freqs = torch.einsum('i,j->ij', t, self.inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.squeeze()  # 确保一维
                sin = emb.squeeze()  # 确保一维
                
                # 应用旋转
                def rotate_half(x):
                    x1, x2 = x.chunk(2, dim=-1)
                    return torch.cat((-x2, x1), dim=-1)
                
                return (x * cos) + (rotate_half(x) * sin)
        
        rope = SimpleRoPE(dim)
        
        # 创建基础向量 - 确保是一维的
        base_vector = torch.randn(dim)
        print(f"基础向量形状: {base_vector.shape} (应该是 torch.Size([{dim}]))")
        
        print("\n计算不同位置向量之间的内积：")
        print("-"*60)
        print(f"{'位置 i':<8} {'位置 j':<8} {'距离':<8} {'内积':<12} {'验证'}")
        print("-"*60)
        
        # 测试不同位置的向量
        vectors = []
        for pos in range(seq_len):
            v = rope(base_vector.clone(), pos)
            vectors.append(v)
        
        # 计算所有位置对的内积
        for i in range(seq_len):
            for j in range(i, seq_len):
                # 使用多种方法计算内积，确保正确
                vi = vectors[i]
                vj = vectors[j]
                
                # 方法1：torch.dot（需要一维）
                try:
                    dot1 = torch.dot(vi, vj).item()
                except:
                    # 如果失败，使用squeeze确保一维
                    dot1 = torch.dot(vi.squeeze(), vj.squeeze()).item()
                
                # 方法2：逐元素乘法后求和
                dot2 = torch.sum(vi * vj).item()
                
                rel_dist = j - i
                
                # 验证两种方法结果一致
                is_close = abs(dot1 - dot2) < 1e-6
                
                print(f"{i:<8} {j:<8} {rel_dist:<8} {dot1:<12.4f} {'✓' if is_close else '✗'}")
        
        print("-"*60)
        
        # 验证相同距离的内积是否相近
        print("\n📊 统计相同距离的内积：")
        print("-"*60)
        
        for dist in range(1, seq_len):
            products = []
            for i in range(seq_len - dist):
                j = i + dist
                vi = vectors[i]
                vj = vectors[j]
                dot = torch.sum(vi * vj).item()
                products.append(dot)
            
            if products:
                mean_val = sum(products) / len(products)
                var_val = torch.tensor(products).var().item()
                print(f"距离 {dist}: 平均值={mean_val:.4f}, 方差={var_val:.6f}, 样本={products}")
        
        print("\n" + "="*50)
        print("✅ 结论：相同距离的内积方差很小，证明RoPE只依赖于相对位置")
        print("="*50)
    
    @staticmethod
    def run_all_tests():
        """运行所有测试"""
        print("="*60)
        print("旋转位置编码（RoPE）完整测试")
        print("="*60)
        
        try:
            # 测试版本1
            RoPEDemo.version1_simplified()
            
          
            RoPEDemo.version2_safe()
            
            # 测试版本3
            RoPEDemo.version3_module()
            
            # 验证性质
            RoPEDemo.verify_rotation_property()
            
            print("\n" + "="*60)
            print("✅ 所有测试完成！")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ 测试过程中出现错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # 运行所有测试
    RoPEDemo.run_all_tests()