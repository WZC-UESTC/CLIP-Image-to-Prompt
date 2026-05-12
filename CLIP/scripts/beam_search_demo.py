"""
beam_search_demo.py
束搜索（Beam Search）对比实验
对比贪心解码、束搜索在不同beam width下的效果
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time

# 导入你的模块
from blip_caption import BLIPCaptionGenerator
from clip_score import CLIPScorer
from transformer_model import SemanticReconstructor


class BeamSearchDecoder:
    """
    束搜索解码器
    用于对比不同解码策略的效果
    """
    
    def __init__(self, model, tokenizer=None):
        """
        初始化束搜索解码器
        
        Args:
            model: 语言模型
            tokenizer: 分词器
        """
        self.model = model
        self.tokenizer = tokenizer
    
    def greedy_decode(self, input_ids, max_length=50):
        """
        贪心解码：每一步选择概率最大的词
        
        Args:
            input_ids: 输入token IDs [1, seq_len]
            max_length: 最大生成长度
        
        Returns:
            生成的token序列
        """
        self.model.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # 前向传播
                outputs = self.model(generated)
                next_token_logits = outputs[:, -1, :]  # 取最后一个位置的logits
                
                # 选择概率最大的词
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                # 拼接
                generated = torch.cat([generated, next_token], dim=-1)
                
                # 如果生成结束符，停止
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return generated
    
    def beam_search_decode(self, input_ids, beam_width=3, max_length=50, length_penalty=1.0):
        """
        束搜索解码
        
        Args:
            input_ids: 输入token IDs [1, seq_len]
            beam_width: 束宽
            max_length: 最大生成长度
            length_penalty: 长度惩罚系数
        
        Returns:
            生成的token序列
        """
        self.model.eval()
        
        # 初始化beam
        beams = [{
            'sequence': input_ids.clone(),
            'score': 0.0,
            'finished': False
        }]
        
        with torch.no_grad():
            for step in range(max_length):
                new_beams = []
                
                for beam in beams:
                    if beam['finished']:
                        new_beams.append(beam)
                        continue
                    
                    # 前向传播
                    outputs = self.model(beam['sequence'])
                    next_token_logits = outputs[:, -1, :]  # [1, vocab_size]
                    
                    # 转为概率
                    probs = F.softmax(next_token_logits, dim=-1)
                    
                    # 取top-k个候选
                    topk_probs, topk_indices = torch.topk(probs, beam_width, dim=-1)
                    
                    for i in range(beam_width):
                        token_idx = topk_indices[0, i].item()
                        token_prob = topk_probs[0, i].item()
                        
                        # 计算新序列的分数（对数概率累加）
                        new_score = beam['score'] + np.log(token_prob + 1e-10)
                        
                        # 创建新序列
                        new_seq = torch.cat([beam['sequence'], 
                                            torch.tensor([[token_idx]])], dim=-1)
                        
                        new_beam = {
                            'sequence': new_seq,
                            'score': new_score,
                            'finished': (token_idx == self.tokenizer.eos_token_id)
                        }
                        
                        new_beams.append(new_beam)
                
                # 按分数排序，保留top beam_width个
                new_beams.sort(key=lambda x: x['score'] / (len(x['sequence'][0]) ** length_penalty), 
                              reverse=True)
                beams = new_beams[:beam_width]
                
                # 如果所有beam都结束了，提前停止
                if all(b['finished'] for b in beams):
                    break
        
        # 返回分数最高的序列
        best_beam = max(beams, key=lambda x: x['score'])
        return best_beam['sequence']
    
    def contrastive_decode(self, input_ids, beam_width=3, temperature=0.8, top_k=50, top_p=0.9):
        """
        对比解码（对比束搜索和采样方法的组合）
        
        Args:
            input_ids: 输入token IDs
            beam_width: 束宽
            temperature: 温度参数（越高越随机）
            top_k: top-k采样参数
            top_p: top-p (nucleus) 采样参数
        
        Returns:
            生成的token序列
        """
        # 这里实现更复杂的解码策略对比
        # 作为扩展功能，暂时留空
        pass


class BeamSearchExperiment:
    """
    束搜索对比实验
    比较不同解码策略的效果
    """
    
    def __init__(self, device='cpu'):
        """
        初始化实验
        """
        self.device = device
        self.results = []
        
        # 加载BLIP（用于生成初始描述）
        print("加载BLIP模型...")
        self.blip = BLIPCaptionGenerator(device=device)
        
        # 加载CLIP评分器
        print("加载CLIP评分器...")
        self.scorer = CLIPScorer(device=device)
        
        # 加载Transformer模型（用于重构）
        print("加载Transformer模型...")
        self.transformer = SemanticReconstructor(use_rope=True)
        self.transformer.eval()
        
        # 注意：这里需要tokenizer
        # 实际项目中你需要加载真实的分词器
        self.tokenizer = self._get_dummy_tokenizer()
    
    def _get_dummy_tokenizer(self):
        """获取模拟的分词器（实际项目中替换为真实分词器）"""
        class DummyTokenizer:
            def __init__(self):
                self.eos_token_id = 2
                self.pad_token_id = 0
                self.vocab_size = 30000
            
            def decode(self, token_ids):
                # 模拟解码，实际项目中用真实的分词器
                return f"Generated text with {len(token_ids[0])} tokens"
        
        return DummyTokenizer()
    
    def prepare_test_data(self, image_folder="../images", num_images=5):
        """
        准备测试数据
        
        Returns:
            list of (image_path, blip_caption)
        """
        if not os.path.exists(image_folder):
            print(f"❌ 图像文件夹不存在: {image_folder}")
            return []
        
        # 获取图像文件
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        for f in os.listdir(image_folder):
            if any(f.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_folder, f))
        
        image_files = image_files[:min(num_images, len(image_files))]
        
        # 生成BLIP描述
        test_data = []
        for img_path in tqdm(image_files, desc="生成BLIP描述"):
            caption = self.blip.generate_caption(img_path)
            if caption:
                test_data.append((img_path, caption))
        
        print(f"✅ 准备测试数据: {len(test_data)} 条")
        return test_data
    
    def run_comparison(self, test_data, beam_widths=[1, 3, 5]):
        """
        运行对比实验
        
        Args:
            test_data: 测试数据 [(image_path, caption)]
            beam_widths: 要对比的束宽列表（beam_width=1相当于贪心解码）
        """
        print("\n" + "="*70)
        print("束搜索对比实验")
        print("="*70)
        
        all_results = []
        
        for img_path, caption in tqdm(test_data, desc="处理图像"):
            # 将文本转为token IDs（实际项目中需要真实分词）
            # 这里用模拟数据
            input_ids = torch.randint(0, 100, (1, 10))
            
            row = {'image': os.path.basename(img_path), 'blip_caption': caption}
            
            # 对不同束宽进行解码
            for beam_width in beam_widths:
                decoder = BeamSearchDecoder(self.transformer, self.tokenizer)
                
                # 记录时间
                start_time = time.time()
                
                if beam_width == 1:
                    # 贪心解码
                    generated = decoder.greedy_decode(input_ids)
                    strategy = "greedy"
                else:
                    # 束搜索
                    generated = decoder.beam_search_decode(input_ids, beam_width=beam_width)
                    strategy = f"beam{beam_width}"
                
                elapsed = time.time() - start_time
                
                # 解码文本（实际项目中需要真实解码）
                generated_text = self.tokenizer.decode(generated)
                
                # 计算CLIP-Score
                clip_score = self.scorer.compute_score(img_path, generated_text)
                
                # 计算生成长度
                gen_length = generated.size(1) - input_ids.size(1)
                
                # 记录结果
                col_score = f'clip_score_{strategy}'
                col_time = f'time_{strategy}'
                col_length = f'length_{strategy}'
                
                if col_score not in row:
                    row[col_score] = clip_score
                    row[col_time] = elapsed
                    row[col_length] = gen_length
                
                print(f"  {strategy}: score={clip_score:.4f}, time={elapsed:.3f}s, length={gen_length}")
            
            all_results.append(row)
        
        # 转换为DataFrame
        df = pd.DataFrame(all_results)
        
        # 计算平均分
        summary = {}
        for bw in beam_widths:
            col = f'clip_score_{"greedy" if bw==1 else f"beam{bw}"}'
            if col in df.columns:
                summary[col] = df[col].mean()
        
        print("\n📊 平均CLIP-Score:")
        for name, score in summary.items():
            print(f"  {name}: {score:.4f}")
        
        # 找出最佳策略
        if len(summary) > 1:
            best = max(summary, key=summary.get)
            print(f"\n✅ 最佳策略: {best} ({summary[best]:.4f})")
        
        return df
    
    def visualize_results(self, df, beam_widths=[1, 3, 5]):
        """
        可视化对比结果
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 子图1：CLIP-Score对比
        ax1 = axes[0, 0]
        x_labels = []
        scores = []
        colors = []
        
        for bw in beam_widths:
            col = f'clip_score_{"greedy" if bw==1 else f"beam{bw}"}'
            if col in df.columns:
                x_labels.append("贪心解码" if bw==1 else f"束搜索 (k={bw})")
                scores.append(df[col].mean())
                colors.append('#FF6B6B' if bw==1 else '#4ECDC4')
        
        bars = ax1.bar(x_labels, scores, color=colors)
        ax1.set_ylabel('平均CLIP-Score')
        ax1.set_title('不同解码策略的CLIP-Score对比')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, v in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.4f}', ha='center', va='bottom')
        
        # 子图2：推理时间对比
        ax2 = axes[0, 1]
        times = []
        
        for bw in beam_widths:
            col = f'time_{"greedy" if bw==1 else f"beam{bw}"}'
            if col in df.columns:
                times.append(df[col].mean())
        
        bars = ax2.bar(x_labels, times, color=colors)
        ax2.set_ylabel('平均推理时间 (秒)')
        ax2.set_title('不同解码策略的推理时间对比')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, v in zip(bars, times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.3f}s', ha='center', va='bottom')
        
        # 子图3：生成长度对比
        ax3 = axes[1, 0]
        lengths = []
        
        for bw in beam_widths:
            col = f'length_{"greedy" if bw==1 else f"beam{bw}"}'
            if col in df.columns:
                lengths.append(df[col].mean())
        
        bars = ax3.bar(x_labels, lengths, color=colors)
        ax3.set_ylabel('平均生成长度')
        ax3.set_title('不同解码策略的生成长度对比')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, v in zip(bars, lengths):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{v:.1f}', ha='center', va='bottom')
        
        # 子图4：综合评分（CLIP-Score / 时间）
        ax4 = axes[1, 1]
        efficiencies = [s/t for s, t in zip(scores, times)]
        
        bars = ax4.bar(x_labels, efficiencies, color=colors)
        ax4.set_ylabel('效率 (CLIP-Score/秒)')
        ax4.set_title('解码效率对比')
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, v in zip(bars, efficiencies):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图片
        output_path = "../outputs/beam_search_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 对比图已保存到: {output_path}")
        plt.show()
    
    def save_results(self, df, filename="beam_search_results.csv"):
        """保存结果"""
        os.makedirs("../outputs", exist_ok=True)
        output_path = f"../outputs/{filename}"
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ 结果已保存到: {output_path}")


def run_beam_search_experiment():
    """
    运行束搜索对比实验（主函数）
    """
    print("="*70)
    print("束搜索对比实验")
    print("="*70)
    
    # 1. 初始化实验
    experiment = BeamSearchExperiment()
    
    # 2. 准备测试数据
    test_data = experiment.prepare_test_data(num_images=3)
    if not test_data:
        print("❌ 没有测试数据，退出实验")
        return
    
    # 3. 运行对比实验
    beam_widths = [1, 3, 5]  # 1=贪心解码, 3/5=束搜索
    df = experiment.run_comparison(test_data, beam_widths=beam_widths)
    
    # 4. 可视化结果
    experiment.visualize_results(df, beam_widths=beam_widths)
    
    # 5. 保存结果
    experiment.save_results(df)
    
    return df


def quick_test():
    """
    快速测试（使用模拟数据）
    """
    print("="*70)
    print("快速测试模式 - 使用模拟数据")
    print("="*70)
    
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 5
    
    data = []
    for i in range(n_samples):
        row = {
            'image': f'img{i+1}.jpg',
            'blip_caption': f'a sample caption for image {i+1}'
        }
        
        # 模拟不同束宽的结果
        # 通常束搜索能获得更高分数，但耗时更长
        row['clip_score_greedy'] = 0.70 + np.random.random() * 0.15
        row['time_greedy'] = 0.05 + np.random.random() * 0.03
        row['length_greedy'] = 15 + np.random.randint(-3, 4)
        
        row['clip_score_beam3'] = row['clip_score_greedy'] * (1.05 + np.random.random() * 0.08)
        row['time_beam3'] = row['time_greedy'] * (2 + np.random.random())
        row['length_beam3'] = row['length_greedy'] + np.random.randint(2, 6)
        
        row['clip_score_beam5'] = row['clip_score_greedy'] * (1.08 + np.random.random() * 0.10)
        row['time_beam5'] = row['time_greedy'] * (3 + np.random.random())
        row['length_beam5'] = row['length_greedy'] + np.random.randint(4, 9)
        
        # 确保不超过1.0
        row['clip_score_beam3'] = min(row['clip_score_beam3'], 0.98)
        row['clip_score_beam5'] = min(row['clip_score_beam5'], 0.98)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    print("\n📊 模拟数据:")
    print(df[['image', 'clip_score_greedy', 'clip_score_beam3', 'clip_score_beam5']].to_string())
    
    # 计算平均分
    print("\n📊 平均CLIP-Score:")
    for col in ['clip_score_greedy', 'clip_score_beam3', 'clip_score_beam5']:
        print(f"  {col}: {df[col].mean():.4f}")
    
    # 可视化
    experiment = BeamSearchExperiment()
    experiment.visualize_results(df, beam_widths=[1, 3, 5])
    
    # 保存
    experiment.save_results(df, "beam_search_simulated.csv")
    
    return df


if __name__ == "__main__":
    print("请选择运行模式：")
    print("1. 快速测试（模拟数据）")
    print("2. 实际对比（需要完整模型）")
    
    choice = input("请输入选项 (1/2): ").strip()
    
    if choice == '1':
        quick_test()
    else:
        run_beam_search_experiment()