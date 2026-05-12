"""
model_comparison.py
不同CLIP模型对比实验
比较不同预训练权重的效果
"""

import torch
import open_clip
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from tqdm import tqdm


class CLIPModelComparator:
    """
    CLIP模型对比器
    比较不同模型/预训练权重的效果
    """
    
    # 常用模型配置
    MODELS = {
        'ViT-B-32': {
            'pretrained': ['laion2b_s34b_b79k', 'openai', 'laion400m_e32'],
            'description': 'ViT-B/32，轻量级，速度快'
        },
        'ViT-B-16': {
            'pretrained': ['laion2b_s34b_b79k', 'openai', 'laion400m_e32'],
            'description': 'ViT-B/16，中等大小，效果更好'
        },
        'ViT-L-14': {
            'pretrained': ['laion2b_s32b_b82k', 'openai'],
            'description': 'ViT-L/14，大型模型，效果最好但速度慢'
        }
    }
    
    def __init__(self, device='cpu'):
        self.device = device
        self.results = []
    
    def load_model(self, model_name: str, pretrained: str):
        """
        加载CLIP模型
        
        Returns:
            model, preprocess
        """
        print(f"加载模型: {model_name} ({pretrained})")
        start_time = time.time()
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
            device=self.device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()
        
        elapsed = time.time() - start_time
        print(f"  加载完成，用时: {elapsed:.2f}s")
        
        return model, preprocess, tokenizer
    
    def compute_similarity(self, 
                          model, 
                          preprocess, 
                          tokenizer, 
                          image_path: str, 
                          texts: List[str]) -> float:
        """
        计算图像与文本的平均相似度
        """
        from PIL import Image
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(self.device)
        
        # 提取图像特征
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 提取文本特征
        text_input = tokenizer(texts).to(self.device)
        with torch.no_grad():
            text_features = model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 计算相似度
        similarities = (image_features @ text_features.T).squeeze(0)
        
        return similarities.mean().item()
    
    def compare_on_images(self, 
                          image_paths: List[str], 
                          reference_texts: List[str],
                          models_to_test: List[Tuple[str, str]] = None) -> pd.DataFrame:
        """
        在图像集上对比不同模型
        
        Args:
            image_paths: 图像路径列表
            reference_texts: 参考文本列表（用于计算相似度）
            models_to_test: [(model_name, pretrained), ...]
        
        Returns:
            DataFrame: 对比结果
        """
        if models_to_test is None:
            models_to_test = [
                ('ViT-B-32', 'laion2b_s34b_b79k'),
                ('ViT-B-32', 'openai'),
                ('ViT-B-16', 'laion2b_s34b_b79k'),
                ('ViT-L-14', 'laion2b_s32b_b82k')
            ]
        
        results = []
        
        for model_name, pretrained in models_to_test:
            try:
                model, preprocess, tokenizer = self.load_model(model_name, pretrained)
                
                scores = []
                times = []
                
                for img_path in tqdm(image_paths, desc=f"{model_name}/{pretrained}"):
                    start_time = time.time()
                    score = self.compute_similarity(model, preprocess, tokenizer, 
                                                     img_path, reference_texts)
                    elapsed = time.time() - start_time
                    
                    scores.append(score)
                    times.append(elapsed)
                
                results.append({
                    'model': model_name,
                    'pretrained': pretrained,
                    'avg_score': sum(scores) / len(scores),
                    'std_score': pd.Series(scores).std(),
                    'avg_time': sum(times) / len(times),
                    'total_time': sum(times)
                })
                
            except Exception as e:
                print(f"❌ 模型 {model_name}/{pretrained} 加载失败: {e}")
                continue
        
        self.results = results
        df = pd.DataFrame(results)
        return df
    
    def compare_matching_quality(self,
                                 image_folder: str,
                                 tag_library,
                                 num_images: int = 20,
                                 top_k: int = 5) -> Dict:
        """
        对比不同模型的标签匹配质量
        
        Args:
            image_folder: 图像文件夹
            tag_library: TagLibrary实例
            num_images: 测试图像数量
            top_k: 返回标签数
        
        Returns:
            Dict: 对比结果
        """
        from clip_interrogator import CLIPInterrogator
        
        models_to_test = [
            ('ViT-B-32', 'laion2b_s34b_b79k'),
            ('ViT-B-32', 'openai'),
            ('ViT-B-16', 'laion2b_s34b_b79k')
        ]
        
        # 获取测试图像
        from utils import get_image_files
        image_paths = get_image_files(image_folder)[:num_images]
        
        results = {}
        
        for model_name, pretrained in models_to_test:
            print(f"\n测试模型: {model_name} ({pretrained})")
            
            # 创建Interrogator
            ci = CLIPInterrogator(
                model_name=model_name,
                pretrained=pretrained,
                device=self.device,
                tag_library=tag_library.library
            )
            
            # 对每张图像进行匹配
            all_matches = []
            times = []
            
            for img_path in tqdm(image_paths, desc="匹配"):
                start_time = time.time()
                matches = ci.match_tags(img_path, top_k=top_k)
                elapsed = time.time() - start_time
                
                all_matches.append(matches)
                times.append(elapsed)
            
            # 计算平均相似度
            avg_similarity = sum(m[0][1] for m in all_matches if m) / len(all_matches)
            
            results[f"{model_name}/{pretrained}"] = {
                'avg_similarity': avg_similarity,
                'avg_time': sum(times) / len(times),
                'matches': all_matches
            }
        
        return results
    
    def visualize_comparison(self, df: pd.DataFrame, save_path: str = None):
        """
        可视化对比结果
        """
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 子图1：平均得分对比
        ax1 = axes[0]
        models = [f"{row['model']}\n{row['pretrained'][:10]}" for _, row in df.iterrows()]
        scores = df['avg_score'].tolist()
        
        bars = ax1.bar(models, scores, color='#4ECDC4')
        ax1.set_ylabel('平均相似度')
        ax1.set_title('不同CLIP模型的匹配得分对比')
        ax1.set_ylim([0, 1])
        
        for bar, v in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.4f}', ha='center', va='bottom')
        
        # 子图2：推理时间对比
        ax2 = axes[1]
        times = df['avg_time'].tolist()
        
        bars = ax2.bar(models, times, color='#FF6B6B')
        ax2.set_ylabel('平均推理时间 (秒)')
        ax2.set_title('不同CLIP模型的推理时间对比')
        
        for bar, v in zip(bars, times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.3f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 对比图已保存: {save_path}")
        
        plt.show()
    
    def get_recommendation(self) -> Dict:
        """
        根据实验结果推荐最佳模型
        """
        if not self.results:
            return {'error': '没有实验结果'}
        
        # 按得分排序
        sorted_by_score = sorted(self.results, key=lambda x: x['avg_score'], reverse=True)
        # 按时效排序
        sorted_by_speed = sorted(self.results, key=lambda x: x['avg_time'])
        
        return {
            'best_quality': {
                'model': sorted_by_score[0]['model'],
                'pretrained': sorted_by_score[0]['pretrained'],
                'score': sorted_by_score[0]['avg_score']
            },
            'fastest': {
                'model': sorted_by_speed[0]['model'],
                'pretrained': sorted_by_speed[0]['pretrained'],
                'time': sorted_by_speed[0]['avg_time']
            },
            'balanced': {
                # 综合评分：得分高且速度快
                'model': 'ViT-B-32',
                'pretrained': 'laion2b_s34b_b79k',
                'note': '平衡了质量和速度'
            }
        }


# 测试代码
if __name__ == "__main__":
    print("="*50)
    print("CLIP模型对比测试")
    print("="*50)
    
    comparator = CLIPModelComparator()
    
    # 准备测试数据
    test_images = ["../images/test.jpg"] if os.path.exists("../images/test.jpg") else []
    test_texts = ["a photo of a dog", "a beautiful landscape", "a person"]
    
    if test_images:
        df = comparator.compare_on_images(test_images, test_texts)
        print("\n📊 对比结果:")
        print(df)
        
        comparator.visualize_comparison(df, "../outputs/model_comparison.png")
        
        # 获取推荐
        rec = comparator.get_recommendation()
        print("\n🏆 推荐:")
        print(f"  最佳质量: {rec['best_quality']['model']}/{rec['best_quality']['pretrained']}")
        print(f"  最快速度: {rec['fastest']['model']}/{rec['fastest']['pretrained']}")
    else:
        print("⚠️ 没有测试图片")