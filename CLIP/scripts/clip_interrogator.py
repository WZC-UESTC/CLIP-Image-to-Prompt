"""
clip_interrogator.py
CLIP Interrogator核心实现
功能：图像→标签匹配的最佳文本选择算法
"""

import torch
import open_clip
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import time
import os


class CLIPInterrogator:
    """
    CLIP Interrogator核心类
    实现图像到文本标签的最佳匹配算法
    """
    
    def __init__(self, 
                 model_name='ViT-B-32',
                 pretrained='laion2b_s34b_b79k',
                 device='cpu',
                 tag_library=None):
        """
        初始化CLIP Interrogator
        
        Args:
            model_name: CLIP模型名称 ('ViT-B-32', 'ViT-B-16', 'ViT-L-14')
            pretrained: 预训练权重
            device: 运行设备
            tag_library: 标签库字典，格式 {'category': [tags]}
        """
        self.device = device
        self.model_name = model_name
        
        # 加载CLIP模型
        print(f"正在加载CLIP模型: {model_name} ({pretrained})")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
            device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        
        # 加载标签库
        self.tag_library = tag_library or self._default_tag_library()
        self._build_tag_embeddings()  # 预计算标签嵌入
        
        print(f"✅ CLIP Interrogator初始化完成")
        print(f"   模型: {model_name}")
        print(f"   标签库大小: {self._total_tags()}")
    
    def _default_tag_library(self):
        """默认标签库"""
        return {
            'objects': [
                'dog', 'cat', 'bird', 'fish', 'horse', 'cow', 'sheep',
                'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'train', 'airplane',
                'tree', 'flower', 'grass', 'mountain', 'river', 'lake', 'ocean',
                'building', 'house', 'castle', 'bridge', 'tower', 'church',
                'person', 'man', 'woman', 'child', 'baby', 'group'
            ],
            'scenes': [
                'beach', 'forest', 'desert', 'mountain', 'city', 'countryside',
                'indoor', 'outdoor', 'street', 'park', 'garden', 'farm',
                'office', 'classroom', 'kitchen', 'bedroom', 'living room'
            ],
            'styles': [
                'realistic', 'photorealistic', 'cartoon', 'anime', 'sketch',
                'oil painting', 'watercolor', 'digital art', '3d render',
                'vintage', 'modern', 'minimalist', 'colorful', 'dark'
            ],
            'colors': [
                'red', 'blue', 'green', 'yellow', 'black', 'white', 'orange',
                'purple', 'pink', 'brown', 'gray', 'gold', 'silver'
            ],
            'qualities': [
                'beautiful', 'cute', 'ugly', 'scary', 'funny', 'sad', 'happy',
                'bright', 'dark', 'sharp', 'blurry', 'detailed', 'simple'
            ]
        }
    
    def _total_tags(self):
        """计算标签总数"""
        return sum(len(tags) for tags in self.tag_library.values())
    
    def _build_tag_embeddings(self):
        """预计算所有标签的文本嵌入（加速匹配）"""
        print("正在预计算标签嵌入...")
        start_time = time.time()
        
        all_tags = []
        self.tag_category = {}  # 记录每个标签的类别
        
        for category, tags in self.tag_library.items():
            for tag in tags:
                all_tags.append(tag)
                self.tag_category[tag] = category
        
        self.tag_list = all_tags
        
        # 分批计算嵌入（避免显存不足）
        batch_size = 64
        embeddings = []
        
        for i in range(0, len(all_tags), batch_size):
            batch = all_tags[i:i+batch_size]
            text_input = self.tokenizer(batch).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text_input)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                embeddings.append(text_features)
        
        self.tag_embeddings = torch.cat(embeddings, dim=0)
        
        elapsed = time.time() - start_time
        print(f"✅ 标签嵌入预计算完成！用时: {elapsed:.2f}s")
    
    def extract_image_features(self, image_path: str) -> torch.Tensor:
        """
        提取图像特征
        
        Args:
            image_path: 图像路径
        
        Returns:
            torch.Tensor: 归一化的图像特征向量
        """
        # 加载并预处理图像
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def match_tags(self, 
                   image_path: str, 
                   top_k: int = 5,
                   threshold: float = 0.0,
                   return_scores: bool = True) -> List:
        """
        最佳文本匹配算法：计算图像与所有标签的相似度，返回top-k
        
        Args:
            image_path: 图像路径
            top_k: 返回的标签数量
            threshold: 相似度阈值（低于阈值的标签不返回）
            return_scores: 是否返回相似度分数
        
        Returns:
            List: [(tag, score, category), ...] 或 [tag, ...]
        """
        # 1. 提取图像特征
        image_features = self.extract_image_features(image_path)
        
        # 2. 计算与所有标签的相似度
        similarities = (image_features @ self.tag_embeddings.T).squeeze(0)
        
        # 3. 排序获取top-k
        values, indices = torch.topk(similarities, min(top_k, len(self.tag_list)))
        
        # 4. 过滤阈值
        results = []
        for idx, score in zip(indices, values):
            if score.item() < threshold:
                continue
            tag = self.tag_list[idx]
            category = self.tag_category.get(tag, 'unknown')
            
            if return_scores:
                results.append((tag, score.item(), category))
            else:
                results.append(tag)
        
        return results
    
    def match_by_category(self, 
                          image_path: str, 
                          categories: List[str] = None,
                          top_k_per_category: int = 2) -> Dict:
        """
        按类别匹配标签
        
        Args:
            image_path: 图像路径
            categories: 要匹配的类别列表，None表示所有类别
            top_k_per_category: 每个类别返回的标签数
        
        Returns:
            Dict: {category: [(tag, score), ...]}
        """
        results = {}
        
        # 指定要匹配的类别
        target_categories = categories or list(self.tag_library.keys())
        
        for category in target_categories:
            if category not in self.tag_library:
                continue
            
            # 获取该类别的标签和嵌入
            category_tags = self.tag_library[category]
            if not category_tags:
                continue
            
            # 找到该类别的标签索引
            indices = [self.tag_list.index(tag) for tag in category_tags if tag in self.tag_list]
            if not indices:
                continue
            
            # 计算相似度
            image_features = self.extract_image_features(image_path)
            cat_embeddings = self.tag_embeddings[indices]
            similarities = (image_features @ cat_embeddings.T).squeeze(0)
            
            # 排序
            values, idxs = torch.topk(similarities, min(top_k_per_category, len(indices)))
            
            # 收集结果
            category_results = []
            for val, idx in zip(values, idxs):
                tag = category_tags[idx]
                category_results.append((tag, val.item()))
            
            results[category] = category_results
        
        return results
    
    def interrogate(self, 
                    image_path: str,
                    top_k: int = 5,
                    include_blip: bool = False,
                    blip_generator=None) -> Dict:
        """
        完整的图像探询流程
        
        Args:
            image_path: 图像路径
            top_k: 返回的标签数量
            include_blip: 是否包含BLIP生成的描述
            blip_generator: BLIPCaptionGenerator实例
        
        Returns:
            Dict: 包含匹配结果和描述
        """
        result = {
            'image': os.path.basename(image_path),
            'matched_tags': self.match_tags(image_path, top_k=top_k),
            'category_tags': self.match_by_category(image_path, top_k_per_category=2)
        }
        
        # 可选：添加BLIP描述
        if include_blip and blip_generator:
            try:
                blip_caption = blip_generator.generate_caption(image_path)
                result['blip_caption'] = blip_caption
            except Exception as e:
                print(f"BLIP生成失败: {e}")
                result['blip_caption'] = None
        
        return result
    
    def batch_interrogate(self,
                          image_folder: str,
                          top_k: int = 5,
                          max_images: int = 50,
                          output_file: str = None) -> List[Dict]:
        """
        批量处理图像
        
        Args:
            image_folder: 图像文件夹路径
            top_k: 返回的标签数量
            max_images: 最大处理数量
            output_file: 输出文件路径
        
        Returns:
            List[Dict]: 所有图像的结果
        """
        from tqdm import tqdm
        import os
        
        # 获取图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for f in os.listdir(image_folder):
            if any(f.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_folder, f))
        
        image_files = image_files[:max_images]
        print(f"找到 {len(image_files)} 张图像，开始处理...")
        
        results = []
        for img_path in tqdm(image_files, desc="CLIP匹配"):
            result = self.interrogate(img_path, top_k=top_k)
            results.append(result)
        
        # 保存结果
        if output_file:
            import json
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✅ 结果已保存: {output_file}")
        
        return results


# 测试代码
if __name__ == "__main__":
    print("="*50)
    print("CLIP Interrogator 测试")
    print("="*50)
    
    # 创建实例
    ci = CLIPInterrogator(device='cpu')
    
    # 测试单张图片（如果有测试图片）
    test_image = "../images/test.jpg"
    if os.path.exists(test_image):
        result = ci.interrogate(test_image, top_k=5)
        
        print("\n📊 匹配结果:")
        print(f"图像: {result['image']}")
        print(f"Top标签:")
        for tag, score, cat in result['matched_tags']:
            print(f"  [{cat}] {tag}: {score:.4f}")
        
        print("\n📊 按类别结果:")
        for cat, tags in result['category_tags'].items():
            if tags:
                print(f"  {cat}: {[t[0] for t in tags]}")
    else:
        print(f"⚠️ 测试图片不存在: {test_image}")
        print("请先在 images 文件夹中放入测试图片")