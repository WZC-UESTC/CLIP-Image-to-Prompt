"""
clip_score.py
使用CLIP计算图像和文本的相似度
"""

import torch
import open_clip
from PIL import Image
import numpy as np

class CLIPScorer:
    """CLIP-Score计算器"""
    
    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k', device='cpu'):
        """
        初始化CLIP模型
        
        Args:
            model_name: CLIP模型名称
            pretrained: 预训练权重
            device: 运行设备 ('cpu' 或 'cuda')
        """
        self.device = device
        
        print(f"加载CLIP模型: {model_name}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
            device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        print("CLIP模型加载完成")
    
    def image_to_features(self, image_path):
        """提取图像特征"""
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def text_to_features(self, text):
        """提取文本特征"""
        text_input = self.tokenizer([text]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def compute_score(self, image_path, text):
        """
        计算图像和文本的相似度（CLIP-Score）
        
        Returns:
            float: 相似度分数 (0-1之间)
        """
        image_features = self.image_to_features(image_path)
        text_features = self.text_to_features(text)
        
        similarity = (image_features @ text_features.T).item()
        return similarity
    
    def batch_compute(self, image_text_pairs):
        """
        批量计算CLIP-Score
        
        Args:
            image_text_pairs: list of (image_path, text)
        
        Returns:
            list of scores
        """
        scores = []
        for img_path, text in image_text_pairs:
            score = self.compute_score(img_path, text)
            scores.append(score)
        return scores


# 测试代码
if __name__ == "__main__":
    scorer = CLIPScorer()
    
    # 测试
    test_score = scorer.compute_score("test.jpg", "a photo of a dog")
    print(f"CLIP-Score: {test_score:.4f}")