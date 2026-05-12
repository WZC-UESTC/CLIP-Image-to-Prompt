"""
clip_blip_integration.py
将BLIP、CLIP和你的Transformer模型集成
实现三阶段流程：BLIP生成 + CLIP筛选 + Transformer重构
"""

import torch
import open_clip
from PIL import Image
import os
from blip_caption import BLIPCaptionGenerator
import pandas as pd
from tqdm import tqdm


class IntegratedPipeline:
    """
    完整的图像→提示词生成管道
    三阶段：BLIP生成 + CLIP筛选 + Transformer重构
    """
    
    def __init__(self, device='cpu', use_rope=True, blip_local_path=None):
        """
        初始化所有模型
        
        Args:
            device: 运行设备
            use_rope: 是否在Transformer中使用RoPE
            blip_local_path: BLIP模型的本地路径
        """
        print("="*60)
        print("初始化集成管道...")
        print("="*60)
        
        self.device = device
        
        # 1. 加载BLIP（包含风格描述功能）
        print("\n[1/3] 加载BLIP模型...")
        
        if blip_local_path is None:
            blip_local_path = r"C:\Users\13113\Desktop\CLIP\models\blip-image-captioning-base"
        
        if os.path.exists(blip_local_path):
            print(f"📁 使用本地模型: {blip_local_path}")
            self.blip = BLIPCaptionGenerator(device=device, local_model_path=blip_local_path)
        else:
            print(f"⚠️ 本地模型不存在: {blip_local_path}")
            self.blip = BLIPCaptionGenerator(device=device)
        
        # 2. 加载CLIP
        print("\n[2/3] 加载CLIP模型...")
        try:
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', 
                pretrained='laion2b_s34b_b79k',
                device=device
            )
            self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
            self.clip_model.eval()
            print("✅ CLIP模型加载完成")
        except Exception as e:
            print(f"❌ CLIP模型加载失败: {e}")
            raise
        
        # 3. 加载标签库
        self.load_tag_library()
        
        print("\n✅ 所有模型加载完成！")
    
    def load_tag_library(self, tag_file=None):
        """加载标签库"""
        # 扩展标签库，增加风格相关标签
        default_tags = [
            # 物体标签
            "dog", "cat", "car", "tree", "person", "building", "flower", "sky", "water", "mountain",
            # 场景标签
            "beach", "city", "indoor", "outdoor", "forest", "street", "office", "home", "park", "ocean",
            # 氛围标签
            "sunny", "cloudy", "night", "day", "colorful", "dark", "bright", "warm", "cold", "foggy",
            # 风格标签
            "painting", "photograph", "sketch", "cartoon", "realistic", "abstract", "vintage", "modern",
            "minimalist", "colorful", "monochrome", "vibrant", "muted", "pastel", "neon", "natural",
            # 情感标签
            "happy", "sad", "peaceful", "chaotic", "romantic", "mysterious", "dramatic", "calm",
            "energetic", "melancholic", "joyful", "serene", "dynamic", "static"
        ]
        
        if tag_file and os.path.exists(tag_file):
            with open(tag_file, 'r') as f:
                self.tag_list = [line.strip() for line in f.readlines()]
        else:
            self.tag_list = default_tags
        
        print(f"📚 加载标签库: {len(self.tag_list)} 个标签")
    
    def clip_filter_tags(self, image_path, top_k=8):
        """
        用CLIP筛选与图像最匹配的标签（扩充到8个）
        
        Args:
            image_path: 图像路径
            top_k: 返回的最匹配标签数量（默认8）
        
        Returns:
            list of (tag, score)
        """
        image = Image.open(image_path).convert('RGB')
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        batch_size = 50
        all_scores = []
        
        for i in range(0, len(self.tag_list), batch_size):
            batch_tags = self.tag_list[i:i+batch_size]
            text_input = self.clip_tokenizer(batch_tags).to(self.device)
            
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_input)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                scores = (image_features @ text_features.T).squeeze(0)
                all_scores.extend([(tag, score.item()) for tag, score in zip(batch_tags, scores)])
        
        top_tags = sorted(all_scores, key=lambda x: x[1], reverse=True)[:top_k]
        
        return top_tags
    
    def process_image(self, image_path, return_all=False):
        """
        处理单张图像，生成最终提示词（包含风格描述）
        
        Args:
            image_path: 图像路径
            return_all: 是否返回中间结果
        
        Returns:
            最终提示词，或包含中间结果的字典
        """
        print(f"\n🖼️ 处理图像: {os.path.basename(image_path)}")
        
        # 阶段1：BLIP生成完整描述（内容+风格，20个token）
        print("  📝 阶段1: BLIP生成描述...")
        blip_result = self.blip.generate_complete_description(image_path)
        print(f"     内容描述: {blip_result['content']}")
        print(f"     风格描述: {blip_result['style']}")
        
        # 阶段2：CLIP筛选标签（扩充到8个）
        print("  🔖 阶段2: CLIP筛选标签...")
        top_tags = self.clip_filter_tags(image_path, top_k=8)
        
        # 分离内容标签和风格标签
        content_tags = [tag for tag, score in top_tags[:4]]
        style_tags = [tag for tag, score in top_tags[4:]]
        
        tag_text = f"内容: {', '.join(content_tags)}; 风格: {', '.join(style_tags)}"
        print(f"     筛选标签: {tag_text}")
        
        # 组合输入
        combined_input = f"内容: {blip_result['content']} 风格: {blip_result['style']} 标签: {tag_text}"
        print(f"  🔗 组合输入: {combined_input}")
        
        # 阶段3：准备Transformer输入（待实现）
        print("  🔄 阶段3: 准备Transformer输入...")
        
        # 最终提示词
        final_prompt = combined_input
        
        print(f"  ✅ 最终提示词 (token数约: {len(final_prompt.split())})")
        
        if return_all:
            return {
                'image': image_path,
                'content_description': blip_result['content'],
                'style_description': blip_result['style'],
                'top_tags': top_tags,
                'content_tags': content_tags,
                'style_tags': style_tags,
                'combined': combined_input,
                'final_prompt': final_prompt
            }
        return final_prompt
    
    def batch_process(self, image_folder, output_file=None, max_images=10):
        """
        批量处理图像
        """
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        for f in os.listdir(image_folder):
            if any(f.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_folder, f))
        
        image_files = image_files[:max_images]
        print(f"\n📁 找到 {len(image_files)} 张图像，开始批量处理...")
        
        results = []
        for img_path in tqdm(image_files, desc="处理进度"):
            result = self.process_image(img_path, return_all=True)
            results.append(result)
        
        if output_file:
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"\n✅ 结果已保存到: {output_file}")
        
        return results


def test_pipeline():
    """测试集成管道"""
    
    blip_local_path = r"C:\Users\13113\Desktop\CLIP\models\blip-image-captioning-base"
    
    if not os.path.exists(blip_local_path):
        print(f"❌ 本地模型不存在: {blip_local_path}")
        return
    
    print("\n🔧 创建集成管道...")
    pipeline = IntegratedPipeline(use_rope=True, blip_local_path=blip_local_path)
    
    test_image = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "test.jpg")
    if not os.path.exists(test_image):
        print(f"❌ 测试图片不存在: {test_image}")
        return
    
    result = pipeline.process_image(test_image, return_all=True)
    
    print("\n" + "="*60)
    print("最终结果:")
    print(f"📝 内容描述: {result['content_description']}")
    print(f"🎨 风格描述: {result['style_description']}")
    print(f"🏷️ 内容标签: {result['content_tags']}")
    print(f"🎭 风格标签: {result['style_tags']}")
    print(f"✨ 最终提示词: {result['final_prompt']}")
    print("="*60)
    
    return pipeline


if __name__ == "__main__":
    test_pipeline()