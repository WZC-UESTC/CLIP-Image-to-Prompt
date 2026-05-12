"""
BLIP 图像描述生成模块
用于为图像生成初步的文字描述和风格描述
"""

import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import time
import traceback

# 设置镜像（作为备用）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class BLIPCaptionGenerator:
    """BLIP图像描述生成器"""
    
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", device="cpu", local_model_path=None):
        """
        初始化BLIP模型
        
        Args:
            model_name: 模型名称，默认使用base版本（最轻量）
            device: 运行设备，"cpu" 或 "cuda"
            local_model_path: 本地模型路径，如果提供则从本地加载
        """
        print(f"正在加载 BLIP 模型: {model_name}")
        start_time = time.time()
        
        # 确定模型加载路径
        if local_model_path and os.path.exists(local_model_path):
            model_path = local_model_path
            print(f"📁 从本地加载模型: {model_path}")
        else:
            model_path = model_name
            print(f"🌐 从网络加载模型: {model_path}")
        
        # 加载处理器和模型
        try:
            print("  加载处理器...")
            self.processor = BlipProcessor.from_pretrained(model_path)
            print("  ✅ 处理器加载成功")
            
            print("  加载模型...")
            self.model = BlipForConditionalGeneration.from_pretrained(model_path)
            print("  ✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            traceback.print_exc()
            raise
        
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()
        
        load_time = time.time() - start_time
        print(f"✅ BLIP 模型加载完成！用时: {load_time:.2f} 秒")
        print(f"模型参数量: {self.count_parameters():.2f} M")
    
    def count_parameters(self):
        """计算模型参数量"""
        return sum(p.numel() for p in self.model.parameters()) / 1e6
    
    def generate_caption(self, image_path, max_length=20, num_beams=5, min_length=10):
        """
        为单张图像生成描述（扩充token到20）
        
        Args:
            image_path: 图像文件路径
            max_length: 生成描述的最大长度（默认20）
            num_beams: 束搜索的宽度
            min_length: 最小生成长度
            
        Returns:
            生成的描述文本
        """
        try:
            if not os.path.exists(image_path):
                print(f"     ❌ 图像文件不存在: {image_path}")
                return None
            
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    temperature=0.8,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            return str(caption).strip() if caption else ""
            
        except Exception as e:
            print(f"❌ 生成描述失败: {e}")
            return None
    
    def generate_style_description(self, image_path, max_length=15, num_beams=5):
        """
        为图像生成风格描述（艺术风格、色调、氛围等）
        
        Args:
            image_path: 图像文件路径
            max_length: 生成描述的最大长度（默认15）
            num_beams: 束搜索的宽度
            
        Returns:
            风格描述文本
        """
        try:
            if not os.path.exists(image_path):
                print(f"     ❌ 图像文件不存在: {image_path}")
                return None
            
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=5,
                    num_beams=num_beams,
                    temperature=0.9,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=2,
                    do_sample=True,
                    top_p=0.9
                )
            
            style_desc = self.processor.decode(out[0], skip_special_tokens=True)
            
            return str(style_desc).strip() if style_desc else ""
            
        except Exception as e:
            print(f"❌ 生成风格描述失败: {e}")
            return None
    
    def generate_complete_description(self, image_path):
        """
        生成完整描述（内容描述 + 风格描述）
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            包含内容和风格的完整描述字典
        """
        print(f"  📷 分析图像: {os.path.basename(image_path)}")
        
        # 生成内容描述（20个token）
        print("     生成内容描述...")
        content = self.generate_caption(image_path, max_length=20, min_length=10)
        
        # 生成风格描述（15个token）
        print("     生成风格描述...")
        style = self.generate_style_description(image_path, max_length=15)
        
        # 组合完整描述
        full_description = f"{content} [Style: {style}]" if content and style else content or style or ""
        
        return {
            'content': content,
            'style': style,
            'full': full_description
        }
    
    def batch_generate(self, image_folder, output_file=None, max_images=100):
        """
        批量处理文件夹中的图像（包含风格描述）
        
        Args:
            image_folder: 图像文件夹路径
            output_file: 输出文件路径（可选）
            max_images: 最大处理图像数量
            
        Returns:
            图像路径和描述的列表
        """
        try:
            from tqdm import tqdm
            import csv
        except ImportError:
            print("⚠️ tqdm 未安装")
            tqdm = None
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        if not os.path.exists(image_folder):
            print(f"❌ 图像文件夹不存在: {image_folder}")
            return []
        
        for file in os.listdir(image_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_folder, file))
        
        image_files = image_files[:max_images]
        print(f"找到 {len(image_files)} 张图片，开始处理...")
        
        results = []
        iterator = tqdm(image_files, desc="生成描述") if tqdm else image_files
        
        for img_path in iterator:
            desc = self.generate_complete_description(img_path)
            if desc['full']:
                results.append({
                    'image': os.path.basename(img_path),
                    'content_description': desc['content'],
                    'style_description': desc['style'],
                    'full_description': desc['full']
                })
        
        if output_file and results:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['image', 'content_description', 'style_description', 'full_description'])
                writer.writeheader()
                writer.writerows(results)
            print(f"✅ 结果已保存到: {output_file}")
        
        return results


def test_single_image():
    """测试单张图像生成（包含风格描述）"""
    print("="*60)
    print("测试单张图像生成 - 扩充token到20 + 风格描述")
    print("="*60)
    
    local_model_path = r"C:\Users\13113\Desktop\CLIP\models\blip-image-captioning-base"
    
    if os.path.exists(local_model_path):
        print(f"✅ 找到本地模型: {local_model_path}")
        generator = BLIPCaptionGenerator(local_model_path=local_model_path, device="cpu")
    else:
        print(f"❌ 本地模型不存在: {local_model_path}")
        return None
    
    test_image = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "test.jpg")
    
    if not os.path.exists(test_image):
        print(f"❌ 测试图片不存在: {test_image}")
        return None
    
    print(f"\n处理图片: {test_image}")
    result = generator.generate_complete_description(test_image)
    
    print("\n" + "="*60)
    print("生成结果:")
    print(f"📝 内容描述: {result['content']}")
    print(f"🎨 风格描述: {result['style']}")
    print(f"✨ 完整描述: {result['full']}")
    print("="*60)
    
    return result


if __name__ == "__main__":
    test_single_image()