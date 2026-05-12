"""
blip_language_model_comparison_fixed.py
对比实验：BLIP + 不同语言模型（T5 / BART / GPT2 / CLIP-Interrogator）
修复 Windows 路径问题
"""

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib参数
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class BLIPLanguageModelComparator:
    """
    BLIP + 不同语言模型的对比实验
    左侧显示真实图片，右侧显示各模型输出
    """
    
    def __init__(self, image_folder):
        # 规范化路径 - 修复 Windows 路径问题
        self.image_folder = os.path.abspath(image_folder)  # 转换为绝对路径
        print(f"📁 Image folder (absolute): {self.image_folder}")
        
        # 初始化 BLIP 模型
        self.blip_model = None
        self.blip_processor = None
        self._init_blip()
        
        # 初始化 CLIP-Interrogator
        self.clip_interrogator = None
        self._init_clip_interrogator()
        
        # 初始化其他语言模型
        self.t5_model = None
        self.t5_tokenizer = None
        self.bart_model = None
        self.bart_tokenizer = None
        self.gpt2_model = None
        self.gpt2_tokenizer = None
        self._init_language_models()
    
    def _init_blip(self):
        """初始化 BLIP 模型"""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch
            
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            if torch.cuda.is_available():
                self.blip_model = self.blip_model.cuda()
                print("✓ BLIP model loaded on GPU")
            else:
                print("✓ BLIP model loaded on CPU")
        except Exception as e:
            print(f"⚠️ Failed to load BLIP model: {e}")
            self.blip_model = None
    
    def _init_clip_interrogator(self):
        """初始化 CLIP-Interrogator"""
        try:
            from clip_interrogator import Config, Interrogator
            import torch
            
            config = Config()
            config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            config.clip_model_name = 'ViT-L/14'
            config.blip_model_name = 'blip-base'
            
            self.clip_interrogator = Interrogator(config)
            print(f"✓ CLIP-Interrogator initialized on {config.device}")
        except ImportError:
            print("⚠️ CLIP-Interrogator not installed. Install with: pip install clip-interrogator")
            self.clip_interrogator = None
        except Exception as e:
            print(f"⚠️ Failed to initialize CLIP-Interrogator: {e}")
            self.clip_interrogator = None
    
    def _init_language_models(self):
        """初始化语言模型"""
        try:
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
            self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
            print("✓ T5 model loaded")
        except Exception as e:
            print(f"⚠️ Failed to load T5: {e}")
        
        try:
            from transformers import BartTokenizer, BartForConditionalGeneration
            self.bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
            self.bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
            print("✓ BART model loaded")
        except Exception as e:
            print(f"⚠️ Failed to load BART: {e}")
        
        try:
            from transformers import GPT2Tokenizer, GPT2LMHeadModel
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
            print("✓ GPT2 model loaded")
        except Exception as e:
            print(f"⚠️ Failed to load GPT2: {e}")
    
    def get_blip_caption(self, image):
        """使用 BLIP 生成图像描述"""
        if self.blip_model is None:
            return "BLIP model not available"
        
        try:
            import torch
            inputs = self.blip_processor(image, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            out = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            return f"BLIP error: {str(e)[:50]}"
    
    def get_clip_interrogator_caption(self, image):
        """使用 CLIP-Interrogator 生成详细描述"""
        if self.clip_interrogator is None:
            return "CLIP-Interrogator not available"
        
        try:
            caption = self.clip_interrogator.interrogate(image)
            return caption
        except Exception as e:
            return f"CLIP-Int error: {str(e)[:50]}"
    
    def enhance_with_t5(self, caption):
        """使用 T5 增强描述"""
        if self.t5_model is None:
            return f"T5 enhanced: {caption}"
        
        try:
            prompt = f"expand the description: {caption}"
            inputs = self.t5_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.t5_model.generate(**inputs, max_length=100, num_beams=4)
            enhanced = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return enhanced if len(enhanced) > len(caption) else f"Enhanced: {caption}"
        except:
            return f"T5: {caption}"
    
    def enhance_with_bart(self, caption):
        """使用 BART 增强描述"""
        if self.bart_model is None:
            return f"BART enhanced: {caption}"
        
        try:
            prompt = f"Paraphrase and expand: {caption}"
            inputs = self.bart_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.bart_model.generate(**inputs, max_length=100, num_beams=4)
            enhanced = self.bart_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return enhanced if len(enhanced) > len(caption) else f"BART: {caption}"
        except:
            return f"BART: {caption}"
    
    def enhance_with_gpt2(self, caption):
        """使用 GPT2 增强描述"""
        if self.gpt2_model is None:
            return f"GPT2 enhanced: {caption}"
        
        try:
            prompt = f"Describe this image in detail: {caption}"
            inputs = self.gpt2_tokenizer(prompt, return_tensors="pt", max_length=50, truncation=True)
            outputs = self.gpt2_model.generate(**inputs, max_length=100, num_beams=4, temperature=0.7)
            enhanced = self.gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
            if enhanced.startswith(prompt):
                enhanced = enhanced[len(prompt):]
            return enhanced.strip() if enhanced.strip() else f"GPT2: {caption}"
        except:
            return f"GPT2: {caption}"
    
    def get_image_path(self, image_name):
        """获取图片完整路径 - 修复路径问题"""
        # 使用 os.path.join 确保跨平台兼容
        return os.path.join(self.image_folder, image_name)
    
    def image_exists(self, image_name):
        """检查图片是否存在"""
        path = self.get_image_path(image_name)
        exists = os.path.exists(path)
        if not exists:
            print(f"   检查路径: {path}")
            print(f"   是否存在: {exists}")
        return exists
    
    def list_images_in_folder(self):
        """列出文件夹中的所有图片"""
        if not os.path.exists(self.image_folder):
            print(f"❌ 文件夹不存在: {self.image_folder}")
            return []
        
        images = [f for f in os.listdir(self.image_folder) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        print(f"📸 找到 {len(images)} 张图片:")
        for img in images:
            print(f"   - {img}")
        return images
    
    def load_image(self, image_path, max_size=(400, 400)):
        """加载图片并调整大小"""
        try:
            img = Image.open(image_path).convert('RGB')
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            return img
        except Exception as e:
            print(f"  Warning: Could not load {image_path}: {e}")
            return None
    
    def create_comparison_figure(self, image_name, image_path, save_path):
        """创建对比图"""
        # 加载图片
        img = self.load_image(image_path)
        if img is None:
            print(f"  ✗ Failed to load image: {image_name}")
            return False
        
        # 生成所有描述
        print(f"  Generating captions for {image_name}...")
        
        blip_caption = self.get_blip_caption(img)
        print(f"    BLIP: {blip_caption[:60]}...")
        
        clip_caption = self.get_clip_interrogator_caption(img)
        print(f"    CLIP-Interrogator: {clip_caption[:60]}...")
        
        t5_caption = self.enhance_with_t5(blip_caption)
        bart_caption = self.enhance_with_bart(blip_caption)
        gpt2_caption = self.enhance_with_gpt2(blip_caption)
        
        outputs = {
            'blip': blip_caption,
            't5': t5_caption,
            'bart': bart_caption,
            'gpt2': gpt2_caption,
            'clip_interrogator': clip_caption
        }
        
        # 创建画布
        fig = plt.figure(figsize=(20, 7))
        
        # 左侧：真实图片
        ax_img = fig.add_axes([0.02, 0.05, 0.20, 0.85])
        ax_img.imshow(np.array(img))
        ax_img.set_title(f'Input Image: {image_name}', fontsize=12, fontweight='bold')
        ax_img.axis('off')
        
        # 右侧四个模型
        models = [
            {'name': 'T5', 'color': '#3498db', 'output_key': 't5', 'x': 0.24},
            {'name': 'BART', 'color': '#2ecc71', 'output_key': 'bart', 'x': 0.45},
            {'name': 'GPT2', 'color': '#e74c3c', 'output_key': 'gpt2', 'x': 0.66},
            {'name': 'CLIP-Int.', 'color': '#9b59b6', 'output_key': 'clip_interrogator', 'x': 0.80}
        ]
        
        for model in models:
            ax = fig.add_axes([model['x'], 0.05, 0.17, 0.85])
            ax.axis('off')
            ax.set_facecolor('#fafafa')
            
            ax.text(0.5, 0.94, model['name'], transform=ax.transAxes,
                   fontsize=14, fontweight='bold', ha='center',
                   color=model['color'],
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=model['color'], pad=0.3))
            
            ax.text(0.05, 0.82, 'BLIP Input:', transform=ax.transAxes,
                   fontsize=9, fontweight='bold', color='#7f8c8d')
            
            wrapped_blip = self._wrap_text(outputs['blip'], 28)
            y_pos = 0.75
            for line in wrapped_blip:
                ax.text(0.05, y_pos, f'"{line}"', transform=ax.transAxes,
                       fontsize=8, color='#555', style='italic')
                y_pos -= 0.05
            
            ax.axhline(y=0.64, color='#ddd', linewidth=1, linestyle='-')
            
            ax.text(0.05, 0.57, 'Output:', transform=ax.transAxes,
                   fontsize=10, fontweight='bold', color=model['color'])
            
            output_text = outputs[model['output_key']]
            wrapped_output = self._wrap_text(output_text, 28)
            y_pos = 0.50
            for line in wrapped_output:
                ax.text(0.05, y_pos, line, transform=ax.transAxes,
                       fontsize=8, color='#333')
                y_pos -= 0.05
            
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(model['color'])
                spine.set_linewidth(2)
        
        plt.suptitle('BLIP + Language Model Comparison (with CLIP-Interrogator)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path}")
        return True
    
    def _wrap_text(self, text, width):
        """文本换行"""
        if not text:
            return [""]
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else [text[:width] + "..."]
    
    def run_comparison(self, image_names, output_folder):
        """批量运行对比实验"""
        os.makedirs(output_folder, exist_ok=True)
        
        print("="*70)
        print("BLIP + Language Model Comparison Experiment")
        print("="*70)
        print(f"Image folder: {self.image_folder}")
        print(f"Output folder: {os.path.abspath(output_folder)}")
        print("-"*70)
        
        # 先列出文件夹中的图片
        available_images = self.list_images_in_folder()
        
        results = []
        for i, image_name in enumerate(image_names):
            print(f"\n[{i+1}/{len(image_names)}] Processing: {image_name}")
            
            image_path = self.get_image_path(image_name)
            print(f"  完整路径: {image_path}")
            
            if not self.image_exists(image_name):
                print(f"  ✗ Image not found!")
                continue
            
            base_name = os.path.splitext(image_name)[0]
            save_filename = f"comparison_{base_name}.png"
            save_path = os.path.join(output_folder, save_filename)
            
            success = self.create_comparison_figure(image_name, image_path, save_path)
            if success:
                results.append(image_name)
        
        print("\n" + "="*70)
        print(f"✅ Successfully processed {len(results)}/{len(image_names)} images")
        print(f"📁 Results saved to: {os.path.abspath(output_folder)}")
        print("="*70)
        
        return results


# ============================================================
# 主函数
# ============================================================
def main():
    """主函数"""
    print("="*70)
    print("BLIP + Language Model Comparison Experiment")
    print("With Real CLIP-Interrogator Integration")
    print("="*70)
    
    # 显示当前工作目录
    current_dir = os.getcwd()
    print(f"\n📂 Current working directory: {current_dir}")
    
    # 设置路径 - 使用绝对路径避免混淆
    image_folder = "test_images"  # 相对路径
    output_folder = "results"
    
    # 转换为绝对路径
    image_folder_abs = os.path.abspath(image_folder)
    output_folder_abs = os.path.abspath(output_folder)
    
    print(f"📁 Image folder (absolute): {image_folder_abs}")
    print(f"📁 Output folder (absolute): {output_folder_abs}")
    
    # 检查图片文件夹
    if not os.path.exists(image_folder_abs):
        print(f"\n❌ Image folder not found: {image_folder_abs}")
        print("Please create the folder and add your images.")
        return
    
    # 列出可用的图片
    available_images = [f for f in os.listdir(image_folder_abs) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    if not available_images:
        print(f"\n❌ No images found in {image_folder_abs}")
        print("Please add some images (jpg, png, etc.) to the folder.")
        return
    
    print(f"\n📸 Found {len(available_images)} images:")
    for img in available_images:
        print(f"   - {img}")
    
    # 使用找到的图片，而不是硬编码的列表
    test_images = available_images
    
    # 运行对比
    comparator = BLIPLanguageModelComparator(image_folder_abs)
    results = comparator.run_comparison(test_images, output_folder_abs)
    
    print("\n" + "="*70)
    print("🎉 Done!")
    print(f"📁 Results saved to: {output_folder_abs}")
    print("="*70)


if __name__ == "__main__":
    main()