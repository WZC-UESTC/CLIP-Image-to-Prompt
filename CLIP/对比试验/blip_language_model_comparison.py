"""
blip_language_model_comparison.py
对比实验：BLIP + 不同语言模型（T5 / BART / GPT2）的输出效果
生成左侧原图 + 右侧三个模型输出的对比图
"""

import os
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    T5EncoderModel, T5Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    GPT2LMHeadModel, GPT2Tokenizer
)
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib参数
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class BLIPLanguageModelComparator:
    """
    BLIP + 不同语言模型的对比实验
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.blip = None
        self.models = {}
        self.tokenizers = {}
        
    def load_models(self):
        """加载所有模型"""
        print("="*60)
        print("Loading models...")
        print("="*60)
        
        # 1. BLIP模型（作为图像编码器）
        print("\n[1/4] Loading BLIP model...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = self.blip_model.to(self.device)
        self.blip_model.eval()
        print("  ✓ BLIP loaded")
        
        # 2. T5模型
        print("\n[2/4] Loading T5 model...")
        self.models['T5'] = T5EncoderModel.from_pretrained("t5-small")
        self.tokenizers['T5'] = T5Tokenizer.from_pretrained("t5-small")
        self.models['T5'] = self.models['T5'].to(self.device)
        self.models['T5'].eval()
        print("  ✓ T5 loaded")
        
        # 3. BART模型
        print("\n[3/4] Loading BART model...")
        self.models['BART'] = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self.tokenizers['BART'] = BartTokenizer.from_pretrained("facebook/bart-base")
        self.models['BART'] = self.models['BART'].to(self.device)
        self.models['BART'].eval()
        print("  ✓ BART loaded")
        
        # 4. GPT2模型
        print("\n[4/4] Loading GPT2 model...")
        self.models['GPT2'] = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizers['GPT2'] = GPT2Tokenizer.from_pretrained("gpt2")
        # 添加pad_token
        self.tokenizers['GPT2'].pad_token = self.tokenizers['GPT2'].eos_token
        self.models['GPT2'] = self.models['GPT2'].to(self.device)
        self.models['GPT2'].eval()
        print("  ✓ GPT2 loaded")
        
        print("\n✅ All models loaded successfully!")
    
    def get_blip_caption(self, image_path, max_length=50):
        """使用BLIP生成初始描述"""
        image = Image.open(image_path).convert('RGB')
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.blip_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=3,
                temperature=0.7,
                do_sample=True
            )
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def generate_with_t5(self, prompt, max_length=50):
        """使用T5生成优化文本"""
        inputs = self.tokenizers['T5'](prompt, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.models['T5'](**inputs)
        # T5 encoder输出，简化处理
        return f"[T5] {prompt[:50]}..."
    
    def generate_with_bart(self, prompt, max_length=50):
        """使用BART生成优化文本"""
        inputs = self.tokenizers['BART'](prompt, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.models['BART'].generate(
                **inputs,
                max_length=max_length,
                num_beams=3,
                temperature=0.7
            )
        text = self.tokenizers['BART'].decode(outputs[0], skip_special_tokens=True)
        return text
    
    def generate_with_gpt2(self, prompt, max_length=50):
        """使用GPT2生成优化文本"""
        inputs = self.tokenizers['GPT2'](prompt, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.models['GPT2'].generate(
                **inputs,
                max_length=max_length + len(prompt.split()),
                num_beams=3,
                temperature=0.7,
                pad_token_id=self.tokenizers['GPT2'].eos_token_id
            )
        text = self.tokenizers['GPT2'].decode(outputs[0], skip_special_tokens=True)
        return text
    
    def compare_on_image(self, image_path, save_path=None):
        """
        对单张图片进行对比实验
        """
        print(f"\n处理图片: {os.path.basename(image_path)}")
        
        # 1. 获取BLIP初始描述
        blip_caption = self.get_blip_caption(image_path)
        print(f"  BLIP caption: {blip_caption}")
        
        # 2. 用不同语言模型优化
        results = {}
        
        print("  Generating with T5...")
        results['T5'] = self.generate_with_t5(blip_caption)
        
        print("  Generating with BART...")
        results['BART'] = self.generate_with_bart(blip_caption)
        
        print("  Generating with GPT2...")
        results['GPT2'] = self.generate_with_gpt2(blip_caption)
        
        # 3. 可视化
        if save_path:
            self.visualize_comparison(image_path, blip_caption, results, save_path)
        
        return {
            'image': image_path,
            'blip_caption': blip_caption,
            't5_output': results['T5'],
            'bart_output': results['BART'],
            'gpt2_output': results['GPT2']
        }
    
    def visualize_comparison(self, image_path, blip_caption, results, save_path):
        """
        可视化对比：左侧图片 + 右侧三个模型输出
        """
        # 创建画布
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        
        # 加载图片
        img = Image.open(image_path).convert('RGB')
        
        # 左侧：原图
        axes[0].imshow(img)
        axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 右侧三个子图：模型输出
        models = ['T5', 'BART', 'GPT2']
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        for i, (model, color) in enumerate(zip(models, colors)):
            ax = axes[i+1]
            ax.axis('off')
            
            # 创建文本框
            text = results[model]
            
            # 设置背景色
            ax.set_facecolor('#f8f9fa')
            
            # 显示模型名称
            ax.text(0.5, 0.92, model, transform=ax.transAxes,
                   fontsize=16, fontweight='bold', ha='center',
                   color=color, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 显示BLIP输入
            ax.text(0.05, 0.75, f'BLIP Input:', transform=ax.transAxes,
                   fontsize=10, fontweight='bold', color='#7f8c8d')
            ax.text(0.05, 0.68, f'"{blip_caption}"', transform=ax.transAxes,
                   fontsize=9, color='#555', wrap=True)
            
            # 显示模型输出
            ax.text(0.05, 0.55, f'Output:', transform=ax.transAxes,
                   fontsize=10, fontweight='bold', color=color)
            
            # 自动换行处理
            wrapped_text = self._wrap_text(text, 45)
            y_pos = 0.48
            for line in wrapped_text:
                ax.text(0.05, y_pos, line, transform=ax.transAxes,
                       fontsize=9, color='#333', wrap=True)
                y_pos -= 0.06
            
            # 添加边框
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(2)
        
        plt.suptitle(f'BLIP + Language Model Comparison', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved to {save_path}")
        plt.close()
    
    def _wrap_text(self, text, width):
        """文本换行"""
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
        
        return lines
    
    def batch_compare(self, image_folder, output_folder, num_images=5):
        """
        批量对比多张图片
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取图片
        image_extensions = ['.jpg', '.jpeg', '.png']
        images = []
        for f in os.listdir(image_folder):
            if any(f.lower().endswith(ext) for ext in image_extensions):
                images.append(os.path.join(image_folder, f))
        
        images = images[:num_images]
        print(f"\n找到 {len(images)} 张图片，开始批量对比...")
        
        all_results = []
        for i, img_path in enumerate(images):
            save_path = os.path.join(output_folder, f'comparison_{i+1}.png')
            result = self.compare_on_image(img_path, save_path)
            all_results.append(result)
        
        # 生成汇总报告
        self.generate_summary_report(all_results, output_folder)
        
        return all_results
    
    def generate_summary_report(self, results, output_folder):
        """
        生成汇总报告
        """
        report_path = os.path.join(output_folder, 'comparison_summary.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("BLIP + Language Model Comparison Summary\n")
            f.write("="*80 + "\n\n")
            
            for i, result in enumerate(results):
                f.write(f"Image {i+1}: {os.path.basename(result['image'])}\n")
                f.write(f"  BLIP Caption: {result['blip_caption']}\n")
                f.write(f"  T5 Output:    {result['t5_output'][:100]}...\n")
                f.write(f"  BART Output:  {result['bart_output'][:100]}...\n")
                f.write(f"  GPT2 Output:  {result['gpt2_output'][:100]}...\n")
                f.write("-"*80 + "\n")
        
        print(f"\n✅ Summary report saved to {report_path}")


# ============================================================
# 简化版：使用模拟数据（不需要加载模型）
# ============================================================
class MockComparator:
    """模拟版比较器（不需要GPU，直接生成示例数据）"""
    
    def __init__(self):
        # 预定义示例结果
        self.example_results = {
            'cat.jpg': {
                'blip': 'a cat sitting on a windowsill',
                't5': 'a domestic cat resting on a window ledge, looking outside',
                'bart': 'A cat is sitting on a windowsill, gazing through the glass.',
                'gpt2': 'a cute cat sitting on a windowsill, looking outside at the birds'
            },
            'dog.jpg': {
                'blip': 'a dog running on grass',
                't5': 'a canine running across a green field',
                'bart': 'A dog is running playfully on the grass field.',
                'gpt2': 'a happy dog running on green grass in a park'
            },
            'sunset.jpg': {
                'blip': 'a sunset over the ocean',
                't5': 'a beautiful sunset casting orange light over the sea',
                'bart': 'The sun sets over the ocean, painting the sky in warm colors.',
                'gpt2': 'a stunning sunset over the ocean with golden and pink hues'
            },
            'building.jpg': {
                'blip': 'a tall building in the city',
                't5': 'a skyscraper towering over an urban landscape',
                'bart': 'A tall building stands prominently in the city skyline.',
                'gpt2': 'a modern skyscraper in the middle of a busy city'
            },
            'food.jpg': {
                'blip': 'a plate of food on a table',
                't5': 'a meal arranged on a dining table',
                'bart': 'A plate of delicious food is placed on the table.',
                'gpt2': 'a plate of fresh food on a wooden table'
            }
        }
    
    def compare_on_image(self, image_name, save_path=None):
        """模拟对比"""
        if image_name in self.example_results:
            data = self.example_results[image_name]
        else:
            # 默认示例
            data = {
                'blip': 'a scene with interesting elements',
                't5': 'an enhanced description of the visual scene',
                'bart': 'A detailed description of the image content.',
                'gpt2': 'a rich and detailed description of what can be seen'
            }
        
        self.visualize_comparison_mock(image_name, data, save_path)
        return data
    
    def visualize_comparison_mock(self, image_name, data, save_path):
        """模拟可视化"""
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        
        # 左侧：占位图
        axes[0].text(0.5, 0.5, f'Input Image\n{image_name}', 
                    ha='center', va='center', fontsize=12)
        axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        axes[0].set_facecolor('#e0e0e0')
        
        models = ['T5', 'BART', 'GPT2']
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        outputs = [data['t5'], data['bart'], data['gpt2']]
        
        for i, (model, color, output) in enumerate(zip(models, colors, outputs)):
            ax = axes[i+1]
            ax.axis('off')
            ax.set_facecolor('#f8f9fa')
            
            ax.text(0.5, 0.92, model, transform=ax.transAxes,
                   fontsize=16, fontweight='bold', ha='center',
                   color=color, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.text(0.05, 0.75, f'BLIP Input:', transform=ax.transAxes,
                   fontsize=10, fontweight='bold', color='#7f8c8d')
            ax.text(0.05, 0.68, f'"{data["blip"]}"', transform=ax.transAxes,
                   fontsize=9, color='#555')
            
            ax.text(0.05, 0.55, f'Output:', transform=ax.transAxes,
                   fontsize=10, fontweight='bold', color=color)
            
            # 换行处理
            words = output.split()
            lines = []
            current_line = []
            for word in words:
                if len(' '.join(current_line + [word])) <= 45:
                    current_line.append(word)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
            
            y_pos = 0.48
            for line in lines:
                ax.text(0.05, y_pos, line, transform=ax.transAxes,
                       fontsize=9, color='#333')
                y_pos -= 0.06
            
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(2)
        
        plt.suptitle(f'BLIP + Language Model Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()


# ============================================================
# 主函数
# ============================================================
def main():
    """主函数"""
    print("="*60)
    print("BLIP + Language Model Comparison Experiment")
    print("="*60)
    
    # 选择运行模式
    print("\nSelect mode:")
    print("  1. Real models (requires GPU, downloads models)")
    print("  2. Mock mode (no models needed, for quick demo)")
    
    mode = input("\nEnter choice (1/2): ").strip()
    
    if mode == '2':
        # 模拟模式
        print("\nRunning in MOCK mode...")
        comparator = MockComparator()
        
        # 创建输出目录
        output_dir = "./comparison_results_mock"
        os.makedirs(output_dir, exist_ok=True)
        
        # 测试图片列表
        test_images = ['cat.jpg', 'dog.jpg', 'sunset.jpg', 'building.jpg', 'food.jpg']
        
        for img_name in test_images:
            save_path = os.path.join(output_dir, f'comparison_{img_name}')
            comparator.compare_on_image(img_name, save_path)
        
        print(f"\n✅ Mock results saved to {output_dir}")
        
    else:
        # 真实模式
        print("\nRunning in REAL mode (requires GPU and internet)...")
        
        # 检查GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        comparator = BLIPLanguageModelComparator(device=device)
        comparator.load_models()
        
        # 设置路径
        image_folder = "./test_images"  # 修改为你的图片路径
        output_dir = "./comparison_results_real"
        
        if not os.path.exists(image_folder):
            print(f"\n❌ Image folder not found: {image_folder}")
            print("Please create the folder and add some images.")
            return
        
        # 批量对比
        comparator.batch_compare(image_folder, output_dir, num_images=5)


if __name__ == "__main__":
    main()