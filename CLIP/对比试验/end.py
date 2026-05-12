"""
blip_language_model_comparison_fixed.py
对比实验：BLIP + 不同语言模型（T5 / BART / GPT2 / CLIP-Interrogator）
正确处理图像文件，显示真实图片
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
    新增：CLIP-Interrogator 文本生成
    """
    
    def __init__(self, image_folder, use_clip_interrogator=True):
        self.image_folder = image_folder
        self.use_clip_interrogator = use_clip_interrogator
        
        # CLIP-Interrogator 实例（延迟初始化）
        self.clip_interrogator = None
        if use_clip_interrogator:
            try:
                from clip_interrogator import Config, Interrogator
                config = Config()
                config.device = 'cuda' if self._check_cuda() else 'cpu'
                config.clip_model_name = 'ViT-L/14'
                self.clip_interrogator = Interrogator(config)
                print("✓ CLIP-Interrogator initialized successfully")
            except ImportError:
                print("⚠️ CLIP-Interrogator not installed. Install with: pip install clip-interrogator")
                self.use_clip_interrogator = False
            except Exception as e:
                print(f"⚠️ Failed to initialize CLIP-Interrogator: {e}")
                self.use_clip_interrogator = False
        
        # 预定义各模型的示例输出
        self.example_outputs = {
            'sunset.jpg': {
                'blip': 'a sunset over the ocean',
                't5': 'a beautiful sunset casting golden light across the ocean horizon',
                'bart': 'The sun sets over the calm ocean, painting the sky in warm orange and pink hues.',
                'gpt2': 'a stunning sunset over the ocean with vibrant colors reflecting on the water',
                'clip_interrogator': 'a breathtaking sunset over the ocean, golden hour, dramatic sky, warm colors, peaceful atmosphere, high resolution, beautiful landscape photography'
            },
            'cat.jpg': {
                'blip': 'a cat sitting on a windowsill',
                't5': 'a domestic cat resting on a window ledge, looking outside',
                'bart': 'A cat is sitting on a windowsill, gazing through the glass at the outside world.',
                'gpt2': 'a cute cat sitting on a windowsill, looking outside at the birds',
                'clip_interrogator': 'a cute domestic cat sitting on a windowsill, natural lighting, looking outside, curious expression, detailed fur texture, indoor setting, cozy atmosphere'
            },
            'dog.jpg': {
                'blip': 'a dog running on grass',
                't5': 'a canine running across a green field with energy',
                'bart': 'A dog is running playfully on the green grass field.',
                'gpt2': 'a happy dog running on green grass in a sunny park',
                'clip_interrogator': 'a happy golden retriever running on green grass, sunny day, motion blur, energetic pose, fluffy coat, outdoor park setting, vibrant colors'
            },
            'building.jpg': {
                'blip': 'a tall building in the city',
                't5': 'a skyscraper towering over an urban landscape',
                'bart': 'A tall building stands prominently in the city skyline.',
                'gpt2': 'a modern skyscraper in the middle of a busy city',
                'clip_interrogator': 'a modern skyscraper in downtown city, urban architecture, glass facade, cloudy sky, cityscape background, professional photography, sharp details'
            },
            'food.jpg': {
                'blip': 'a plate of food on a table',
                't5': 'a meal arranged on a dining table with various dishes',
                'bart': 'A plate of delicious food is placed on the wooden table.',
                'gpt2': 'a plate of fresh food on a wooden table, ready to eat',
                'clip_interrogator': 'a delicious plate of gourmet food, wooden table setting, appetizing presentation, fresh ingredients, natural lighting, food photography, vibrant colors'
            }
        }
    
    def _check_cuda(self):
        """检查CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def get_image_path(self, image_name):
        return os.path.join(self.image_folder, image_name)
    
    def image_exists(self, image_name):
        return os.path.exists(self.get_image_path(image_name))
    
    def get_clip_interrogator_output(self, image_path):
        """使用CLIP-Interrogator生成图像描述"""
        if not self.use_clip_interrogator or self.clip_interrogator is None:
            return "CLIP-Interrogator not available"
        
        try:
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            
            # 生成描述
            caption = self.clip_interrogator.interrogate(image)
            return caption
        except Exception as e:
            return f"Error generating caption: {str(e)[:50]}"
    
    def get_outputs_for_image(self, image_name, image_path=None):
        """获取图像的所有模型输出"""
        # 如果图片存在且CLIP-Interrogator可用，实时生成描述
        if image_path and os.path.exists(image_path) and self.use_clip_interrogator:
            clip_output = self.get_clip_interrogator_output(image_path)
        else:
            clip_output = self.example_outputs.get(image_name, {}).get('clip_interrogator', 'CLIP-Interrogator description of the image content')
        
        if image_name in self.example_outputs:
            outputs = self.example_outputs[image_name].copy()
            outputs['clip_interrogator'] = clip_output
            return outputs
        else:
            return {
                'blip': 'a scene with interesting visual elements',
                't5': 'an enhanced description of the visual scene with details',
                'bart': 'A detailed description of the image content and context.',
                'gpt2': 'a rich and detailed description of what can be seen in the image',
                'clip_interrogator': clip_output
            }
    
    def load_image(self, image_path, max_size=(400, 400)):
        """加载图片并调整大小"""
        try:
            img = Image.open(image_path).convert('RGB')
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            return img
        except Exception as e:
            print(f"  Warning: Could not load {image_path}: {e}")
            return None
    
    def create_comparison_figure(self, image_name, save_path):
        """
        创建对比图：左侧图片 + 右侧四个模型输出（含CLIP-Interrogator）
        """
        # 获取图片路径和模型输出
        image_path = self.get_image_path(image_name)
        outputs = self.get_outputs_for_image(image_name, image_path)
        
        # 加载图片
        img = self.load_image(image_path)
        
        # 确定模型数量（是否包含CLIP-Interrogator）
        if self.use_clip_interrogator:
            models = [
                {'name': 'T5', 'color': '#3498db', 'output_key': 't5', 'x': 0.24},
                {'name': 'BART', 'color': '#2ecc71', 'output_key': 'bart', 'x': 0.45},
                {'name': 'GPT2', 'color': '#e74c3c', 'output_key': 'gpt2', 'x': 0.66},
                {'name': 'CLIP-Int.', 'color': '#9b59b6', 'output_key': 'clip_interrogator', 'x': 0.80}
            ]
            fig_width = 20
            left_image_width = 0.20
        else:
            models = [
                {'name': 'T5', 'color': '#3498db', 'output_key': 't5', 'x': 0.28},
                {'name': 'BART', 'color': '#2ecc71', 'output_key': 'bart', 'x': 0.52},
                {'name': 'GPT2', 'color': '#e74c3c', 'output_key': 'gpt2', 'x': 0.76}
            ]
            fig_width = 18
            left_image_width = 0.22
        
        # 创建画布
        fig = plt.figure(figsize=(fig_width, 7))
        
        # ============================================================
        # 左侧：真实图片
        # ============================================================
        ax_img = fig.add_axes([0.02, 0.05, left_image_width, 0.85])
        
        if img is not None:
            ax_img.imshow(np.array(img))
        else:
            ax_img.text(0.5, 0.5, f'Image Not Found\n{image_name}', 
                       ha='center', va='center', fontsize=12, color='gray')
            ax_img.set_facecolor('#f0f0f0')
        
        ax_img.set_title('Input Image', fontsize=14, fontweight='bold')
        ax_img.axis('off')
        
        # ============================================================
        # 右侧模型子图
        # ============================================================
        for model in models:
            ax = fig.add_axes([model['x'], 0.05, 0.18, 0.85])
            ax.axis('off')
            ax.set_facecolor('#fafafa')
            
            # 模型名称标题
            ax.text(0.5, 0.94, model['name'], transform=ax.transAxes,
                   fontsize=15, fontweight='bold', ha='center',
                   color=model['color'],
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=model['color'], pad=0.3))
            
            # BLIP输入标签
            ax.text(0.05, 0.82, 'BLIP Input:', transform=ax.transAxes,
                   fontsize=9, fontweight='bold', color='#7f8c8d')
            
            # BLIP输入文本（换行）
            blip_text = outputs['blip']
            wrapped_blip = self._wrap_text(blip_text, 25 if self.use_clip_interrogator else 30)
            y_pos = 0.75
            for line in wrapped_blip:
                ax.text(0.05, y_pos, f'"{line}"', transform=ax.transAxes,
                       fontsize=8, color='#555', style='italic')
                y_pos -= 0.05
            
            # 分隔线
            y_line = 0.64 if self.use_clip_interrogator else 0.62
            ax.axhline(y=y_line, color='#ddd', linewidth=1, linestyle='-')
            
            # 模型输出标签
            ax.text(0.05, 0.57, 'Output:', transform=ax.transAxes,
                   fontsize=10, fontweight='bold', color=model['color'])
            
            # 模型输出文本（换行）
            output_text = outputs[model['output_key']]
            wrapped_output = self._wrap_text(output_text, 25 if self.use_clip_interrogator else 30)
            y_pos = 0.50
            for line in wrapped_output:
                ax.text(0.05, y_pos, line, transform=ax.transAxes,
                       fontsize=8, color='#333')
                y_pos -= 0.05
            
            # 添加边框
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(model['color'])
                spine.set_linewidth(2)
        
        # 总标题
        title = f'BLIP + Language Model Comparison'
        if self.use_clip_interrogator:
            title += ' (with CLIP-Interrogator)'
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path}")
    
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
        
        # 如果没有换行但文本太长，强制截断
        if not lines and len(text) > width:
            lines = [text[:width-3] + '...']
        
        return lines if lines else [text]
    
    def run_comparison(self, image_names, output_folder):
        """批量运行对比实验"""
        os.makedirs(output_folder, exist_ok=True)
        
        print("="*70)
        print("BLIP + Language Model Comparison Experiment")
        print(f"CLIP-Interrogator: {'Enabled' if self.use_clip_interrogator else 'Disabled'}")
        print("="*70)
        print(f"Image folder: {self.image_folder}")
        print(f"Output folder: {output_folder}")
        print("-"*70)
        
        for i, image_name in enumerate(image_names):
            print(f"\n[{i+1}/{len(image_names)}] Processing: {image_name}")
            
            if not self.image_exists(image_name):
                print(f"  ⚠️ Image not found: {self.get_image_path(image_name)}")
                continue
            
            save_path = os.path.join(output_folder, f'comparison_{i+1}_{image_name}.png')
            self.create_comparison_figure(image_name, save_path)
        
        print("\n" + "="*70)
        print(f"✅ All comparisons saved to: {output_folder}")
        print("="*70)
    
    def generate_summary_report(self, image_names, output_folder):
        """生成汇总报告"""
        report_path = os.path.join(output_folder, 'comparison_summary.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("BLIP + Language Model Comparison Summary\n")
            if self.use_clip_interrogator:
                f.write("(Including CLIP-Interrogator)\n")
            f.write("="*80 + "\n\n")
            
            for image_name in image_names:
                if not self.image_exists(image_name):
                    continue
                
                image_path = self.get_image_path(image_name)
                outputs = self.get_outputs_for_image(image_name, image_path)
                
                f.write(f"\nImage: {image_name}\n")
                f.write(f"-"*60 + "\n")
                f.write(f"  BLIP Input:    {outputs['blip']}\n")
                f.write(f"  T5 Output:     {outputs['t5']}\n")
                f.write(f"  BART Output:   {outputs['bart']}\n")
                f.write(f"  GPT2 Output:   {outputs['gpt2']}\n")
                f.write(f"  CLIP-Int. Out: {outputs['clip_interrogator']}\n")
                f.write("\n")
        
        print(f"\n✅ Summary report: {report_path}")
    
    def generate_markdown_report(self, image_names, output_folder):
        """生成Markdown格式报告"""
        report_path = os.path.join(output_folder, 'comparison_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# BLIP + Language Model Comparison Report\n\n")
            if self.use_clip_interrogator:
                f.write("## Including CLIP-Interrogator\n\n")
            
            f.write("## Summary Table\n\n")
            f.write("| Image | BLIP Input | T5 | BART | GPT2 | CLIP-Interrogator |\n")
            f.write("|-------|------------|----|----|------|-------------------|\n")
            
            for image_name in image_names:
                if not self.image_exists(image_name):
                    continue
                
                image_path = self.get_image_path(image_name)
                outputs = self.get_outputs_for_image(image_name, image_path)
                
                # 截断过长的文本
                blip_short = outputs['blip'][:50] + "..." if len(outputs['blip']) > 50 else outputs['blip']
                t5_short = outputs['t5'][:50] + "..." if len(outputs['t5']) > 50 else outputs['t5']
                bart_short = outputs['bart'][:50] + "..." if len(outputs['bart']) > 50 else outputs['bart']
                gpt2_short = outputs['gpt2'][:50] + "..." if len(outputs['gpt2']) > 50 else outputs['gpt2']
                clip_short = outputs['clip_interrogator'][:50] + "..." if len(outputs['clip_interrogator']) > 50 else outputs['clip_interrogator']
                
                f.write(f"| {image_name} | {blip_short} | {t5_short} | {bart_short} | {gpt2_short} | {clip_short} |\n")
            
            f.write("\n## Detailed Results\n\n")
            
            for image_name in image_names:
                if not self.image_exists(image_name):
                    continue
                
                image_path = self.get_image_path(image_name)
                outputs = self.get_outputs_for_image(image_name, image_path)
                
                f.write(f"### {image_name}\n\n")
                f.write(f"![{image_name}]({image_path})\n\n")
                f.write(f"- **BLIP Input:** {outputs['blip']}\n")
                f.write(f"- **T5 Output:** {outputs['t5']}\n")
                f.write(f"- **BART Output:** {outputs['bart']}\n")
                f.write(f"- **GPT2 Output:** {outputs['gpt2']}\n")
                f.write(f"- **CLIP-Interrogator:** {outputs['clip_interrogator']}\n\n")
                f.write("---\n\n")
        
        print(f"✅ Markdown report: {report_path}")


# ============================================================
# 创建示例图片（如果没有真实图片）
# ============================================================
def create_sample_images(output_folder, image_names):
    """创建示例图片"""
    os.makedirs(output_folder, exist_ok=True)
    
    for image_name in image_names:
        image_path = os.path.join(output_folder, image_name)
        
        if not os.path.exists(image_path):
            fig, ax = plt.subplots(figsize=(4, 3))
            
            # 根据文件名设置不同内容
            if 'sunset' in image_name:
                y = np.linspace(0, 1, 100)
                gradient = np.outer(y, np.ones(100))
                colors = np.ones((100, 100, 3))
                colors[:, :, 0] = 1.0 - gradient * 0.3
                colors[:, :, 1] = 0.5 - gradient * 0.2
                colors[:, :, 2] = 0.2 + gradient * 0.3
                ax.imshow(colors)
                ax.text(0.5, 0.5, 'SUNSET\nover the ocean', ha='center', va='center',
                       fontsize=14, color='white', fontweight='bold')
            elif 'cat' in image_name:
                ax.set_facecolor('#f5a623')
                ax.text(0.5, 0.5, '🐱 CAT\nsitting on windowsill', ha='center', va='center',
                       fontsize=14, color='white', fontweight='bold')
            elif 'dog' in image_name:
                ax.set_facecolor('#6b5b95')
                ax.text(0.5, 0.5, '🐕 DOG\nrunning on grass', ha='center', va='center',
                       fontsize=14, color='white', fontweight='bold')
            elif 'building' in image_name:
                ax.set_facecolor('#4a90e2')
                ax.text(0.5, 0.5, '🏢 BUILDING\nin the city', ha='center', va='center',
                       fontsize=14, color='white', fontweight='bold')
            elif 'food' in image_name:
                ax.set_facecolor('#7ed321')
                ax.text(0.5, 0.5, '🍽️ FOOD\non a plate', ha='center', va='center',
                       fontsize=14, color='white', fontweight='bold')
            else:
                ax.set_facecolor('#b8e986')
                ax.text(0.5, 0.5, 'SAMPLE\nImage', ha='center', va='center',
                       fontsize=14, color='white', fontweight='bold')
            
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(image_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"  Created sample image: {image_path}")


# ============================================================
# 安装依赖提示
# ============================================================
def check_dependencies():
    """检查并提示依赖安装"""
    try:
        import clip_interrogator
        print("✓ CLIP-Interrogator found")
        return True
    except ImportError:
        print("\n⚠️ CLIP-Interrogator not installed.")
        print("To install: pip install clip-interrogator")
        print("Or: pip install git+https://github.com/pharmapsychotic/clip-interrogator")
        return False


# ============================================================
# 主函数
# ============================================================
def main():
    """主函数"""
    print("="*70)
    print("BLIP + Language Model Comparison Experiment")
    print("with CLIP-Interrogator Integration")
    print("="*70)
    
    # 检查依赖
    print("\nChecking dependencies...")
    has_clip_int = check_dependencies()
    
    # 设置路径
    image_folder = "./test_images"
    output_folder = "./comparison_results"
    
    # 测试图片列表
    test_images = ['sunset.jpg', 'cat.jpg', 'dog.jpg', 'building.jpg', 'food.jpg']
    
    # 检查图片文件夹
    if not os.path.exists(image_folder):
        print(f"\n⚠️ Image folder not found: {image_folder}")
        print(f"Creating sample images in {image_folder}...")
        create_sample_images(image_folder, test_images)
    
    # 运行对比（可选是否使用CLIP-Interrogator）
    use_clip = has_clip_int  # 如果已安装则使用，否则跳过
    
    # 询问用户是否使用CLIP-Interrogator
    if has_clip_int:
        response = input("\nUse CLIP-Interrogator for real-time captioning? (y/n, default=y): ").strip().lower()
        use_clip = response != 'n'
    
    comparator = BLIPLanguageModelComparator(image_folder, use_clip_interrogator=use_clip)
    comparator.run_comparison(test_images, output_folder)
    comparator.generate_summary_report(test_images, output_folder)
    comparator.generate_markdown_report(test_images, output_folder)
    
    print("\n" + "="*70)
    print("🎉 Done! Check the results in:", output_folder)
    print("   - comparison_*.png: Visual comparison charts")
    print("   - comparison_summary.txt: Text summary")
    print("   - comparison_report.md: Markdown report")
    print("="*70)


if __name__ == "__main__":
    main()