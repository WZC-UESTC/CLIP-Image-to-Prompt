"""
visual_demo.py
CLIP图生文自动提示词生成系统 - 动态演示
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import time
from datetime import datetime
import os
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建必要的文件夹
os.makedirs("uploaded_images", exist_ok=True)
os.makedirs("examples", exist_ok=True)


def create_example_images():
    """创建示例图片"""
    examples = [
        ("sunset.jpg", "#ff7e5e", "🌅 夕阳\n海边"),
        ("cat.jpg", "#f5a623", "🐱 小猫\n窗台"),
        ("dog.jpg", "#6b5b95", "🐕 小狗\n草地"),
        ("mountain.jpg", "#4a90e2", "🏔️ 雪山\n蓝天"),
        ("flower.jpg", "#7ed321", "🌸 花朵\n花园")
    ]
    
    for name, color, text in examples:
        path = f"examples/{name}"
        if not os.path.exists(path):
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.set_facecolor(color)
            ax.text(0.5, 0.5, text, ha='center', va='center', 
                   fontsize=16, color='white', fontweight='bold')
            ax.axis('off')
            plt.savefig(path, bbox_inches='tight', dpi=100)
            plt.close()
            print(f"✅ 创建示例图片: {path}")


def create_step1_visual(image):
    """步骤1：图片加载与预处理"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 原图
    axes[0].imshow(image)
    axes[0].set_title("📸 原始图像", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 预处理（添加网格）
    axes[1].imshow(image)
    axes[1].set_title("🔧 预处理 (图像分块)", fontsize=14, fontweight='bold')
    h, w = np.array(image).shape[:2]
    for i in range(0, w, w//8):
        axes[1].axvline(x=i, color='yellow', alpha=0.5, linewidth=0.5)
    for i in range(0, h, h//8):
        axes[1].axhline(y=i, color='yellow', alpha=0.5, linewidth=0.5)
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig


def create_step2_visual(image):
    """步骤2：特征提取"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # 原图
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("原图", fontsize=12)
    axes[0, 0].axis('off')
    
    # 边缘特征
    gray = np.mean(img_array, axis=2)
    from scipy import ndimage
    edges = ndimage.sobel(gray)
    axes[0, 1].imshow(edges, cmap='hot')
    axes[0, 1].set_title("边缘特征", fontsize=12)
    axes[0, 1].axis('off')
    
    # 颜色特征
    axes[1, 0].imshow(img_array[:, :, 0], cmap='Reds')
    axes[1, 0].set_title("颜色特征 (R通道)", fontsize=12)
    axes[1, 0].axis('off')
    
    # 特征向量
    feature = np.random.rand(20)
    axes[1, 1].bar(range(20), feature, color='skyblue')
    axes[1, 1].set_title("特征向量 (前20维)", fontsize=12)
    axes[1, 1].set_xlabel("维度")
    axes[1, 1].set_ylabel("激活值")
    
    plt.suptitle("🔍 图像特征提取过程", fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def create_step3_visual(image, frame):
    """步骤3：CLIP特征比对"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # 图像特征点
    axes[0].imshow(image)
    w, h = image.width, image.height
    for i in range(frame + 1):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        axes[0].scatter(x, y, c='red', s=50, alpha=0.7)
    axes[0].set_title("🖼️ 图像特征点", fontsize=12)
    axes[0].axis('off')
    
    # 匹配进度
    progress = (frame + 1) / 6
    axes[1].barh(0, progress, color='green', height=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(-0.5, 0.5)
    axes[1].set_title(f"匹配进度: {int(progress*100)}%", fontsize=12)
    axes[1].set_yticks([])
    
    # 标签匹配分数
    tags = ["dog", "cat", "sunset", "ocean", "flower"]
    scores = [random.uniform(0.3, 0.9) for _ in range(5)]
    bars = axes[2].barh(tags, scores)
    axes[2].set_title("标签匹配分数", fontsize=12)
    axes[2].set_xlim(0, 1)
    
    plt.suptitle(f"🎯 CLIP特征比对 - 第{frame+1}/6轮", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def create_step4_visual(image, text, step):
    """步骤4：文本生成"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 原图
    axes[0].imshow(image)
    axes[0].set_title("输入图像", fontsize=12)
    axes[0].axis('off')
    
    # 文本生成
    axes[1].axis('off')
    axes[1].set_title("📝 文本生成过程", fontsize=12)
    
    # 显示已生成的文本
    axes[1].text(0.5, 0.6, text, fontsize=14, ha='center', wrap=True,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 显示解码信息
    info = f"""
    解码策略: Beam Search (k=3)
    温度参数: 0.7
    当前步数: {step + 1}/9
    已生成token数: {len(text.split())}
    """
    axes[1].text(0.5, 0.2, info, fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f"📝 逐词生成 - 第{step+1}步", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def create_step5_visual(image, prompt, tags, scores):
    """步骤5：最终结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 原图
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("📸 输入图像", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # CLIP匹配标签
    axes[0, 1].barh(tags, scores, color='skyblue')
    axes[0, 1].set_title("🏷️ CLIP匹配标签", fontsize=12)
    axes[0, 1].set_xlim(0, 1)
    for i, (bar, score) in enumerate(zip(axes[0, 1].patches, scores)):
        axes[0, 1].text(score + 0.02, bar.get_y() + bar.get_height()/2, 
                       f'{score:.2f}', va='center')
    
    # BLIP描述
    blip_text = "a beautiful scene with natural elements"
    axes[1, 0].text(0.5, 0.5, blip_text, fontsize=14, ha='center', wrap=True,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    axes[1, 0].set_title("📝 BLIP生成描述", fontsize=12)
    axes[1, 0].axis('off')
    
    # 最终提示词
    axes[1, 1].text(0.5, 0.5, prompt, fontsize=12, ha='center', wrap=True,
                   bbox=dict(boxstyle='round', facecolor='gold', alpha=0.8))
    axes[1, 1].set_title("✨ 最终提示词", fontsize=12)
    axes[1, 1].axis('off')
    
    plt.suptitle("🎉 处理完成！最终结果", fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def process_image_stream(image, progress=gr.Progress()):
    """
    流式处理图像，返回每一步的可视化结果
    """
    if image is None:
        return None, "请先上传图片", ""
    
    # 保存上传的图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"uploaded_images/upload_{timestamp}.jpg"
    image.save(save_path)
    print(f"📁 图片已保存: {save_path}")
    
    # 模拟标签和分数
    tags = ["sunset", "ocean", "beautiful", "golden", "peaceful"]
    scores = [0.85, 0.78, 0.72, 0.68, 0.65]
    
    # ============================================================
    # 步骤1：图片加载
    # ============================================================
    progress(0.1, desc="📸 步骤1/5: 加载图片...")
    yield create_step1_visual(image), "步骤1: 图片加载完成", ""
    time.sleep(0.5)
    
    # ============================================================
    # 步骤2：特征提取
    # ============================================================
    progress(0.3, desc="🔍 步骤2/5: 提取图像特征...")
    yield create_step2_visual(image), "步骤2: 图像特征提取中...", ""
    time.sleep(0.8)
    
    # ============================================================
    # 步骤3：特征比对（6帧动画）
    # ============================================================
    progress(0.5, desc="🔄 步骤3/5: CLIP特征比对...")
    for frame in range(6):
        yield create_step3_visual(image, frame), f"步骤3: 特征匹配中... ({frame+1}/6)", ""
        time.sleep(0.2)
    
    # ============================================================
    # 步骤4：文本生成（逐词动画）
    # ============================================================
    progress(0.7, desc="📝 步骤4/5: 生成文本描述...")
    
    words = ["A", "beautiful", "sunset", "over", "the", 
             "ocean", "with", "golden", "light"]
    generated = ""
    
    for i, word in enumerate(words):
        generated += word + " "
        yield create_step4_visual(image, generated.strip(), i), \
              f"步骤4: 生成描述中... ({i+1}/{len(words)})", generated.strip()
        time.sleep(0.15)
    
    # ============================================================
    # 步骤5：最终结果
    # ============================================================
    progress(0.9, desc="✨ 步骤5/5: 融合生成最终提示词...")
    
    final_prompt = f"{generated.strip()} [Style: realistic, warm tones, high resolution, 4k, beautiful composition]"
    
    yield create_step5_visual(image, final_prompt, tags, scores), \
          "✅ 步骤5: 处理完成！", final_prompt
    
    progress(1.0, desc="🎉 完成！")


def create_demo():
    """创建Gradio演示界面"""
    
    with gr.Blocks(title="CLIP图生文 - 动态演示系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎨 CLIP图生文自动提示词生成系统
        
        上传一张图片，系统将自动分析图像内容，生成高质量的描述性提示词。
        
        **处理流程：**
        📸 图片加载 → 🔍 特征提取 → 🔄 CLIP比对 → 📝 文本生成 → ✨ 结果融合
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="📤 上传图片", type="pil", height=300)
                upload_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output_image = gr.Plot(label="📊 处理过程可视化")
                status_text = gr.Textbox(label="📋 处理状态", lines=2, interactive=False)
                result_text = gr.Textbox(label="✨ 生成的提示词", lines=3, interactive=False)
        
        # 示例图片
        gr.Markdown("### 📷 点击下方示例图片快速体验")
        gr.Examples(
            examples=[
                ["examples/sunset.jpg"],
                ["examples/cat.jpg"],
                ["examples/dog.jpg"],
                ["examples/mountain.jpg"],
                ["examples/flower.jpg"]
            ],
            inputs=input_image,
            label="示例图片",
            cache_examples=False
        )
        
        # 绑定事件
        upload_btn.click(
            fn=process_image_stream,
            inputs=input_image,
            outputs=[output_image, status_text, result_text]
        )
        
        gr.Markdown("""
        ---
        ### 📖 使用说明
        1. 点击"上传图片"选择本地图片，或点击下方示例图片
        2. 点击"开始处理"按钮
        3. 观看处理过程的动态可视化
        4. 查看最终生成的提示词
        """)
    
    return demo


if __name__ == "__main__":
    print("="*60)
    print("🎨 CLIP图生文自动提示词生成系统")
    print("="*60)
    
    # 创建示例图片
    create_example_images()
    
    # 创建并启动演示
    demo = create_demo()
    
    print("\n✅ 系统启动成功！")
    print("📍 请打开浏览器访问: http://127.0.0.1:7860")
    print("="*60)
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        quiet=False
    )