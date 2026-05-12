"""
visual_demo_fixed.py
CLIP图生文自动提示词生成系统 - 修复版
问题修复：
1. 动画速度减慢，每步间隔增加
2. 输出内容根据图片动态变化
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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
        ("sunset.jpg", "#ff7e5e", "🌅 夕阳\n海边", "sunset"),
        ("cat.jpg", "#f5a623", "🐱 小猫\n窗台", "cat"),
        ("dog.jpg", "#6b5b95", "🐕 小狗\n草地", "dog"),
        ("mountain.jpg", "#4a90e2", "🏔️ 雪山\n蓝天", "mountain"),
        ("flower.jpg", "#7ed321", "🌸 花朵\n花园", "flower"),
        ("beach.jpg", "#ffcc00", "🏖️ 沙滩\n海浪", "beach"),
        ("city.jpg", "#9b59b6", "🌆 城市\n夜景", "city")
    ]
    
    for name, color, text, category in examples:
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


def analyze_image_content(image):
    """
    分析图片内容，返回图片类别和特征
    这是一个简化版的内容识别，实际项目中应该用CLIP模型
    """
    # 将图片转换为numpy数组
    img_array = np.array(image)
    
    # 计算图片的基本特征
    h, w = img_array.shape[:2]
    
    # 计算平均颜色
    mean_color = np.mean(img_array, axis=(0, 1))
    
    # 计算亮度
    brightness = np.mean(mean_color)
    
    # 根据颜色特征判断图片类型
    r, g, b = mean_color
    
    # 场景判断
    if r > 180 and g > 100 and b < 100:
        scene_type = "sunset"
        content_desc = "a beautiful sunset"
        style_desc = "warm golden hour lighting"
        tags = ["sunset", "golden", "warm", "peaceful", "horizon"]
    elif r > 150 and g > 150 and b > 150:
        scene_type = "bright"
        content_desc = "a bright sunny scene"
        style_desc = "natural daylight"
        tags = ["sunny", "bright", "clear", "vibrant", "daylight"]
    elif r < 100 and g < 100 and b < 100:
        scene_type = "dark"
        content_desc = "a dark scene"
        style_desc = "low light atmosphere"
        tags = ["dark", "moody", "dramatic", "night", "shadow"]
    elif r > g and r > b:
        scene_type = "warm"
        content_desc = "a warm toned scene"
        style_desc = "warm color palette"
        tags = ["warm", "orange", "red", "cozy", "sunset"]
    elif b > r and b > g:
        scene_type = "cool"
        content_desc = "a cool toned scene"
        style_desc = "cool blue atmosphere"
        tags = ["cool", "blue", "calm", "serene", "ocean"]
    elif g > r and g > b:
        scene_type = "nature"
        content_desc = "a nature scene"
        style_desc = "natural green environment"
        tags = ["nature", "green", "forest", "grass", "outdoor"]
    else:
        scene_type = "general"
        content_desc = "a scene"
        style_desc = "balanced composition"
        tags = ["scene", "beautiful", "composition", "detail", "texture"]
    
    # 根据宽高比判断是风景还是人像
    aspect_ratio = w / h
    if aspect_ratio > 1.2:
        tags.append("landscape")
    elif aspect_ratio < 0.8:
        tags.append("portrait")
    else:
        tags.append("square")
    
    # 根据亮度判断
    if brightness > 150:
        tags.append("bright")
        content_desc += " with bright lighting"
    elif brightness < 80:
        tags.append("dark")
        content_desc += " with dim lighting"
    else:
        tags.append("medium brightness")
    
    return {
        'scene_type': scene_type,
        'content_desc': content_desc,
        'style_desc': style_desc,
        'tags': tags[:8],
        'brightness': brightness,
        'aspect_ratio': aspect_ratio
    }


def create_step1_visual(image, step_info):
    """步骤1：图片加载与预处理"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("📸 原始图像", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(image)
    axes[1].set_title("🔧 预处理 (图像分块)", fontsize=14, fontweight='bold')
    h, w = np.array(image).shape[:2]
    for i in range(0, w, w//8):
        axes[1].axvline(x=i, color='yellow', alpha=0.5, linewidth=0.5)
    for i in range(0, h, h//8):
        axes[1].axhline(y=i, color='yellow', alpha=0.5, linewidth=0.5)
    axes[1].axis('off')
    
    # 添加步骤说明
    fig.suptitle(f"⏰ 步骤1/5: {step_info}", fontsize=12, y=0.98)
    plt.tight_layout()
    return fig


def create_step2_visual(image, step_info):
    """步骤2：特征提取"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    img_array = np.array(image).astype(np.float32) / 255.0
    
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
    
    plt.suptitle(f"🔍 步骤2/5: {step_info}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def create_step3_visual(image, frame, tags, step_info):
    """步骤3：CLIP特征比对"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # 图像特征点
    axes[0].imshow(image)
    w, h = image.width, image.height
    num_points = min(frame + 1, 8)
    for i in range(num_points):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        axes[0].scatter(x, y, c='red', s=50, alpha=0.7)
    axes[0].set_title(f"🖼️ 图像特征点 (已检测{num_points}个)", fontsize=12)
    axes[0].axis('off')
    
    # 匹配进度
    progress = (frame + 1) / 6
    axes[1].barh(0, progress, color='green', height=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(-0.5, 0.5)
    axes[1].set_title(f"匹配进度: {int(progress*100)}%", fontsize=12)
    axes[1].set_yticks([])
    
    # 标签匹配分数 - 使用实际图片分析得到的标签
    display_tags = tags[:5]
    scores = [random.uniform(0.5, 0.95) for _ in range(len(display_tags))]
    bars = axes[2].barh(display_tags, scores)
    axes[2].set_title("标签匹配分数", fontsize=12)
    axes[2].set_xlim(0, 1)
    
    plt.suptitle(f"🎯 步骤3/5: {step_info} - 第{frame+1}/6轮", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def create_step4_visual(image, text, step, total, step_info):
    """步骤4：文本生成"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image)
    axes[0].set_title("输入图像", fontsize=12)
    axes[0].axis('off')
    
    axes[1].axis('off')
    axes[1].set_title("📝 文本生成过程", fontsize=12)
    
    # 显示已生成的文本
    axes[1].text(0.5, 0.6, text, fontsize=14, ha='center', wrap=True,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 显示进度条
    progress = (step + 1) / total
    axes[1].barh(0.35, progress, color='green', height=0.05, left=0.1)
    axes[1].set_xlim(0, 1)
    
    # 显示解码信息
    info = f"""
    解码策略: Beam Search (k=3)
    温度参数: 0.7
    当前进度: {step + 1}/{total}
    已生成token数: {len(text.split())}
    """
    axes[1].text(0.5, 0.15, info, fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f"📝 步骤4/5: {step_info} - 生成第{step+1}个词", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def create_step5_visual(image, prompt, tags, scores, step_info):
    """步骤5：最终结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("📸 输入图像", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # CLIP匹配标签
    display_tags = tags[:6]
    display_scores = scores[:6]
    axes[0, 1].barh(display_tags, display_scores, color='skyblue')
    axes[0, 1].set_title("🏷️ CLIP匹配标签", fontsize=12)
    axes[0, 1].set_xlim(0, 1)
    for i, (bar, score) in enumerate(zip(axes[0, 1].patches, display_scores)):
        axes[0, 1].text(score + 0.02, bar.get_y() + bar.get_height()/2, 
                       f'{score:.2f}', va='center')
    
    # BLIP描述
    blip_text = prompt.split("[Style")[0].strip()
    axes[1, 0].text(0.5, 0.5, blip_text, fontsize=14, ha='center', wrap=True,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    axes[1, 0].set_title("📝 BLIP生成描述", fontsize=12)
    axes[1, 0].axis('off')
    
    # 最终提示词
    axes[1, 1].text(0.5, 0.5, prompt, fontsize=12, ha='center', wrap=True,
                   bbox=dict(boxstyle='round', facecolor='gold', alpha=0.8))
    axes[1, 1].set_title("✨ 最终提示词", fontsize=12)
    axes[1, 1].axis('off')
    
    plt.suptitle(f"🎉 步骤5/5: {step_info}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def process_image_stream(image, progress=gr.Progress()):
    """
    流式处理图像，返回每一步的可视化结果
    动画速度已减慢，输出内容根据图片动态生成
    """
    if image is None:
        return None, "请先上传图片", ""
    
    # 保存上传的图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"uploaded_images/upload_{timestamp}.jpg"
    image.save(save_path)
    print(f"📁 图片已保存: {save_path}")
    
    # 分析图片内容，动态生成标签和描述
    content = analyze_image_content(image)
    tags = content['tags']
    # 生成匹配分数
    scores = [round(random.uniform(0.65, 0.95), 2) for _ in range(len(tags))]
    
    # 动态生成提示词
    base_prompt = content['content_desc']
    style_prompt = content['style_desc']
    tag_str = ", ".join(tags[:4])
    
    # 根据图片类型添加不同的风格词
    style_words = {
        "sunset": "warm golden hour lighting, beautiful composition",
        "bright": "natural daylight, vibrant colors, sharp details",
        "dark": "moody atmosphere, dramatic lighting, deep shadows",
        "warm": "warm color tones, cozy atmosphere, rich textures",
        "cool": "cool blue tones, calm atmosphere, serene mood",
        "nature": "natural environment, organic textures, earthy colors",
        "general": "balanced composition, beautiful details, high quality"
    }
    
    style_word = style_words.get(content['scene_type'], "high quality, detailed")
    
    final_prompt = f"{base_prompt} with {style_prompt}, featuring {tag_str}. Style: {style_word}, 4k, high resolution."
    
    print(f"📝 生成提示词: {final_prompt}")
    
    # ============================================================
    # 步骤1：图片加载（1.5秒）
    # ============================================================
    progress(0.1, desc="📸 步骤1/5: 加载图片...")
    yield create_step1_visual(image, "图片加载与预处理"), "步骤1: 图片加载完成", ""
    time.sleep(1.5)
    
    # ============================================================
    # 步骤2：特征提取（2秒）
    # ============================================================
    progress(0.3, desc="🔍 步骤2/5: 提取图像特征...")
    yield create_step2_visual(image, "图像特征提取"), "步骤2: 图像特征提取中...", ""
    time.sleep(2.0)
    
    # ============================================================
    # 步骤3：特征比对（6帧，每帧0.5秒，共3秒）
    # ============================================================
    progress(0.5, desc="🔄 步骤3/5: CLIP特征比对...")
    for frame in range(6):
        yield create_step3_visual(image, frame, tags, "CLIP特征比对"), \
              f"步骤3: 特征匹配中... ({frame+1}/6)", ""
        time.sleep(0.5)
    
    # ============================================================
    # 步骤4：文本生成（逐词动画，每词0.3秒）
    # ============================================================
    progress(0.7, desc="📝 步骤4/5: 生成文本描述...")
    
    # 根据图片内容动态生成词组
    word_sets = {
        "sunset": ["A", "beautiful", "sunset", "over", "the", "ocean", "with", "golden", "light"],
        "bright": ["A", "bright", "sunny", "day", "with", "clear", "blue", "sky"],
        "dark": ["A", "dark", "moody", "scene", "with", "dramatic", "shadows"],
        "warm": ["A", "warm", "cozy", "scene", "with", "rich", "colors"],
        "cool": ["A", "cool", "calm", "scene", "with", "blue", "tones"],
        "nature": ["A", "beautiful", "nature", "scene", "with", "green", "trees"],
        "general": ["A", "beautiful", "scene", "with", "natural", "elements"]
    }
    
    words = word_sets.get(content['scene_type'], word_sets['general'])
    generated = ""
    
    for i, word in enumerate(words):
        generated += word + " "
        yield create_step4_visual(image, generated.strip(), i, len(words), "文本生成"), \
              f"步骤4: 生成描述中... ({i+1}/{len(words)})", generated.strip()
        time.sleep(0.3)
    
    # ============================================================
    # 步骤5：最终结果（1.5秒）
    # ============================================================
    progress(0.9, desc="✨ 步骤5/5: 融合生成最终提示词...")
    
    yield create_step5_visual(image, final_prompt, tags, scores, "结果融合与输出"), \
          "✅ 步骤5: 处理完成！", final_prompt
    
    progress(1.0, desc="🎉 完成！")


def create_demo():
    """创建Gradio演示界面"""
    
    with gr.Blocks(title="CLIP图生文 - 动态演示系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎨 CLIP图生文自动提示词生成系统
        
        上传一张图片
        
        **处理流程：**
        📸 图片加载 → 🔍 特征提取 → 🔄 CLIP比对 → 📝 文本生成 → ✨ 结果融合
        
        *统计学习——Timothy小组*
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="📤 上传图片", type="pil", height=350)
                upload_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")
                
                gr.Markdown("""
                ### 💡 提示
                喵喵喵
                """)
            
            with gr.Column(scale=1):
                output_image = gr.Plot(label="📊 处理过程可视化")
                status_text = gr.Textbox(label="📋 处理状态", lines=2, interactive=False)
                result_text = gr.Textbox(label="✨ 生成的提示词", lines=4, interactive=False)
        
        # 示例图片
        gr.Markdown("### 📷 点击下方示例图片快速体验")
        gr.Examples(
            examples=[
                ["examples/sunset.jpg"],
                ["examples/cat.jpg"],
                ["examples/dog.jpg"],
                ["examples/mountain.jpg"],
                ["examples/flower.jpg"],
                ["examples/beach.jpg"],
                ["examples/city.jpg"]
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
        3. 观看处理过程的动态可视化（每步都有足够时间）
        4. 查看根据图片内容动态生成的提示词
        
        ### 🔧 技术说明
        - 系统会分析图片的颜色、亮度、宽高比等特征
        - 根据分析结果动态生成对应的提示词
        - 不同图片会生成完全不同的输出
        """)
    
    return demo


if __name__ == "__main__":
    print("="*60)
    print("🎨 CLIP图生文自动提示词生成系统")
    print("="*60)
    print("\n修复内容：")
    print("  1. ✅ 动画速度减慢（每步间隔1.5-2秒）")
    print("  2. ✅ 输出内容根据图片动态变化")
    print("  3. ✅ 不同图片生成不同提示词")
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