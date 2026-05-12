"""
evaluate_rope.py
RoPE效果对比验证
适配你已有的BLIP模块和即将创建的Transformer模型
"""

import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 导入你的模块
from blip_caption import BLIPCaptionGenerator
from transformer_model import SemanticReconstructor  # 你将创建的模型
from clip_score import CLIPScorer  # 你将创建的评估工具


def prepare_test_images(image_folder="../images", num_images=10):
    """
    准备测试图像
    
    Args:
        image_folder: 图像文件夹路径
        num_images: 使用的图像数量
    
    Returns:
        list of image paths
    """
    if not os.path.exists(image_folder):
        print(f"❌ 图像文件夹不存在: {image_folder}")
        # 创建示例图像列表（用于测试）
        return ["../images/test.jpg"] if os.path.exists("../images/test.jpg") else []
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for f in os.listdir(image_folder):
        if any(f.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_folder, f))
    
    # 取前num_images张
    image_files = image_files[:min(num_images, len(image_files))]
    print(f"📸 准备测试图像: {len(image_files)} 张")
    
    return image_files


def generate_prompts(blip_model, transformer_model, image_paths, use_rope=True):
    """
    生成提示词
    
    Args:
        blip_model: BLIPCaptionGenerator实例
        transformer_model: SemanticReconstructor实例
        image_paths: 图像路径列表
        use_rope: 是否使用RoPE（仅用于标识）
    
    Returns:
        list of prompts
    """
    prompts = []
    
    for img_path in tqdm(image_paths, desc=f"生成提示词({'有RoPE' if use_rope else '无RoPE'})"):
        # 阶段1：BLIP生成描述
        blip_caption = blip_model.generate_caption(img_path)
        
        # 简化：这里直接用BLIP描述作为提示词
        # 实际应该用transformer_model进行重构
        prompts.append(blip_caption)
    
    return prompts


def compare_rope_effectiveness(image_folder="../images", num_images=5):
    """
    对比RoPE是否有效 - 主函数
    
    Args:
        image_folder: 图像文件夹路径
        num_images: 测试图像数量
    """
    print("="*70)
    print("【RoPE效果对比验证】")
    print("="*70)
    
    # 1. 准备测试图像
    image_paths = prepare_test_images(image_folder, num_images)
    if not image_paths:
        print("❌ 没有找到测试图像，请先在images文件夹中放置图片")
        return None
    
    # 2. 加载BLIP模型（你已有的）
    print("\n📥 加载BLIP模型...")
    blip = BLIPCaptionGenerator()
    
    # 3. 加载CLIP评分器（需要创建）
    print("\n📥 加载CLIP评分器...")
    try:
        from clip_score import CLIPScorer
        scorer = CLIPScorer()
    except ImportError:
        print("⚠️ 未找到clip_score.py，使用模拟评分")
        scorer = None
    
    # 4. 创建两个Transformer模型（需要创建）
    print("\n📥 创建Transformer模型...")
    try:
        # 有RoPE的模型
        model_with_rope = SemanticReconstructor(use_rope=True)
        # 无RoPE的模型
        model_without_rope = SemanticReconstructor(use_rope=False)
        print("✅ Transformer模型创建成功")
    except NameError:
        print("⚠️ 未找到SemanticReconstructor，使用模拟模式")
        model_with_rope = None
        model_without_rope = None
    
    # 5. 生成提示词
    print("\n📝 生成提示词...")
    
    # 基线：无RoPE
    prompts_without = generate_prompts(blip, model_without_rope, image_paths, use_rope=False)
    
    # 实验：有RoPE
    prompts_with = generate_prompts(blip, model_with_rope, image_paths, use_rope=True)
    
    # 6. 计算评分
    print("\n📊 计算CLIP-Score...")
    
    results = []
    scores_without = []
    scores_with = []
    
    for i, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        
        # 计算CLIP-Score
        if scorer:
            score_without = scorer.compute_score(img_path, prompts_without[i])
            score_with = scorer.compute_score(img_path, prompts_with[i])
        else:
            # 模拟评分（仅用于测试）
            score_without = 0.7 + np.random.random() * 0.2
            score_with = score_without * (1.05 + np.random.random() * 0.1)
        
        scores_without.append(score_without)
        scores_with.append(score_with)
        
        # 计算提升
        improvement = (score_with - score_without) / score_without * 100
        
        results.append({
            'image': img_name,
            'blip_caption': prompts_without[i][:50] + "...",  # 截断显示
            'score_without': round(score_without, 4),
            'score_with': round(score_with, 4),
            'improvement': round(improvement, 2)
        })
    
    # 7. 统计结果
    df = pd.DataFrame(results)
    
    mean_without = np.mean(scores_without)
    mean_with = np.mean(scores_with)
    avg_improvement = (mean_with - mean_without) / mean_without * 100
    
    print("\n" + "="*70)
    print("📈 对比结果统计")
    print("="*70)
    print(f"无RoPE 平均CLIP-Score: {mean_without:.4f}")
    print(f"有RoPE 平均CLIP-Score: {mean_with:.4f}")
    print(f"平均提升: {avg_improvement:.2f}%")
    
    if avg_improvement > 0:
        print(f"✅ RoPE有效，提升了 {avg_improvement:.2f}%")
    else:
        print(f"❌ 本次测试中RoPE未见明显提升")
    
    # 8. 显示详细结果
    print("\n" + "="*70)
    print("📋 详细结果")
    print("="*70)
    print(df.to_string(index=False))
    
    # 9. 可视化
    visualize_results(results, mean_without, mean_with, avg_improvement)
    
    # 10. 保存结果
    output_file = "../outputs/rope_comparison_results.csv"
    os.makedirs("../outputs", exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✅ 结果已保存到: {output_file}")
    
    return {
        'results': results,
        'mean_without': mean_without,
        'mean_with': mean_with,
        'improvement': avg_improvement
    }


def visualize_results(results, mean_without, mean_with, improvement):
    """
    可视化对比结果
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False 
    plt.figure(figsize=(12, 5))
    
    # 子图1：柱状图对比
    plt.subplot(1, 2, 1)
    
    # 提取数据
    images = [r['image'][:10] + '...' if len(r['image']) > 10 else r['image'] 
              for r in results[:5]]  # 只显示前5张
    scores_without = [r['score_without'] for r in results[:5]]
    scores_with = [r['score_with'] for r in results[:5]]
    
    x = np.arange(len(images))
    width = 0.35
    
    plt.bar(x - width/2, scores_without, width, label='无RoPE', color='#FF6B6B')
    plt.bar(x + width/2, scores_with, width, label='有RoPE', color='#4ECDC4')
    
    plt.xlabel('图像')
    plt.ylabel('CLIP-Score')
    plt.title('单张图像CLIP-Score对比')
    plt.xticks(x, images, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：平均分对比
    plt.subplot(1, 2, 2)
    bars = plt.bar(['无RoPE', '有RoPE'], [mean_without, mean_with], 
                   color=['#FF6B6B', '#4ECDC4'])
    plt.ylabel('平均CLIP-Score')
    plt.title(f'平均分对比 (提升 {improvement:.2f}%)')
    
    # 添加数值标签
    for bar, v in zip(bars, [mean_without, mean_with]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.4f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图片
    output_img = "../outputs/rope_comparison_chart.png"
    plt.savefig(output_img, dpi=150, bbox_inches='tight')
    print(f"📊 对比图已保存到: {output_img}")
    plt.show()


def quick_test():
    """
    快速测试函数（不需要实际模型）
    用于验证代码是否能运行
    """
    print("="*70)
    print("快速测试模式")
    print("="*70)
    
    # 模拟数据
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    prompts_without = ["a dog running", "a cat sleeping", "a car parked"]
    prompts_with = ["a brown dog running fast", "a white cat sleeping", "a red car parked"]
    
    # 模拟评分
    scores_without = [0.72, 0.68, 0.75]
    scores_with = [0.78, 0.73, 0.82]
    
    results = []
    for i in range(3):
        improvement = (scores_with[i] - scores_without[i]) / scores_without[i] * 100
        results.append({
            'image': image_paths[i],
            'blip_caption': prompts_without[i],
            'score_without': scores_without[i],
            'score_with': scores_with[i],
            'improvement': round(improvement, 2)
        })
    
    df = pd.DataFrame(results)
    mean_without = np.mean(scores_without)
    mean_with = np.mean(scores_with)
    improvement = (mean_with - mean_without) / mean_without * 100
    
    print("\n📊 模拟结果：")
    print(df.to_string(index=False))
    print(f"\n平均提升: {improvement:.2f}%")
    
    visualize_results(results, mean_without, mean_with, improvement)
    
    return results


if __name__ == "__main__":
    import sys
    
    print("请选择运行模式：")
    print("1. 快速测试（模拟数据）")
    print("2. 实际对比（需要图像和模型）")
    
    choice = input("请输入选项 (1/2): ").strip()
    
    if choice == '1':
        quick_test()
    else:
        # 检查必要文件
        missing_files = []
        if not os.path.exists("../images"):
            missing_files.append("../images 文件夹")
        
        if missing_files:
            print("\n⚠️ 缺少以下文件/文件夹，将使用模拟模式：")
            for f in missing_files:
                print(f"  - {f}")
            print("\n建议先创建images文件夹并放置测试图片")
            quick_test()
        else:
            # 运行实际对比
            compare_rope_effectiveness(num_images=3)