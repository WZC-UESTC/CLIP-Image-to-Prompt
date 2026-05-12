"""
rope_base_frequency_analysis.py
RoPE基频调参实验结果可视化分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 数据加载与预处理
# ============================================================
def load_and_preprocess(data_path):
    """加载数据并预处理"""
    df = pd.read_csv(data_path)
    
    print("="*70)
    print("RoPE基频调参实验结果分析")
    print("="*70)
    print(f"总样本数: {len(df)}")
    print(f"数据列: {df.columns.tolist()}")
    
    # 定义各配置对应的分数列
    configs = {
        'Baseline (贪心)': 'score_baseline',
        'Beam Only (k=3)': 'score_best_beam',
        'RoPE base=10000': 'score_rope_10000',
        'RoPE base=5000': 'score_rope_5000',
        'RoPE base=20000': 'score_rope_20000',
        'RoPE base=100000': 'score_rope_100000'
    }
    
    return df, configs


# ============================================================
# 2. 图1：各配置平均分对比柱状图
# ============================================================
def plot_avg_scores_comparison(df, configs, save_path):
    """图1：各配置平均CLIP-Score对比"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 计算平均分
    names = []
    scores = []
    stds = []
    
    for name, col in configs.items():
        if col in df.columns:
            scores.append(df[col].mean())
            stds.append(df[col].std())
            names.append(name)
    
    # 颜色渐变：基线最暗，最优最亮
    colors = ['#95a5a6', '#3498db', '#2ecc71', '#27ae60', '#f39c12', '#e74c3c']
    
    bars = ax.bar(names, scores, yerr=stds, capsize=5, color=colors[:len(names)], 
                  edgecolor='black', linewidth=1.5)
    ax.set_ylabel('平均 CLIP-Score', fontsize=12)
    ax.set_title('不同配置的CLIP-Score对比', fontsize=14)
    ax.set_ylim([0, 0.5])
    ax.axhline(y=scores[0], color='gray', linestyle='--', alpha=0.5, label=f'基线: {scores[0]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签和提升百分比
    baseline = scores[0]
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.008,
                f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        if i > 0:
            improvement = (score - baseline) / baseline * 100
            color = '#2ecc71' if improvement > 0 else '#e74c3c'
            ax.text(bar.get_x() + bar.get_width()/2, height - 0.025,
                    f'{improvement:+.1f}%', ha='center', va='bottom', 
                    fontsize=9, color=color, fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图1已保存: {save_path}")
    plt.close()


# ============================================================
# 3. 图2：基频参数对性能的影响曲线
# ============================================================
def plot_base_frequency_curve(df, save_path):
    """图2：基频参数对性能的影响曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 提取基频和对应分数
    base_freqs = [5000, 10000, 20000, 100000]
    scores = []
    
    for freq in base_freqs:
        col = f'score_rope_{freq}'
        if col in df.columns:
            scores.append(df[col].mean())
    
    # 添加基线作为参考
    base_freqs.insert(0, 0)
    scores.insert(0, df['score_best_beam'].mean())
    
    # 绘制曲线
    ax.plot(base_freqs, scores, 'o-', linewidth=2, markersize=8, color='#3498db')
    ax.set_xscale('log')  # 对数坐标
    ax.set_xlabel('基频参数 (base)', fontsize=12)
    ax.set_ylabel('平均 CLIP-Score', fontsize=12)
    ax.set_title('RoPE基频参数对性能的影响', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 标注最优值
    best_idx = np.argmax(scores)
    ax.scatter([base_freqs[best_idx]], [scores[best_idx]], 
               s=200, c='red', marker='*', zorder=5, label='最优基频')
    ax.annotate(f'最优: base={base_freqs[best_idx]}\nscore={scores[best_idx]:.4f}',
                xy=(base_freqs[best_idx], scores[best_idx]),
                xytext=(base_freqs[best_idx]*1.5, scores[best_idx]-0.01),
                fontsize=10, ha='left')
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图2已保存: {save_path}")
    plt.close()


# ============================================================
# 4. 图3：箱线图 - 各配置分数分布
# ============================================================
def plot_box_distribution(df, configs, save_path):
    """图3：各配置分数分布箱线图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 准备数据
    plot_data = []
    labels = []
    
    for name, col in configs.items():
        if col in df.columns:
            plot_data.append(df[col].dropna())
            labels.append(name)
    
    # 绘制箱线图
    bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, showmeans=True)
    
    # 设置颜色
    colors = ['#95a5a6', '#3498db', '#2ecc71', '#27ae60', '#f39c12', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('CLIP-Score', fontsize=12)
    ax.set_title('各配置分数分布对比', fontsize=14)
    ax.set_ylim([0, 0.5])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图3已保存: {save_path}")
    plt.close()


# ============================================================
# 5. 图4：提升幅度分布直方图
# ============================================================
def plot_improvement_distribution(df, save_path):
    """图4：相对于基线的提升幅度分布"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    configs_to_compare = [
        ('Beam Only', 'score_best_beam', axes[0, 0]),
        ('RoPE base=10000', 'score_rope_10000', axes[0, 1]),
        ('RoPE base=5000', 'score_rope_5000', axes[1, 0]),
        ('RoPE base=20000', 'score_rope_20000', axes[1, 1])
    ]
    
    baseline = df['score_baseline']
    
    for name, col, ax in configs_to_compare:
        if col in df.columns:
            improvements = df[col] - baseline
            improvements = improvements.dropna()
            
            # 绘制直方图
            n, bins, patches = ax.hist(improvements, bins=30, edgecolor='black', 
                                        alpha=0.7, color='#3498db')
            
            # 着色：正改进绿色，负改进红色
            for patch, bin_edge in zip(patches, bins[:-1]):
                if bin_edge >= 0:
                    patch.set_facecolor('#2ecc71')
                else:
                    patch.set_facecolor('#e74c3c')
            
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='零提升')
            ax.axvline(x=improvements.mean(), color='blue', linestyle='-', 
                       linewidth=2, label=f'平均: {improvements.mean():.4f}')
            ax.set_xlabel('CLIP-Score 提升值', fontsize=10)
            ax.set_ylabel('图片数量', fontsize=10)
            ax.set_title(f'{name} 提升分布\n改进比例: {(improvements > 0).mean()*100:.1f}%', fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图4已保存: {save_path}")
    plt.close()


# ============================================================
# 6. 图5：性能雷达图
# ============================================================
def plot_radar_chart(df, configs, save_path):
    """图5：多维度性能雷达图"""
    # 定义评估维度
    dimensions = ['平均分', '中位数', '最高分', '稳定性(1/标准差)', '效率(分/秒)']
    
    # 计算各配置在各维度的得分
    config_scores = {}
    
    for name, col in configs.items():
        if col in df.columns:
            scores = df[col].dropna()
            if len(scores) > 0:
                config_scores[name] = {
                    '平均分': scores.mean(),
                    '中位数': scores.median(),
                    '最高分': scores.max(),
                    '稳定性(1/标准差)': 1 / scores.std() if scores.std() > 0 else 0,
                    '效率(分/秒)': scores.mean() / df['total_time_seconds'].mean() if 'total_time_seconds' in df.columns else scores.mean()
                }
    
    # 归一化
    all_values = []
    for dim in dimensions:
        dim_values = [config_scores[c][dim] for c in config_scores if c in config_scores]
        max_val = max(dim_values) if dim_values else 1
        for c in config_scores:
            config_scores[c][dim] = config_scores[c][dim] / max_val
    
    # 绘制雷达图
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = ['#95a5a6', '#3498db', '#2ecc71', '#27ae60', '#f39c12', '#e74c3c']
    
    for i, (name, color) in enumerate(zip(config_scores.keys(), colors)):
        values = [config_scores[name][dim] for dim in dimensions]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('多维度性能雷达图', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图5已保存: {save_path}")
    plt.close()


# ============================================================
# 7. 图6：最优配置与基线对比散点图
# ============================================================
def plot_scatter_comparison(df, best_col, save_path):
    """图6：最优配置 vs 基线 散点图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    baseline = df['score_baseline']
    best = df[best_col]
    
    # 计算提升
    improvement = best - baseline
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvement]
    
    scatter = ax.scatter(baseline, best, c=colors, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
    # 对角线
    max_val = max(baseline.max(), best.max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='y=x (持平)')
    
    # 趋势线
    z = np.polyfit(baseline, best, 1)
    p = np.poly1d(z)
    x_line = np.linspace(baseline.min(), baseline.max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'趋势线 (斜率={z[0]:.3f})')
    
    ax.set_xlabel('基线 CLIP-Score', fontsize=12)
    ax.set_ylabel('最优配置 CLIP-Score', fontsize=12)
    ax.set_title('最优配置 vs 基线 分数对比', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    better_count = (improvement > 0).sum()
    total = len(improvement)
    mean_imp = improvement.mean()
    text = f'改进案例: {better_count}/{total} ({better_count/total*100:.1f}%)\n平均提升: {mean_imp:.4f}'
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图6已保存: {save_path}")
    plt.close()


# ============================================================
# 8. 生成统计报告
# ============================================================
def generate_statistics_report(df, configs, output_path):
    """生成详细统计报告"""
    report = []
    report.append("="*70)
    report.append("RoPE基频调参实验统计报告")
    report.append("="*70)
    report.append(f"\n总样本数: {len(df)}")
    
    report.append("\n" + "-"*50)
    report.append("1. 各配置平均性能")
    report.append("-"*50)
    
    baseline = df['score_baseline'].mean()
    report.append(f"\n基线 (贪心解码): {baseline:.4f}")
    
    for name, col in configs.items():
        if col in df.columns and name != 'Baseline (贪心)':
            score = df[col].mean()
            improvement = (score - baseline) / baseline * 100
            report.append(f"{name}: {score:.4f} (提升 {improvement:+.2f}%)")
    
    report.append("\n" + "-"*50)
    report.append("2. 基频参数最优选择")
    report.append("-"*50)
    
    freq_scores = {}
    for freq in [5000, 10000, 20000, 100000]:
        col = f'score_rope_{freq}'
        if col in df.columns:
            freq_scores[freq] = df[col].mean()
    
    if freq_scores:
        best_freq = max(freq_scores, key=freq_scores.get)
        report.append(f"\n各基频平均分:")
        for freq, score in freq_scores.items():
            report.append(f"  base={freq}: {score:.4f}")
        report.append(f"\n🏆 最优基频: base={best_freq} (score={freq_scores[best_freq]:.4f})")
    
    report.append("\n" + "-"*50)
    report.append("3. 统计显著性检验")
    report.append("-"*50)
    
    for name, col in configs.items():
        if col in df.columns and name != 'Baseline (贪心)':
            t_stat, p_value = stats.ttest_rel(df['score_baseline'], df[col])
            report.append(f"\n{name} vs 基线:")
            report.append(f"  t统计量: {t_stat:.4f}")
            report.append(f"  p值: {p_value:.6f}")
            report.append(f"  差异显著性: {'显著' if p_value < 0.05 else '不显著'}")
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\n✅ 统计报告已保存: {output_path}")
    return report


# ============================================================
# 9. 主函数
# ============================================================
def main():
    """主函数"""
    # 文件路径（请根据实际情况修改）
    data_path = "C:/Users/13113/Desktop/数据分析/rope_experiment_results.csv"
    output_dir = "C:/Users/13113/Desktop/数据分析/charts"
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    df, configs = load_and_preprocess(data_path)
    
    # 生成所有图表
    print("\n开始生成可视化图表...")
    print("-"*50)
    
    # 图1：平均分对比
    plot_avg_scores_comparison(df, configs, f'{output_dir}/1_avg_scores.png')
    
    # 图2：基频影响曲线
    plot_base_frequency_curve(df, f'{output_dir}/2_base_frequency_curve.png')
    
    # 图3：箱线图分布
    plot_box_distribution(df, configs, f'{output_dir}/3_box_distribution.png')
    
    # 图4：提升分布
    plot_improvement_distribution(df, f'{output_dir}/4_improvement_distribution.png')
    
    # 图5：雷达图
    plot_radar_chart(df, configs, f'{output_dir}/5_radar_chart.png')
    
    # 图6：散点对比（使用最优配置）
    # 确定最优配置列
    best_col = 'score_rope_10000'  # 根据你的数据调整
    if best_col in df.columns:
        plot_scatter_comparison(df, best_col, f'{output_dir}/6_scatter_comparison.png')
    
    # 生成统计报告
    generate_statistics_report(df, configs, f'{output_dir}/statistics_report.txt')
    
    print("\n" + "="*70)
    print("🎉 所有图表生成完成！")
    print(f"📁 输出目录: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()