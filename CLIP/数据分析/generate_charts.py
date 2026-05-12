"""
generate_charts.py
基于8万张图片实验结果生成对比图表
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

# 读取数据
df = pd.read_csv('C:/Users/13113/Desktop/数据分析/final_comparison_dataset.csv')

print("="*60)
print("数据基本信息")
print("="*60)
print(f"总样本数: {len(df)}")
print(f"成功案例数: {len(df[df['status'] == 'Success'])}")
print(f"列名: {df.columns.tolist()}")

# ============================================================
# 图1：整体性能对比箱线图
# ============================================================
def plot_box_comparison(df, save_path):
    """图1：基线 vs 你的方法 分数分布对比"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 子图1：分数分布箱线图
    ax1 = axes[0]
    data_to_plot = [df['score_baseline'].dropna(), df['score_ours'].dropna()]
    bp = ax1.boxplot(data_to_plot, labels=['基线', '你的方法'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#FF6B6B')
    bp['boxes'][1].set_facecolor('#4ECDC4')
    ax1.set_ylabel('CLIP-Score')
    ax1.set_title('CLIP-Score分布对比')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    
    # 子图2：时间分布箱线图
    ax2 = axes[1]
    time_data = [df['time_baseline'].dropna(), df['time_ours'].dropna()]
    bp2 = ax2.boxplot(time_data, labels=['基线', '你的方法'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('#FF6B6B')
    bp2['boxes'][1].set_facecolor('#4ECDC4')
    ax2.set_ylabel('耗时 (秒)')
    ax2.set_title('推理时间对比')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图1已保存: {save_path}")
    plt.close()

# ============================================================
# 图2：提升分布直方图
# ============================================================
def plot_improvement_hist(df, save_path):
    """图2：提升值的分布"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1：提升直方图
    ax1 = axes[0]
    improvements = df['improvement'].dropna()
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvements]
    
    n, bins, patches = ax1.hist(improvements, bins=30, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('CLIP-Score提升值')
    ax1.set_ylabel('图片数量')
    ax1.set_title(f'提升分布 (平均提升: {improvements.mean():.4f})')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零提升线')
    ax1.axvline(x=improvements.mean(), color='blue', linestyle='-', linewidth=2, label=f'平均提升: {improvements.mean():.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2：累计提升曲线
    ax2 = axes[1]
    sorted_improvements = np.sort(improvements)
    cumulative = np.arange(1, len(sorted_improvements) + 1) / len(sorted_improvements)
    ax2.plot(sorted_improvements, cumulative, 'b-', linewidth=2)
    ax2.fill_between(sorted_improvements, cumulative, alpha=0.3)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('CLIP-Score提升值')
    ax2.set_ylabel('累计比例')
    ax2.set_title('提升值累计分布曲线')
    ax2.grid(True, alpha=0.3)
    
    # 标注中位数
    median_val = improvements.median()
    ax2.axvline(x=median_val, color='green', linestyle=':', linewidth=2, label=f'中位数: {median_val:.4f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图2已保存: {save_path}")
    plt.close()

# ============================================================
# 图3：散点图（基线 vs 你的方法）
# ============================================================
def plot_scatter_comparison(df, save_path):
    """图3：基线分数 vs 你的方法分数散点图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 筛选有效数据
    valid = df[(df['score_baseline'].notna()) & (df['score_ours'].notna())]
    
    # 绘制散点
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in valid['improvement']]
    ax.scatter(valid['score_baseline'], valid['score_ours'], c=colors, alpha=0.5, s=20)
    
    # 添加对角线
    max_val = max(valid['score_baseline'].max(), valid['score_ours'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='y=x (持平)')
    
    # 添加趋势线
    z = np.polyfit(valid['score_baseline'], valid['score_ours'], 1)
    p = np.poly1d(z)
    ax.plot(valid['score_baseline'], p(valid['score_baseline']), 'r-', linewidth=2, label=f'趋势线 (斜率={z[0]:.3f})')
    
    ax.set_xlabel('基线 CLIP-Score')
    ax.set_ylabel('你的方法 CLIP-Score')
    ax.set_title('基线 vs 你的方法 分数对比散点图')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    better_count = (valid['improvement'] > 0).sum()
    total = len(valid)
    text = f'改进案例: {better_count}/{total} ({better_count/total*100:.1f}%)'
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图3已保存: {save_path}")
    plt.close()

# ============================================================
# 图4：性能对比雷达图
# ============================================================
def plot_radar_chart(df, save_path):
    """图4：多维度性能对比雷达图"""
    # 计算各项指标
    metrics = {
        '平均CLIP-Score': [df['score_baseline'].mean(), df['score_ours'].mean()],
        '中位数CLIP-Score': [df['score_baseline'].median(), df['score_ours'].median()],
        '最高分': [df['score_baseline'].max(), df['score_ours'].max()],
        '稳定性(1/标准差)': [1/df['score_baseline'].std(), 1/df['score_ours'].std()],
        '成功率': [1.0, (df['improvement'] > 0).mean()],
        '效率(分数/秒)': [
            df['score_baseline'].mean() / df['time_baseline'].mean(),
            df['score_ours'].mean() / df['time_ours'].mean()
        ]
    }
    
    # 归一化
    categories = list(metrics.keys())
    baseline_values = []
    ours_values = []
    
    for cat in categories:
        b, o = metrics[cat]
        max_val = max(b, o)
        baseline_values.append(b / max_val if max_val > 0 else 0)
        ours_values.append(o / max_val if max_val > 0 else 0)
    
    # 闭合图形
    baseline_values += baseline_values[:1]
    ours_values += ours_values[:1]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, baseline_values, 'o-', linewidth=2, label='基线', color='#FF6B6B')
    ax.fill(angles, baseline_values, alpha=0.25, color='#FF6B6B')
    
    ax.plot(angles, ours_values, 'o-', linewidth=2, label='你的方法', color='#4ECDC4')
    ax.fill(angles, ours_values, alpha=0.25, color='#4ECDC4')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('多维度性能对比雷达图', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图4已保存: {save_path}")
    plt.close()

# ============================================================
# 图5：时间-质量权衡图
# ============================================================
def plot_time_quality_tradeoff(df, save_path):
    """图5：时间与质量的权衡"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 计算分组统计
    df['quality_bin'] = pd.cut(df['score_baseline'], bins=10)
    baseline_group = df.groupby('quality_bin').agg({
        'score_baseline': 'mean',
        'time_baseline': 'mean'
    }).dropna()
    
    df['quality_bin_ours'] = pd.cut(df['score_ours'], bins=10)
    ours_group = df.groupby('quality_bin_ours').agg({
        'score_ours': 'mean',
        'time_ours': 'mean'
    }).dropna()
    
    ax.scatter(baseline_group['score_baseline'], baseline_group['time_baseline'], 
               s=100, c='#FF6B6B', label='基线', alpha=0.7, edgecolors='black')
    ax.scatter(ours_group['score_ours'], ours_group['time_ours'], 
               s=100, c='#4ECDC4', label='你的方法', alpha=0.7, edgecolors='black')
    
    # 添加趋势线
    z1 = np.polyfit(baseline_group['score_baseline'], baseline_group['time_baseline'], 1)
    p1 = np.poly1d(z1)
    ax.plot(baseline_group['score_baseline'], p1(baseline_group['score_baseline']), 
            'r--', linewidth=2, alpha=0.7)
    
    z2 = np.polyfit(ours_group['score_ours'], ours_group['time_ours'], 1)
    p2 = np.poly1d(z2)
    ax.plot(ours_group['score_ours'], p2(ours_group['score_ours']), 
            'g--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('CLIP-Score')
    ax.set_ylabel('耗时 (秒)')
    ax.set_title('时间-质量权衡图')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图5已保存: {save_path}")
    plt.close()

# ============================================================
# 图6：性能提升柱状图
# ============================================================
def plot_performance_bars(df, save_path):
    """图6：关键指标对比柱状图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = {
        '平均CLIP-Score': [df['score_baseline'].mean(), df['score_ours'].mean()],
        '中位数CLIP-Score': [df['score_baseline'].median(), df['score_ours'].median()],
        '最高分': [df['score_baseline'].max(), df['score_ours'].max()],
        '效率(分数/秒)': [
            df['score_baseline'].mean() / df['time_baseline'].mean(),
            df['score_ours'].mean() / df['time_ours'].mean()
        ]
    }
    
    x = np.arange(len(metrics))
    width = 0.35
    
    baseline_vals = [metrics[m][0] for m in metrics]
    ours_vals = [metrics[m][1] for m in metrics]
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='基线', color='#FF6B6B')
    bars2 = ax.bar(x + width/2, ours_vals, width, label='你的方法', color='#4ECDC4')
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys())
    ax.set_ylabel('分数')
    ax.set_title('关键指标对比')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars1, baseline_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, ours_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图6已保存: {save_path}")
    plt.close()

# ============================================================
# 图7：成功/失败案例分析
# ============================================================
def plot_case_analysis(df, save_path):
    """图7：成功与失败案例统计"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1：成功/失败比例
    ax1 = axes[0]
    success_count = (df['improvement'] > 0).sum()
    fail_count = (df['improvement'] <= 0).sum()
    equal_count = (df['improvement'] == 0).sum()
    
    labels = ['改进', '持平', '下降']
    sizes = [success_count, equal_count, fail_count]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    explode = (0.05, 0, 0)
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', explode=explode, shadow=True)
    ax1.set_title(f'改进效果分布 (总样本: {len(df)})')
    
    # 子图2：改进幅度分布
    ax2 = axes[1]
    improvements = df['improvement'].dropna()
    
    bins = [-0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.15]
    labels_bins = ['<-0.05', '-0.05~-0.02', '-0.02~0', '0~0.02', '0.02~0.05', '0.05~0.1', '>0.1']
    
    hist, _ = np.histogram(improvements, bins=bins)
    
    bars = ax2.bar(labels_bins, hist, color=['#e74c3c', '#e74c3c', '#f39c12', '#2ecc71', '#2ecc71', '#2ecc71', '#2ecc71'])
    ax2.set_xlabel('改进幅度')
    ax2.set_ylabel('图片数量')
    ax2.set_title('改进幅度分布')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, v in zip(bars, hist):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图7已保存: {save_path}")
    plt.close()

# ============================================================
# 生成统计报告
# ============================================================
def generate_statistics_report(df, save_path):
    """生成统计报告"""
    report = []
    report.append("="*60)
    report.append("实验结果统计报告")
    report.append("="*60)
    report.append(f"\n总样本数: {len(df)}")
    report.append(f"成功案例数: {(df['improvement'] > 0).sum()} ({(df['improvement'] > 0).mean()*100:.1f}%)")
    
    report.append(f"\n【CLIP-Score对比】")
    report.append(f"  基线 - 平均: {df['score_baseline'].mean():.4f}")
    report.append(f"  基线 - 中位数: {df['score_baseline'].median():.4f}")
    report.append(f"  基线 - 标准差: {df['score_baseline'].std():.4f}")
    report.append(f"  你的方法 - 平均: {df['score_ours'].mean():.4f}")
    report.append(f"  你的方法 - 中位数: {df['score_ours'].median():.4f}")
    report.append(f"  你的方法 - 标准差: {df['score_ours'].std():.4f}")
    report.append(f"  平均提升: {df['improvement'].mean():.4f}")
    
    report.append(f"\n【推理时间对比】")
    report.append(f"  基线 - 平均: {df['time_baseline'].mean():.4f}s")
    report.append(f"  你的方法 - 平均: {df['time_ours'].mean():.4f}s")
    report.append(f"  时间增加: {(df['time_ours'].mean() - df['time_baseline'].mean()):.4f}s")
    
    report.append(f"\n【效率对比】")
    baseline_efficiency = df['score_baseline'].mean() / df['time_baseline'].mean()
    ours_efficiency = df['score_ours'].mean() / df['time_ours'].mean()
    report.append(f"  基线效率: {baseline_efficiency:.4f} 分数/秒")
    report.append(f"  你的方法效率: {ours_efficiency:.4f} 分数/秒")
    report.append(f"  效率变化: {(ours_efficiency/baseline_efficiency-1)*100:+.1f}%")
    
    report.append(f"\n【统计检验】")
    t_stat, p_value = stats.ttest_rel(df['score_baseline'], df['score_ours'])
    report.append(f"  配对t检验 p值: {p_value:.6f}")
    report.append(f"  差异显著性: {'显著' if p_value < 0.05 else '不显著'}")
    
    # 保存报告
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\n✅ 统计报告已保存: {save_path}")
    return report

# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    output_dir = 'C:/Users/13113/Desktop/数据分析/charts'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("开始生成对比图表...")
    print("="*60)
    
    # 生成所有图表
    plot_box_comparison(df, f'{output_dir}/1_box_comparison.png')
    plot_improvement_hist(df, f'{output_dir}/2_improvement_hist.png')
    plot_scatter_comparison(df, f'{output_dir}/3_scatter_comparison.png')
    plot_radar_chart(df, f'{output_dir}/4_radar_chart.png')
    plot_time_quality_tradeoff(df, f'{output_dir}/5_time_quality_tradeoff.png')
    plot_performance_bars(df, f'{output_dir}/6_performance_bars.png')
    plot_case_analysis(df, f'{output_dir}/7_case_analysis.png')
    
    # 生成统计报告
    generate_statistics_report(df, f'{output_dir}/statistics_report.txt')
    
    print("\n" + "="*60)
    print("所有图表生成完成！")
    print(f"输出目录: {output_dir}")
    print("="*60)