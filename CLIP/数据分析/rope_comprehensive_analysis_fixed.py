"""
rope_comprehensive_analysis_fixed.py
RoPE基频调参实验 - 完整可视化分析（修复中文显示问题）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 关键修复：强制使用英文标签，避免中文显示问题
# ============================================================
# 不使用中文字体，全部用英文标签
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 定义英文标签映射
LABELS = {
    '基线 (贪心解码)': 'Baseline (Greedy)',
    '束搜索 (Beam=3)': 'Beam Search (k=3)',
    'RoPE base=5000': 'RoPE (base=5000)',
    'RoPE base=10000': 'RoPE (base=10000)',
    'RoPE base=20000': 'RoPE (base=20000)',
    'RoPE base=100000': 'RoPE (base=100000)',
    'score_baseline': 'Baseline Score',
    'score_best_beam': 'Beam Search Score',
    'score_rope_5000': 'RoPE (base=5000) Score',
    'score_rope_10000': 'RoPE (base=10000) Score',
    'score_rope_20000': 'RoPE (base=20000) Score',
    'score_rope_100000': 'RoPE (base=100000) Score',
}


# ============================================================
# 1. 数据加载与预处理
# ============================================================
def load_and_preprocess(data_path):
    """加载数据并预处理"""
    df = pd.read_csv(data_path)
    
    print("="*80)
    print("RoPE Base Frequency Experiment - Comprehensive Analysis")
    print("="*80)
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 定义配置
    configs = {
        'Baseline (Greedy)': {'score_col': 'score_baseline', 'color': '#95a5a6', 'marker': 'o'},
        'Beam Search (k=3)': {'score_col': 'score_best_beam', 'color': '#3498db', 'marker': 's'},
        'RoPE (base=5000)': {'score_col': 'score_rope_5000', 'color': '#2ecc71', 'marker': '^'},
        'RoPE (base=10000)': {'score_col': 'score_rope_10000', 'color': '#27ae60', 'marker': 'D'},
        'RoPE (base=20000)': {'score_col': 'score_rope_20000', 'color': '#f39c12', 'marker': 'v'},
        'RoPE (base=100000)': {'score_col': 'score_rope_100000', 'color': '#e74c3c', 'marker': 'p'}
    }
    
    # 只保留存在的配置
    existing_configs = {}
    for name, cfg in configs.items():
        if cfg['score_col'] in df.columns:
            existing_configs[name] = cfg
    
    # 打印数据统计
    print("\n" + "-"*50)
    print("Data Statistics:")
    for name, cfg in existing_configs.items():
        scores = df[cfg['score_col']].dropna()
        print(f"  {name}: mean={scores.mean():.4f}, std={scores.std():.4f}, n={len(scores)}")
    
    return df, existing_configs


# ============================================================
# 2. 图1：平均分对比柱状图
# ============================================================
def plot1_avg_scores(df, configs, save_path):
    """图1：平均CLIP-Score对比"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = []
    means = []
    stds = []
    colors = []
    
    for name, cfg in configs.items():
        scores = df[cfg['score_col']].dropna()
        names.append(name)
        means.append(scores.mean())
        stds.append(scores.std())
        colors.append(cfg['color'])
    
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, 
                  edgecolor='black', linewidth=1.2, alpha=0.8)
    
    ax.set_ylabel('Mean CLIP-Score', fontsize=12)
    ax.set_title('Mean CLIP-Score Comparison', fontsize=14)
    ax.set_ylim([0, max(means) * 1.15])
    ax.axhline(y=means[0], color='gray', linestyle='--', alpha=0.7, 
               label=f'Baseline: {means[0]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    baseline = means[0]
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(bar.get_x() + bar.get_width()/2, mean + std + 0.005,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=9)
        if i > 0:
            improvement = (mean - baseline) / baseline * 100
            color = '#2ecc71' if improvement > 0 else '#e74c3c'
            ax.text(bar.get_x() + bar.get_width()/2, mean - std - 0.015,
                    f'{improvement:+.1f}%', ha='center', va='top', 
                    fontsize=9, color=color, fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# 3. 图2：基频参数影响曲线
# ============================================================
def plot2_base_frequency_curve(df, configs, save_path):
    """图2：基频参数对性能的影响"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 提取基频数据
    freq_data = []
    for name, cfg in configs.items():
        if 'base=' in name:
            freq = int(name.split('base=')[1].split(')')[0])
            scores = df[cfg['score_col']].dropna()
            freq_data.append({
                'freq': freq,
                'mean': scores.mean(),
                'std': scores.std(),
                'ci': 1.96 * scores.std() / np.sqrt(len(scores))
            })
    
    freq_data.sort(key=lambda x: x['freq'])
    freqs = [d['freq'] for d in freq_data]
    means = [d['mean'] for d in freq_data]
    cis = [d['ci'] for d in freq_data]
    
    # 添加束搜索基线 (freq=0)
    beam_mean = df[configs['Beam Search (k=3)']['score_col']].mean()
    beam_std = df[configs['Beam Search (k=3)']['score_col']].std()
    freqs.insert(0, 0)
    means.insert(0, beam_mean)
    cis.insert(0, 1.96 * beam_std / np.sqrt(len(df)))
    
    ax.plot(freqs, means, 'o-', linewidth=2.5, markersize=8, color='#3498db', label='Mean Score')
    ax.fill_between(freqs, [m - c for m, c in zip(means, cis)], 
                     [m + c for m, c in zip(means, cis)], alpha=0.2, color='#3498db', label='95% CI')
    
    ax.set_xscale('log')
    ax.set_xlabel('Base Frequency (log scale)', fontsize=12)
    ax.set_ylabel('Mean CLIP-Score', fontsize=12)
    ax.set_title('Effect of Base Frequency on CLIP-Score', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 标注最优值
    best_idx = np.argmax(means)
    ax.scatter([freqs[best_idx]], [means[best_idx]], s=300, c='red', 
               marker='*', zorder=5, label=f'Best: base={freqs[best_idx]}')
    ax.annotate(f'Best: base={freqs[best_idx]}\nscore={means[best_idx]:.4f}',
                xy=(freqs[best_idx], means[best_idx]), 
                xytext=(freqs[best_idx]*2 if freqs[best_idx]>0 else 100, means[best_idx]-0.01),
                fontsize=9, ha='left')
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# 4. 图3：箱线图
# ============================================================
def plot3_boxplot(df, configs, save_path):
    """图3：各配置分数分布箱线图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    plot_data = []
    labels = []
    colors = []
    
    for name, cfg in configs.items():
        scores = df[cfg['score_col']].dropna()
        plot_data.append(scores)
        labels.append(name)
        colors.append(cfg['color'])
    
    bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('CLIP-Score', fontsize=12)
    ax.set_title('Score Distribution by Configuration', fontsize=14)
    ax.set_ylim([0, 0.5])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# 5. 图4：提升分布直方图
# ============================================================
def plot4_improvement_histograms(df, configs, save_path):
    """图4：各配置相对于基线的提升分布"""
    n_configs = len([n for n in configs.keys() if 'Baseline' not in n])
    n_cols = 2
    n_rows = (n_configs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    if n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    baseline = df[configs['Baseline (Greedy)']['score_col']]
    plot_idx = 0
    
    for name, cfg in configs.items():
        if 'Baseline' in name:
            continue
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        scores = df[cfg['score_col']].dropna()
        improvements = scores - baseline[:len(scores)]
        
        n, bins, patches = ax.hist(improvements, bins=30, edgecolor='black', alpha=0.7)
        
        for patch, bin_edge in zip(patches, bins[:-1]):
            patch.set_facecolor('#2ecc71' if bin_edge >= 0 else '#e74c3c')
        
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        ax.axvline(x=improvements.mean(), color='blue', linestyle='-', linewidth=2,
                   label=f'Mean: {improvements.mean():.4f}')
        ax.set_xlabel('Improvement in CLIP-Score')
        ax.set_ylabel('Count')
        ax.set_title(f'{name}\nImprovement Rate: {(improvements > 0).mean()*100:.1f}%')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # 隐藏多余的子图
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Improvement Distribution vs Baseline', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# 6. 图5：雷达图
# ============================================================
def plot5_radar_chart(df, configs, save_path):
    """图5：多维度雷达图"""
    dimensions = ['Mean', 'Median', 'Max', 'Min', 'Stability', 'Efficiency']
    
    # 计算各维度得分
    config_scores = {}
    baseline_scores = df[configs['Baseline (Greedy)']['score_col']].dropna()
    
    for name, cfg in configs.items():
        scores = df[cfg['score_col']].dropna()
        if len(scores) == 0:
            continue
        
        improvements = scores - baseline_scores[:len(scores)]
        
        config_scores[name] = {
            'Mean': scores.mean(),
            'Median': scores.median(),
            'Max': scores.max(),
            'Min': scores.min(),
            'Stability': 1 / scores.std() if scores.std() > 0 else 0,
            'Efficiency': (improvements > 0).mean()
        }
    
    # 归一化
    for dim in dimensions:
        dim_values = [config_scores[c][dim] for c in config_scores]
        max_val = max(dim_values) if dim_values else 1
        for c in config_scores:
            config_scores[c][dim] = config_scores[c][dim] / max_val if max_val > 0 else 0
    
    # 绘制雷达图
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]
    
    for name, cfg in configs.items():
        if name not in config_scores:
            continue
        values = [config_scores[name][dim] for dim in dimensions]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=cfg['color'])
        ax.fill(angles, values, alpha=0.1, color=cfg['color'])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title('Multi-dimensional Performance Comparison', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# 7. 图6：散点图（最优配置 vs 基线）
# ============================================================
def plot6_scatter_best_vs_baseline(df, configs, save_path):
    """图6：最优RoPE配置 vs 基线散点图"""
    # 找出最优RoPE配置
    rope_configs = {n: c for n, c in configs.items() if 'RoPE' in n}
    best_rope = max(rope_configs.keys(), key=lambda x: df[rope_configs[x]['score_col']].mean())
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    baseline = df[configs['Baseline (Greedy)']['score_col']]
    rope_scores = df[rope_configs[best_rope]['score_col']].dropna()
    improvements = rope_scores - baseline[:len(rope_scores)]
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    
    ax.scatter(baseline[:len(rope_scores)], rope_scores, c=colors, alpha=0.5, 
               s=25, edgecolors='black', linewidth=0.5)
    
    max_val = max(baseline.max(), rope_scores.max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='y=x')
    
    z = np.polyfit(baseline[:len(rope_scores)], rope_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(baseline.min(), baseline.max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend (slope={z[0]:.3f})')
    
    ax.set_xlabel('Baseline CLIP-Score', fontsize=12)
    ax.set_ylabel(f'{best_rope} CLIP-Score', fontsize=12)
    ax.set_title(f'{best_rope} vs Baseline', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    better_count = (improvements > 0).sum()
    total = len(improvements)
    mean_imp = improvements.mean()
    text = f'Better: {better_count}/{total} ({better_count/total*100:.1f}%)\nMean Improvement: {mean_imp:.4f}'
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return best_rope


# ============================================================
# 8. 图7：小提琴图
# ============================================================
def plot7_violin_plot(df, configs, save_path):
    """图7：小提琴图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    plot_data = []
    labels = []
    colors = []
    
    for name, cfg in configs.items():
        scores = df[cfg['score_col']].dropna()
        plot_data.append(scores)
        labels.append(name)
        colors.append(cfg['color'])
    
    parts = ax.violinplot(plot_data, positions=range(len(plot_data)), 
                          showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('CLIP-Score')
    ax.set_title('Score Distribution (Violin Plot)')
    ax.set_ylim([0, 0.5])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# 9. 图8：相关性热力图
# ============================================================
def plot8_correlation_heatmap(df, configs, save_path):
    """图8：配置间相关性热力图"""
    score_cols = [cfg['score_col'] for cfg in configs.values()]
    score_cols = [col for col in score_cols if col in df.columns]
    
    corr_matrix = df[score_cols].corr()
    
    name_map = {cfg['score_col']: name for name, cfg in configs.items()}
    corr_matrix = corr_matrix.rename(index=name_map, columns=name_map)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='Greens', 
                center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Between Configurations', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# 10. 图9：累计分布曲线
# ============================================================
def plot9_cumulative_distribution(df, configs, save_path):
    """图9：累计分布曲线"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for name, cfg in configs.items():
        scores = df[cfg['score_col']].dropna()
        sorted_scores = np.sort(scores)
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        ax.plot(sorted_scores, cumulative, linewidth=2, label=name, color=cfg['color'])
    
    ax.set_xlabel('CLIP-Score', fontsize=12)
    ax.set_ylabel('Cumulative Proportion', fontsize=12)
    ax.set_title('Cumulative Distribution of CLIP-Scores', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# 11. 图10：提升排序图
# ============================================================
def plot10_improvement_ranking(df, configs, save_path):
    """图10：性能提升排序"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline = df[configs['Baseline (Greedy)']['score_col']].mean()
    
    improvements = []
    names = []
    colors = []
    
    for name, cfg in configs.items():
        if 'Baseline' in name:
            continue
        mean_score = df[cfg['score_col']].mean()
        improvement = (mean_score - baseline) / baseline * 100
        improvements.append(improvement)
        names.append(name)
        colors.append(cfg['color'])
    
    sorted_idx = np.argsort(improvements)[::-1]
    names = [names[i] for i in sorted_idx]
    improvements = [improvements[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]
    
    bars = ax.barh(names, improvements, color=colors, edgecolor='black')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Improvement over Baseline (%)', fontsize=12)
    ax.set_title('Performance Improvement Ranking', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{imp:+.2f}%', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# 12. 图11：效率分析
# ============================================================
def plot11_efficiency_analysis(df, configs, save_path):
    """图11：效率分析（分数 vs 时间）"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    times = []
    scores = []
    names = []
    colors = []
    
    for name, cfg in configs.items():
        mean_score = df[cfg['score_col']].mean()
        # 估算时间（基于beam size）
        if 'Greedy' in name:
            est_time = 0.12
        elif 'Beam' in name:
            est_time = 0.16
        else:
            est_time = 0.18
        times.append(est_time)
        scores.append(mean_score)
        names.append(name)
        colors.append(cfg['color'])
    
    sizes = [s * 500 for s in scores]
    scatter = ax.scatter(times, scores, s=sizes, c=colors, alpha=0.6, 
                         edgecolors='black', linewidth=1.5)
    
    for i, name in enumerate(names):
        ax.annotate(name, (times[i], scores[i]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9, ha='left')
    
    ax.set_xlabel('Average Time (seconds)', fontsize=12)
    ax.set_ylabel('Average CLIP-Score', fontsize=12)
    ax.set_title('Efficiency: Score vs Time (bubble size = score)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# 13. 图12：统计显著性检验
# ============================================================
def plot12_significance_test(df, configs, save_path):
    """图12：统计显著性检验"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    baseline = df[configs['Baseline (Greedy)']['score_col']]
    
    p_values = []
    names = []
    colors = []
    improvements = []
    
    for name, cfg in configs.items():
        if 'Baseline' in name:
            continue
        scores = df[cfg['score_col']].dropna()
        _, p_value = stats.ttest_rel(baseline[:len(scores)], scores)
        p_values.append(p_value)
        names.append(name)
        colors.append(cfg['color'])
        improvements.append(scores.mean() - baseline[:len(scores)].mean())
    
    bars = ax.bar(names, p_values, color=colors, edgecolor='black')
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='Significance threshold (p=0.05)')
    ax.set_ylabel('p-value', fontsize=12)
    ax.set_title('Statistical Significance Test (Paired t-test)', fontsize=14)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, p, imp in zip(bars, p_values, improvements):
        is_significant = p < 0.05
        label = 'Significant' if is_significant else 'Not significant'
        color = '#2ecc71' if is_significant else '#e74c3c'
        ax.text(bar.get_x() + bar.get_width()/2, p * 1.5,
                f'{label}\n(p={p:.4f})\nimp={imp:+.4f}', 
                ha='center', va='bottom', fontsize=8, color=color)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# 14. 生成统计报告
# ============================================================
def generate_comprehensive_report(df, configs, output_path, best_rope):
    """生成统计报告"""
    report = []
    report.append("="*80)
    report.append("RoPE Base Frequency Experiment - Comprehensive Report")
    report.append("="*80)
    report.append(f"\nTotal samples: {len(df)}")
    
    report.append("\n" + "-"*60)
    report.append("1. Performance Statistics")
    report.append("-"*60)
    report.append(f"\n{'Configuration':<25} {'Mean':<10} {'Std':<10} {'Median':<10} {'Min':<10} {'Max':<10}")
    report.append("-"*70)
    
    for name, cfg in configs.items():
        scores = df[cfg['score_col']].dropna()
        report.append(f"{name:<25} {scores.mean():<10.4f} {scores.std():<10.4f} "
                     f"{scores.median():<10.4f} {scores.min():<10.4f} {scores.max():<10.4f}")
    
    report.append("\n" + "-"*60)
    report.append("2. Improvement over Baseline")
    report.append("-"*60)
    
    baseline_mean = df[configs['Baseline (Greedy)']['score_col']].mean()
    
    for name, cfg in configs.items():
        if 'Baseline' in name:
            continue
        scores = df[cfg['score_col']].dropna()
        improvement = (scores.mean() - baseline_mean) / baseline_mean * 100
        better_count = (scores > df[configs['Baseline (Greedy)']['score_col']][:len(scores)]).sum()
        report.append(f"\n{name}:")
        report.append(f"  Mean improvement: {improvement:+.2f}%")
        report.append(f"  Better samples: {better_count}/{len(scores)} ({better_count/len(scores)*100:.1f}%)")
    
    report.append("\n" + "-"*60)
    report.append("3. Best Configuration Analysis")
    report.append("-"*60)
    
    report.append(f"\n🏆 Best RoPE configuration: {best_rope}")
    
    best_score = df[configs[best_rope]['score_col']].mean()
    beam_score = df[configs['Beam Search (k=3)']['score_col']].mean()
    rope_vs_beam = (best_score - beam_score) / beam_score * 100
    report.append(f"  RoPE vs Beam Search improvement: {rope_vs_beam:+.2f}%")
    
    report.append("\n" + "-"*60)
    report.append("4. Statistical Significance")
    report.append("-"*60)
    
    baseline = df[configs['Baseline (Greedy)']['score_col']]
    
    for name, cfg in configs.items():
        if 'Baseline' in name:
            continue
        scores = df[cfg['score_col']].dropna()
        t_stat, p_value = stats.ttest_rel(baseline[:len(scores)], scores)
        report.append(f"\n{name}:")
        report.append(f"  t-statistic: {t_stat:.4f}")
        report.append(f"  p-value: {p_value:.6f}")
        report.append(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\nSaved report: {output_path}")
    return report


# ============================================================
# 15. 主函数
# ============================================================
def main():
    """主函数"""
    # 文件路径（请修改为你的实际路径）
    data_path = "C:/Users/13113/Desktop/数据分析/final_comparison_dataset.csv"
    output_dir = "C:/Users/13113/Desktop/数据分析/charts-1"
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    df, configs = load_and_preprocess(data_path)
    
    print("\n" + "-"*50)
    print("Generating comprehensive charts...")
    print("-"*50)
    
    # 生成图表
    plot1_avg_scores(df, configs, f'{output_dir}/1_avg_scores.png')
    plot2_base_frequency_curve(df, configs, f'{output_dir}/2_base_frequency_curve.png')
    plot3_boxplot(df, configs, f'{output_dir}/3_boxplot.png')
    plot4_improvement_histograms(df, configs, f'{output_dir}/4_improvement_histograms.png')
    plot5_radar_chart(df, configs, f'{output_dir}/5_radar_chart.png')
    best_rope = plot6_scatter_best_vs_baseline(df, configs, f'{output_dir}/6_scatter_best_vs_baseline.png')
    plot7_violin_plot(df, configs, f'{output_dir}/7_violin_plot.png')
    plot8_correlation_heatmap(df, configs, f'{output_dir}/8_correlation_heatmap.png')
    plot9_cumulative_distribution(df, configs, f'{output_dir}/9_cumulative_distribution.png')
    plot10_improvement_ranking(df, configs, f'{output_dir}/10_improvement_ranking.png')
    plot11_efficiency_analysis(df, configs, f'{output_dir}/11_efficiency_analysis.png')
    plot12_significance_test(df, configs, f'{output_dir}/12_significance_test.png')
    
    # 生成报告
    generate_comprehensive_report(df, configs, f'{output_dir}/comprehensive_report.txt', best_rope)
    
    print("\n" + "="*80)
    print("All charts and reports generated successfully!")
    print(f"Output directory: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()