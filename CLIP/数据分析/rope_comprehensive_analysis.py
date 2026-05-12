"""
rope_comprehensive_analysis.py
RoPE基频调参实验 - 全面可视化分析
覆盖所有配置变量，生成多维度对比图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import make_interp_spline
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
sns.set_style("whitegrid")

# ============================================================
# 1. 数据加载与预处理
# ============================================================
def load_and_preprocess(data_path):
    """加载数据并预处理"""
    df = pd.read_csv(data_path)
    
    print("="*80)
    print("RoPE基频调参实验 - 全面可视化分析")
    print("="*80)
    print(f"总样本数: {len(df)}")
    print(f"数据列: {df.columns.tolist()}")
    
    # 定义各配置
    configs = {
        '基线 (贪心解码)': {'score_col': 'score_baseline', 'caption_col': 'cap_baseline', 'color': '#95a5a6'},
        '束搜索 (Beam=3)': {'score_col': 'score_best_beam', 'caption_col': 'cap_best_beam', 'color': '#3498db'},
        'RoPE base=5000': {'score_col': 'score_rope_5000', 'caption_col': 'cap_rope_5000', 'color': '#2ecc71'},
        'RoPE base=10000': {'score_col': 'score_rope_10000', 'caption_col': 'cap_rope_10000', 'color': '#27ae60'},
        'RoPE base=20000': {'score_col': 'score_rope_20000', 'caption_col': 'cap_rope_20000', 'color': '#f39c12'},
        'RoPE base=100000': {'score_col': 'score_rope_100000', 'caption_col': 'cap_rope_100000', 'color': '#e74c3c'}
    }
    
    # 过滤存在的配置
    existing_configs = {}
    for name, cfg in configs.items():
        if cfg['score_col'] in df.columns:
            existing_configs[name] = cfg
    
    return df, existing_configs


# ============================================================
# 2. 图1：所有配置平均分对比柱状图（带误差线）
# ============================================================
def plot1_avg_scores_with_error(df, configs, save_path):
    """图1：所有配置平均分对比柱状图（带误差线）"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
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
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('平均 CLIP-Score', fontsize=12)
    ax.set_title('各配置CLIP-Score对比（带误差线）', fontsize=14)
    ax.set_ylim([0, max(means) * 1.2])
    ax.axhline(y=means[0], color='gray', linestyle='--', alpha=0.7, label=f'基线: {means[0]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    baseline = means[0]
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(bar.get_x() + bar.get_width()/2, mean + std + 0.005,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        if i > 0:
            improvement = (mean - baseline) / baseline * 100
            color = '#2ecc71' if improvement > 0 else '#e74c3c'
            ax.text(bar.get_x() + bar.get_width()/2, mean - std - 0.02,
                    f'{improvement:+.2f}%', ha='center', va='top', 
                    fontsize=9, color=color, fontweight='bold')
    
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图1已保存: {save_path}")
    plt.close()


# ============================================================
# 3. 图2：基频参数影响曲线（带置信区间）
# ============================================================
def plot2_base_frequency_curve(df, configs, save_path):
    """图2：基频参数影响曲线"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 提取基频和对应分数
    freq_data = []
    for name, cfg in configs.items():
        if 'base=' in name:
            freq = int(name.split('base=')[1])
            scores = df[cfg['score_col']].dropna()
            freq_data.append({
                'freq': freq,
                'mean': scores.mean(),
                'std': scores.std(),
                'ci_95': 1.96 * scores.std() / np.sqrt(len(scores))
            })
    
    freq_data.sort(key=lambda x: x['freq'])
    freqs = [d['freq'] for d in freq_data]
    means = [d['mean'] for d in freq_data]
    cis = [d['ci_95'] for d in freq_data]
    
    # 添加束搜索基线
    beam_mean = df[configs['束搜索 (Beam=3)']['score_col']].mean()
    freqs.insert(0, 0)
    means.insert(0, beam_mean)
    cis.insert(0, 0)
    
    # 绘制曲线
    ax.plot(freqs, means, 'o-', linewidth=2.5, markersize=8, color='#3498db', label='平均分')
    ax.fill_between(freqs, [m - c for m, c in zip(means, cis)], 
                     [m + c for m, c in zip(means, cis)], alpha=0.2, color='#3498db', label='95%置信区间')
    
    ax.set_xscale('log')
    ax.set_xlabel('基频参数 (base)', fontsize=12)
    ax.set_ylabel('平均 CLIP-Score', fontsize=12)
    ax.set_title('RoPE基频参数对性能的影响（含置信区间）', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 标注最优值
    best_idx = np.argmax(means)
    ax.scatter([freqs[best_idx]], [means[best_idx]], s=300, c='red', 
               marker='*', zorder=5, label=f'最优: base={freqs[best_idx]}')
    ax.annotate(f'最优基频: base={freqs[best_idx]}\nscore={means[best_idx]:.4f}',
                xy=(freqs[best_idx], means[best_idx]), xytext=(freqs[best_idx]*2, means[best_idx]-0.02),
                fontsize=10, ha='left', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图2已保存: {save_path}")
    plt.close()


# ============================================================
# 4. 图3：所有配置箱线图分布
# ============================================================
def plot3_boxplot_all(df, configs, save_path):
    """图3：所有配置分数分布箱线图"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
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
        patch.set_alpha(0.7)
    
    ax.set_ylabel('CLIP-Score', fontsize=12)
    ax.set_title('各配置分数分布箱线图', fontsize=14)
    ax.set_ylim([0, 0.5])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图3已保存: {save_path}")
    plt.close()


# ============================================================
# 5. 图4：提升幅度分布直方图（多子图）
# ============================================================
def plot4_improvement_histograms(df, configs, save_path):
    """图4：各配置相对于基线的提升分布"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    baseline = df[configs['基线 (贪心解码)']['score_col']]
    
    for idx, (name, cfg) in enumerate(configs.items()):
        if name == '基线 (贪心解码)':
            continue
        if idx - 1 >= len(axes):
            break
            
        ax = axes[idx - 1]
        scores = df[cfg['score_col']].dropna()
        improvements = scores - baseline[:len(scores)]
        
        n, bins, patches = ax.hist(improvements, bins=30, edgecolor='black', alpha=0.7)
        
        for patch, bin_edge in zip(patches, bins[:-1]):
            patch.set_facecolor('#2ecc71' if bin_edge >= 0 else '#e74c3c')
        
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        ax.axvline(x=improvements.mean(), color='blue', linestyle='-', linewidth=2,
                   label=f'平均: {improvements.mean():.4f}')
        ax.set_xlabel('CLIP-Score 提升值')
        ax.set_ylabel('图片数量')
        ax.set_title(f'{name}\n改进比例: {(improvements > 0).mean()*100:.1f}%')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(len(configs) - 1, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('各配置相对于基线的提升分布', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图4已保存: {save_path}")
    plt.close()


# ============================================================
# 6. 图5：性能雷达图（多维度）
# ============================================================
def plot5_radar_chart(df, configs, save_path):
    """图5：多维度性能雷达图"""
    dimensions = ['平均分', '中位数', '最高分', '最低分', '稳定性(1/标准差)', '效率(分/秒)', '改进比例']
    
    config_scores = {}
    
    baseline_scores = df[configs['基线 (贪心解码)']['score_col']].dropna()
    
    for name, cfg in configs.items():
        scores = df[cfg['score_col']].dropna()
        if len(scores) == 0:
            continue
        
        improvements = scores - baseline_scores[:len(scores)]
        
        config_scores[name] = {
            '平均分': scores.mean(),
            '中位数': scores.median(),
            '最高分': scores.max(),
            '最低分': scores.min(),
            '稳定性(1/标准差)': 1 / scores.std() if scores.std() > 0 else 0,
            '效率(分/秒)': scores.mean() / df['total_time_seconds'].mean() if 'total_time_seconds' in df.columns else scores.mean(),
            '改进比例': (improvements > 0).mean()
        }
    
    # 归一化
    for dim in dimensions:
        dim_values = [config_scores[c][dim] for c in config_scores]
        max_val = max(dim_values) if dim_values else 1
        for c in config_scores:
            config_scores[c][dim] = config_scores[c][dim] / max_val if max_val > 0 else 0
    
    # 绘制雷达图
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
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
    ax.set_title('多维度性能雷达图', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图5已保存: {save_path}")
    plt.close()


# ============================================================
# 7. 图6：散点图矩阵（基线 vs 各配置）
# ============================================================
def plot6_scatter_matrix(df, configs, save_path):
    """图6：基线 vs 各配置散点图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    baseline = df[configs['基线 (贪心解码)']['score_col']]
    
    for idx, (name, cfg) in enumerate(configs.items()):
        if name == '基线 (贪心解码)':
            continue
        if idx - 1 >= len(axes):
            break
            
        ax = axes[idx - 1]
        scores = df[cfg['score_col']].dropna()
        improvements = scores - baseline[:len(scores)]
        colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
        
        ax.scatter(baseline[:len(scores)], scores, c=colors, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
        
        max_val = max(baseline.max(), scores.max())
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='y=x')
        
        z = np.polyfit(baseline[:len(scores)], scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(baseline.min(), baseline.max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=1.5, label=f'趋势 (斜率={z[0]:.3f})')
        
        ax.set_xlabel('基线 CLIP-Score')
        ax.set_ylabel(f'{name} CLIP-Score')
        ax.set_title(f'{name} vs 基线\n改进比例: {(improvements > 0).mean()*100:.1f}%')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    for i in range(len(configs) - 1, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('各配置与基线的散点对比', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图6已保存: {save_path}")
    plt.close()


# ============================================================
# 8. 图7：小提琴图（分布密度）
# ============================================================
def plot7_violin_plot(df, configs, save_path):
    """图7：小提琴图展示分布密度"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
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
        pc.set_alpha(0.7)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('CLIP-Score')
    ax.set_title('各配置分数分布小提琴图')
    ax.set_ylim([0, 0.5])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图7已保存: {save_path}")
    plt.close()


# ============================================================
# 9. 图8：热力图（配置间相关性）
# ============================================================
def plot8_correlation_heatmap(df, configs, save_path):
    """图8：各配置间相关性热力图"""
    # 构建相关性矩阵
    score_cols = [cfg['score_col'] for cfg in configs.values()]
    score_cols = [col for col in score_cols if col in df.columns]
    
    corr_matrix = df[score_cols].corr()
    
    # 重命名列
    name_map = {cfg['score_col']: name for name, cfg in configs.items()}
    corr_matrix = corr_matrix.rename(index=name_map, columns=name_map)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', 
                center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('各配置间CLIP-Score相关性热力图', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图8已保存: {save_path}")
    plt.close()


# ============================================================
# 10. 图9：累计分布曲线
# ============================================================
def plot9_cumulative_distribution(df, configs, save_path):
    """图9：各配置累计分布曲线"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for name, cfg in configs.items():
        scores = df[cfg['score_col']].dropna()
        sorted_scores = np.sort(scores)
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        ax.plot(sorted_scores, cumulative, linewidth=2, label=name, color=cfg['color'])
    
    ax.set_xlabel('CLIP-Score', fontsize=12)
    ax.set_ylabel('累计比例', fontsize=12)
    ax.set_title('各配置CLIP-Score累计分布曲线', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图9已保存: {save_path}")
    plt.close()


# ============================================================
# 11. 图10：性能提升排序图
# ============================================================
def plot10_improvement_ranking(df, configs, save_path):
    """图10：各配置性能提升排序"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    baseline = df[configs['基线 (贪心解码)']['score_col']].mean()
    
    improvements = []
    names = []
    colors = []
    
    for name, cfg in configs.items():
        if name == '基线 (贪心解码)':
            continue
        mean_score = df[cfg['score_col']].mean()
        improvement = (mean_score - baseline) / baseline * 100
        improvements.append(improvement)
        names.append(name)
        colors.append(cfg['color'])
    
    # 按提升排序
    sorted_idx = np.argsort(improvements)[::-1]
    names = [names[i] for i in sorted_idx]
    improvements = [improvements[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]
    
    bars = ax.barh(names, improvements, color=colors, edgecolor='black')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('相对于基线的提升百分比 (%)')
    ax.set_title('各配置性能提升排序')
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{imp:+.2f}%', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图10已保存: {save_path}")
    plt.close()


# ============================================================
# 12. 图11：效率分析图（分数 vs 时间）
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
        mean_time = df['total_time_seconds'].mean() if 'total_time_seconds' in df.columns else 0.2
        times.append(mean_time)
        scores.append(mean_score)
        names.append(name)
        colors.append(cfg['color'])
    
    # 绘制气泡图
    sizes = [s * 500 for s in scores]
    scatter = ax.scatter(times, scores, s=sizes, c=colors, alpha=0.6, edgecolors='black', linewidth=1.5)
    
    # 添加标签
    for i, name in enumerate(names):
        ax.annotate(name, (times[i], scores[i]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9, ha='left')
    
    ax.set_xlabel('平均耗时 (秒)', fontsize=12)
    ax.set_ylabel('平均 CLIP-Score', fontsize=12)
    ax.set_title('效率分析：分数 vs 时间', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图11已保存: {save_path}")
    plt.close()


# ============================================================
# 13. 图12：统计显著性检验图
# ============================================================
def plot12_significance_test(df, configs, save_path):
    """图12：统计显著性检验"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    baseline = df[configs['基线 (贪心解码)']['score_col']]
    
    p_values = []
    names = []
    colors = []
    
    for name, cfg in configs.items():
        if name == '基线 (贪心解码)':
            continue
        scores = df[cfg['score_col']].dropna()
        _, p_value = stats.ttest_rel(baseline[:len(scores)], scores)
        p_values.append(p_value)
        names.append(name)
        colors.append(cfg['color'])
    
    # 绘制p值
    bars = ax.bar(names, p_values, color=colors, edgecolor='black')
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='显著性阈值 (p=0.05)')
    ax.set_ylabel('p值', fontsize=12)
    ax.set_title('统计显著性检验（配对t检验）', fontsize=14)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, p in zip(bars, p_values):
        is_significant = p < 0.05
        label = '显著' if is_significant else '不显著'
        color = '#2ecc71' if is_significant else '#e74c3c'
        ax.text(bar.get_x() + bar.get_width()/2, p * 1.5,
                f'{label}\n(p={p:.4f})', ha='center', va='bottom', 
                fontsize=9, color=color)
    
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图12已保存: {save_path}")
    plt.close()


# ============================================================
# 14. 生成统计报告
# ============================================================
def generate_comprehensive_report(df, configs, output_path):
    """生成全面统计报告"""
    report = []
    report.append("="*80)
    report.append("RoPE基频调参实验 - 全面统计报告")
    report.append("="*80)
    report.append(f"\n总样本数: {len(df)}")
    report.append(f"总耗时: {df['total_time_seconds'].sum():.2f} 秒 ({df['total_time_seconds'].sum()/3600:.2f} 小时)")
    
    report.append("\n" + "-"*60)
    report.append("1. 各配置性能统计")
    report.append("-"*60)
    report.append(f"\n{'配置':<20} {'平均分':<10} {'中位数':<10} {'标准差':<10} {'最高分':<10} {'最低分':<10}")
    report.append("-"*70)
    
    baseline_mean = df[configs['基线 (贪心解码)']['score_col']].mean()
    
    for name, cfg in configs.items():
        scores = df[cfg['score_col']].dropna()
        report.append(f"{name:<20} {scores.mean():<10.4f} {scores.median():<10.4f} "
                     f"{scores.std():<10.4f} {scores.max():<10.4f} {scores.min():<10.4f}")
    
    report.append("\n" + "-"*60)
    report.append("2. 相对于基线的提升")
    report.append("-"*60)
    
    for name, cfg in configs.items():
        if name == '基线 (贪心解码)':
            continue
        scores = df[cfg['score_col']].dropna()
        improvement = (scores.mean() - baseline_mean) / baseline_mean * 100
        better_count = (scores > df[configs['基线 (贪心解码)']['score_col']][:len(scores)]).sum()
        report.append(f"\n{name}:")
        report.append(f"  平均提升: {improvement:+.2f}%")
        report.append(f"  改进图片数: {better_count}/{len(scores)} ({better_count/len(scores)*100:.1f}%)")
    
    report.append("\n" + "-"*60)
    report.append("3. 基频参数最优分析")
    report.append("-"*60)
    
    freq_scores = {}
    for name, cfg in configs.items():
        if 'base=' in name:
            freq = int(name.split('base=')[1])
            freq_scores[freq] = df[cfg['score_col']].mean()
    
    if freq_scores:
        best_freq = max(freq_scores, key=freq_scores.get)
        report.append(f"\n各基频平均分:")
        for freq, score in sorted(freq_scores.items()):
            report.append(f"  base={freq}: {score:.4f}")
        report.append(f"\n🏆 最优基频: base={best_freq} (score={freq_scores[best_freq]:.4f})")
        
        # 与束搜索对比
        beam_score = df[configs['束搜索 (Beam=3)']['score_col']].mean()
        rope_improvement = (freq_scores[best_freq] - beam_score) / beam_score * 100
        report.append(f"\nRoPE相比纯束搜索的提升: {rope_improvement:+.2f}%")
    
    report.append("\n" + "-"*60)
    report.append("4. 统计显著性检验")
    report.append("-"*60)
    
    baseline = df[configs['基线 (贪心解码)']['score_col']]
    
    for name, cfg in configs.items():
        if name == '基线 (贪心解码)':
            continue
        scores = df[cfg['score_col']].dropna()
        t_stat, p_value = stats.ttest_rel(baseline[:len(scores)], scores)
        report.append(f"\n{name} vs 基线:")
        report.append(f"  t统计量: {t_stat:.4f}")
        report.append(f"  p值: {p_value:.6f}")
        report.append(f"  差异显著性: {'显著 (p < 0.05)' if p_value < 0.05 else '不显著'}")
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\n✅ 统计报告已保存: {output_path}")
    return report


# ============================================================
# 15. 主函数
# ============================================================
def main():
    """主函数"""
    # 文件路径
    data_path = "C:/Users/13113/Desktop/数据分析/final_comparison_dataset.csv"
    output_dir = "C:/Users/13113/Desktop/数据分析/charts-1"
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    df, configs = load_and_preprocess(data_path)
    
    # 生成所有图表
    print("\n开始生成全面可视化图表...")
    print("-"*50)
    
    # 图表清单
    charts = [
        (plot1_avg_scores_with_error, "1_avg_scores_with_error.png", "各配置平均分对比"),
        (plot2_base_frequency_curve, "2_base_frequency_curve.png", "基频参数影响曲线"),
        (plot3_boxplot_all, "3_boxplot_all.png", "箱线图分布"),
        (plot4_improvement_histograms, "4_improvement_histograms.png", "提升分布直方图"),
        (plot5_radar_chart, "5_radar_chart.png", "多维度雷达图"),
        (plot6_scatter_matrix, "6_scatter_matrix.png", "散点图矩阵"),
        (plot7_violin_plot, "7_violin_plot.png", "小提琴图"),
        (plot8_correlation_heatmap, "8_correlation_heatmap.png", "相关性热力图"),
        (plot9_cumulative_distribution, "9_cumulative_distribution.png", "累计分布曲线"),
        (plot10_improvement_ranking, "10_improvement_ranking.png", "性能提升排序"),
        (plot11_efficiency_analysis, "11_efficiency_analysis.png", "效率分析"),
        (plot12_significance_test, "12_significance_test.png", "显著性检验")
    ]
    
    for plot_func, filename, description in charts:
        print(f"\n正在生成: {description}")
        plot_func(df, configs, f'{output_dir}/{filename}')
    
    # 生成统计报告
    generate_comprehensive_report(df, configs, f'{output_dir}/comprehensive_report.txt')
    
    print("\n" + "="*80)
    print("🎉 所有图表生成完成！")
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 共生成 {len(charts)} 个图表 + 1 个统计报告")
    print("="*80)


if __name__ == "__main__":
    main()