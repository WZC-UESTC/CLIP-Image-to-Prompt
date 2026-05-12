"""
ablation_analysis.py
基于消融实验数据生成对比图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和美化样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 读取数据
df = pd.read_csv('C:/Users/13113/Desktop/数据分析/final_unified_ablation2.csv')

print("="*60)
print("消融实验数据基本信息")
print("="*60)
print(f"总样本数: {len(df)}")
print(f"列名: {df.columns.tolist()}")
print(f"\n数据前5行:")
print(df.head())

# 定义模型配置
configs = {
    'baseline': {'col_score': 'score_baseline', 'col_cap': 'cap_baseline', 'name': '基线 (Beam Search)'},
    'beam': {'col_score': 'score_best_beam', 'col_cap': 'cap_best_beam', 'name': 'Beam Search优化'},
    'rope_10000': {'col_score': 'score_rope_10000', 'col_cap': 'cap_rope_10000', 'name': 'RoPE 10000'},
    'rope_5000': {'col_score': 'score_rope_5000', 'col_cap': 'cap_rope_5000', 'name': 'RoPE 5000'},
    'rope_20000': {'col_score': 'score_rope_20000', 'col_cap': 'cap_rope_20000', 'name': 'RoPE 20000'},
    'rope_100000': {'col_score': 'score_rope_100000', 'col_cap': 'cap_rope_100000', 'name': 'RoPE 100000'}
}

# 计算总耗时（如果存在total_time_seconds列）
if 'total_time_seconds' in df.columns:
    total_time = df['total_time_seconds'].iloc[0] if len(df) > 0 else 0
    print(f"\n总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")

print("\n" + "="*60)
print("各配置分数统计")
print("="*60)
for key, cfg in configs.items():
    scores = df[cfg['col_score']].dropna()
    print(f"\n{cfg['name']}:")
    print(f"  平均分: {scores.mean():.4f}")
    print(f"  中位数: {scores.median():.4f}")
    print(f"  标准差: {scores.std():.4f}")
    print(f"  最高分: {scores.max():.4f}")
    print(f"  最低分: {scores.min():.4f}")

# ============================================================
# 图1：所有配置箱线图对比
# ============================================================
def plot_all_boxplots(df, configs, save_path):
    """图1：所有配置分数分布箱线图"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    data_to_plot = []
    labels = []
    colors_list = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, (key, cfg) in enumerate(configs.items()):
        scores = df[cfg['col_score']].dropna()
        data_to_plot.append(scores)
        labels.append(cfg['name'])
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                     showmeans=True, meanline=True, meanprops=dict(color='red', linewidth=2))
    
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('CLIP-Score', fontsize=12)
    ax.set_title('各配置CLIP-Score分布对比', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图1已保存: {save_path}")
    plt.close()

# ============================================================
# 图2：相对基线提升热图
# ============================================================
def plot_improvement_heatmap(df, configs, save_path):
    """图2：各配置相对于基线的提升热图"""
    baseline_scores = df[configs['baseline']['col_score']].values
    
    improvement_data = {}
    for key, cfg in configs.items():
        if key != 'baseline':
            scores = df[cfg['col_score']].values
            improvement = scores - baseline_scores
            improvement_data[cfg['name']] = improvement
    
    # 计算统计指标
    stats_data = []
    for name, imp in improvement_data.items():
        stats_data.append({
            '配置': name,
            '平均提升': np.mean(imp),
            '中位数提升': np.median(imp),
            '改进比例': (imp > 0).mean() * 100,
            '最佳提升': np.max(imp),
            '最差提升': np.min(imp)
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 子图1：平均提升和中位数提升
    ax1 = axes[0]
    x = np.arange(len(stats_df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, stats_df['平均提升'], width, label='平均提升', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, stats_df['中位数提升'], width, label='中位数提升', color='#3498db', alpha=0.8)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats_df['配置'], rotation=45, ha='right')
    ax1.set_ylabel('CLIP-Score提升值')
    ax1.set_title('各配置相对基线提升统计')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars1, stats_df['平均提升']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, stats_df['中位数提升']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 子图2：改进比例条形图
    ax2 = axes[1]
    colors_bar = ['#2ecc71' if x >= 50 else '#e74c3c' for x in stats_df['改进比例']]
    bars = ax2.bar(stats_df['配置'], stats_df['改进比例'], color=colors_bar, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('改进比例 (%)')
    ax2.set_title('各配置相对基线的改进比例')
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50%基准线')
    ax2.set_ylim([0, 100])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    # 添加数值标签
    for bar, val in zip(bars, stats_df['改进比例']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图2已保存: {save_path}")
    plt.close()
    
    return stats_df

# ============================================================
# 图3：所有配置散点图矩阵
# ============================================================
def plot_scatter_matrix(df, configs, save_path):
    """图3：各配置与基线的散点对比"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    baseline_scores = df[configs['baseline']['col_score']].values
    
    plot_idx = 0
    for key, cfg in configs.items():
        if key != 'baseline':
            ax = axes[plot_idx]
            scores = df[cfg['col_score']].values
            improvements = scores - baseline_scores
            
            colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
            ax.scatter(baseline_scores, scores, c=colors, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
            
            # 添加对角线
            max_val = max(baseline_scores.max(), scores.max())
            ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.7, label='y=x')
            
            # 添加趋势线
            z = np.polyfit(baseline_scores, scores, 1)
            p = np.poly1d(z)
            ax.plot(baseline_scores, p(baseline_scores), 'r-', linewidth=2, alpha=0.7, label=f'趋势 (k={z[0]:.3f})')
            
            ax.set_xlabel('基线 CLIP-Score', fontsize=10)
            ax.set_ylabel(f'{cfg["name"]} CLIP-Score', fontsize=10)
            ax.set_title(f'{cfg["name"]} vs 基线', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # 添加统计信息
            better_count = (improvements > 0).sum()
            total = len(improvements)
            text = f'改进: {better_count}/{total}\n({better_count/total*100:.1f}%)'
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plot_idx += 1
    
    # 隐藏多余的子图
    for i in range(plot_idx, 6):
        axes[i].set_visible(False)
    
    plt.suptitle('各配置与基线对比散点图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图3已保存: {save_path}")
    plt.close()

# ============================================================
# 图4：性能对比雷达图
# ============================================================
def plot_radar_comparison(df, configs, save_path):
    """图4：多维度性能对比雷达图"""
    # 计算各项指标
    metrics = {}
    
    for key, cfg in configs.items():
        scores = df[cfg['col_score']].dropna()
        metrics[cfg['name']] = {
            '平均分': scores.mean(),
            '中位数': scores.median(),
            '最高分': scores.max(),
            '稳定性(1/标准差)': 1 / scores.std() if scores.std() > 0 else 0,
            '最低分': scores.min(),
            'Q3四分位': scores.quantile(0.75)
        }
    
    # 归一化
    categories = list(list(metrics.values())[0].keys())
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    colors_list = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, (name, values) in enumerate(metrics.items()):
        # 归一化
        normalized = []
        for j, cat in enumerate(categories):
            all_vals = [metrics[m][cat] for m in metrics.keys()]
            max_val = max(all_vals)
            min_val = min(all_vals)
            if max_val > min_val:
                norm_val = (values[cat] - min_val) / (max_val - min_val)
            else:
                norm_val = 0.5
            normalized.append(norm_val)
        
        normalized += normalized[:1]
        
        ax.plot(angles, normalized, 'o-', linewidth=2, label=name, color=colors_list[i], alpha=0.8)
        ax.fill(angles, normalized, alpha=0.15, color=colors_list[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('各配置多维度性能对比雷达图', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图4已保存: {save_path}")
    plt.close()

# ============================================================
# 图5：改进幅度分布小提琴图
# ============================================================
def plot_violin_improvements(df, configs, save_path):
    """图5：各配置改进幅度分布小提琴图"""
    baseline_scores = df[configs['baseline']['col_score']].values
    
    improvements_data = []
    labels = []
    
    for key, cfg in configs.items():
        if key != 'baseline':
            scores = df[cfg['col_score']].values
            improvements = scores - baseline_scores
            improvements_data.append(improvements)
            labels.append(cfg['name'])
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 绘制小提琴图
    parts = ax.violinplot(improvements_data, showmeans=True, showmedians=True)
    
    # 设置颜色
    colors_list = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_list[i])
        pc.set_alpha(0.7)
    
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('CLIP-Score提升值', fontsize=12)
    ax.set_title('各配置相对基线改进幅度分布', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='零提升线')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # 添加均值标注
    for i, improvements in enumerate(improvements_data):
        mean_val = np.mean(improvements)
        ax.text(i + 1, mean_val + 0.002, f'μ={mean_val:.4f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图5已保存: {save_path}")
    plt.close()

# ============================================================
# 图6：排名热图
# ============================================================
def plot_ranking_heatmap(df, configs, save_path):
    """图6：各样本在不同配置下的排名热图"""
    # 收集所有分数
    all_scores = []
    labels = []
    
    for key, cfg in configs.items():
        scores = df[cfg['col_score']].values
        all_scores.append(scores)
        labels.append(cfg['name'])
    
    all_scores = np.array(all_scores).T
    
    # 计算排名（分数越高排名越好）
    rankings = np.argsort(-all_scores, axis=1) + 1
    
    # 随机采样100个样本进行可视化
    np.random.seed(42)
    sample_indices = np.random.choice(len(df), min(100, len(df)), replace=False)
    sample_rankings = rankings[sample_indices]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    im = ax.imshow(sample_rankings.T, aspect='auto', cmap='RdYlGn_r', 
                   interpolation='nearest', vmin=1, vmax=len(configs))
    
    ax.set_xlabel('样本编号 (随机采样)', fontsize=12)
    ax.set_ylabel('配置', fontsize=12)
    ax.set_title('各样本在不同配置下的性能排名热图 (1=最佳)', fontsize=14, fontweight='bold')
    
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('排名', fontsize=10)
    
    # 计算每个配置的平均排名
    avg_rankings = rankings.mean(axis=0)
    print("\n各配置平均排名:")
    for label, avg_rank in zip(labels, avg_rankings):
        print(f"  {label}: {avg_rank:.2f}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图6已保存: {save_path}")
    plt.close()
    
    return avg_rankings

# ============================================================
# 图7：最佳配置分布饼图
# ============================================================
def plot_best_config_distribution(df, configs, save_path):
    """图7：每个样本的最佳配置分布"""
    all_scores = []
    labels = []
    
    for key, cfg in configs.items():
        scores = df[cfg['col_score']].values
        all_scores.append(scores)
        labels.append(cfg['name'])
    
    all_scores = np.array(all_scores).T
    
    # 找出每个样本的最佳配置
    best_indices = np.argmax(all_scores, axis=1)
    best_counts = np.bincount(best_indices, minlength=len(labels))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 子图1：饼图
    ax1 = axes[0]
    colors_list = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    explode = [0.05 if count == best_counts.max() else 0 for count in best_counts]
    
    wedges, texts, autotexts = ax1.pie(best_counts, labels=labels, autopct='%1.1f%%',
                                        colors=colors_list, explode=explode, shadow=True)
    ax1.set_title('各配置成为最佳的比例', fontsize=12, fontweight='bold')
    
    # 子图2：条形图
    ax2 = axes[1]
    bars = ax2.bar(labels, best_counts, color=colors_list, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('配置', fontsize=11)
    ax2.set_ylabel('成为最佳的次数', fontsize=11)
    ax2.set_title('各配置成为最佳的次数统计', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, count in zip(bars, best_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图7已保存: {save_path}")
    plt.close()
    
    return best_counts

# ============================================================
# 图8：相关性矩阵热图
# ============================================================
def plot_correlation_matrix(df, configs, save_path):
    """图8：各配置分数相关性矩阵"""
    # 构建相关性矩阵
    score_df = pd.DataFrame()
    for key, cfg in configs.items():
        score_df[cfg['name']] = df[cfg['col_score']]
    
    corr_matrix = score_df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热图
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    # 设置标签
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr_matrix.columns)
    
    ax.set_title('各配置CLIP-Score相关性矩阵', fontsize=14, fontweight='bold', pad=20)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('相关系数', fontsize=10)
    
    # 添加相关系数文本
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white",
                          fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图8已保存: {save_path}")
    plt.close()
    
    return corr_matrix

# ============================================================
# 生成统计报告
# ============================================================
def generate_comprehensive_report(df, configs, stats_df, avg_rankings, best_counts, corr_matrix, save_path):
    """生成完整统计报告"""
    report = []
    report.append("="*80)
    report.append("消融实验完整统计报告")
    report.append("="*80)
    
    report.append(f"\n【实验概况】")
    report.append(f"  总样本数: {len(df)}")
    if 'total_time_seconds' in df.columns:
        total_time = df['total_time_seconds'].iloc[0]
        report.append(f"  总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    
    report.append(f"\n【各配置详细统计】")
    for key, cfg in configs.items():
        scores = df[cfg['col_score']].dropna()
        report.append(f"\n  {cfg['name']}:")
        report.append(f"    平均分: {scores.mean():.6f}")
        report.append(f"    中位数: {scores.median():.6f}")
        report.append(f"    标准差: {scores.std():.6f}")
        report.append(f"    最高分: {scores.max():.6f}")
        report.append(f"    最低分: {scores.min():.6f}")
        report.append(f"    95%置信区间: [{scores.mean() - 1.96*scores.std()/np.sqrt(len(scores)):.6f}, "
                     f"{scores.mean() + 1.96*scores.std()/np.sqrt(len(scores)):.6f}]")
    
    report.append(f"\n【相对基线提升统计】")
    for _, row in stats_df.iterrows():
        report.append(f"\n  {row['配置']}:")
        report.append(f"    平均提升: {row['平均提升']:.6f}")
        report.append(f"    中位数提升: {row['中位数提升']:.6f}")
        report.append(f"    改进比例: {row['改进比例']:.2f}%")
        report.append(f"    最佳提升: {row['最佳提升']:.6f}")
        report.append(f"    最差提升: {row['最差提升']:.6f}")
    
    report.append(f"\n【排名分析】")
    for label, avg_rank in zip([c['name'] for c in configs.values()], avg_rankings):
        report.append(f"  {label}: 平均排名 = {avg_rank:.2f}")
    
    report.append(f"\n【最佳配置统计】")
    for label, count in zip([c['name'] for c in configs.values()], best_counts):
        report.append(f"  {label}: 成为最佳 {count} 次 ({count/len(df)*100:.1f}%)")
    
    report.append(f"\n【相关性分析】")
    report.append(f"  最强正相关: {corr_matrix.unstack().sort_values(ascending=False).iloc[1]}")
    report.append(f"  最弱相关性: {corr_matrix.unstack().sort_values().iloc[0]}")
    
    report.append(f"\n【统计检验】")
    baseline_scores = df[configs['baseline']['col_score']].values
    for key, cfg in configs.items():
        if key != 'baseline':
            scores = df[cfg['col_score']].values
            t_stat, p_value = stats.ttest_rel(baseline_scores, scores)
            report.append(f"\n  {cfg['name']} vs 基线:")
            report.append(f"    配对t检验 p值: {p_value:.6f}")
            report.append(f"    差异显著性: {'显著' if p_value < 0.05 else '不显著'}")
    
    report.append(f"\n【最佳配置推荐】")
    best_config = stats_df.loc[stats_df['平均提升'].idxmax(), '配置']
    best_improvement = stats_df['平均提升'].max()
    report.append(f"  基于平均提升，推荐使用: {best_config}")
    report.append(f"  平均提升: {best_improvement:.6f}")
    
    # 保存报告
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\n✅ 统计报告已保存: {save_path}")
    
    # 打印到控制台
    print("\n" + "="*80)
    for line in report[:50]:  # 打印前50行
        print(line)
    
    return report

# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    output_dir = 'C:/Users/13113/Desktop/数据分析/ablation_charts'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n开始生成消融实验对比图表...")
    print("="*60)
    
    # 生成所有图表
    plot_all_boxplots(df, configs, f'{output_dir}/1_all_boxplots.png')
    stats_df = plot_improvement_heatmap(df, configs, f'{output_dir}/2_improvement_heatmap.png')
    plot_scatter_matrix(df, configs, f'{output_dir}/3_scatter_matrix.png')
    plot_radar_comparison(df, configs, f'{output_dir}/4_radar_comparison.png')
    plot_violin_improvements(df, configs, f'{output_dir}/5_violin_improvements.png')
    avg_rankings = plot_ranking_heatmap(df, configs, f'{output_dir}/6_ranking_heatmap.png')
    best_counts = plot_best_config_distribution(df, configs, f'{output_dir}/7_best_config_distribution.png')
    corr_matrix = plot_correlation_matrix(df, configs, f'{output_dir}/8_correlation_matrix.png')
    
    # 生成统计报告
    generate_comprehensive_report(df, configs, stats_df, avg_rankings, best_counts, corr_matrix, 
                                  f'{output_dir}/ablation_report.txt')
    
    print("\n" + "="*60)
    print("所有图表生成完成！")
    print(f"输出目录: {output_dir}")
    print("="*60)
    
    # 输出关键发现
    print("\n【关键发现摘要】")
    print("-"*60)
    
    baseline_mean = df['score_baseline'].mean()
    for key, cfg in configs.items():
        if key != 'baseline':
            mean_score = df[cfg['col_score']].mean()
            improvement = mean_score - baseline_mean
            better_pct = (df[cfg['col_score']] > df['score_baseline']).mean() * 100
            print(f"{cfg['name']}: 平均分={mean_score:.4f}, 提升={improvement:+.4f}, 改进比例={better_pct:.1f}%")
    
    best_config = max([(cfg['name'], df[cfg['col_score']].mean()) for key, cfg in configs.items() if key != 'baseline'], 
                      key=lambda x: x[1])
    print(f"\n🏆 最佳配置: {best_config[0]} (平均分: {best_config[1]:.4f})")