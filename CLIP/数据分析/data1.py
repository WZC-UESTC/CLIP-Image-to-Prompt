"""
ablation_analysis_enhanced.py
消融实验数据可视化分析 - 增强版
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

# 读取数据 - 请修改为您的实际路径
df = pd.read_csv('C:/Users/13113/Desktop/数据分析/final_unified_ablation2.csv')

# 处理缺失值：将空字符串和NaN替换为None，便于后续处理
df = df.replace(r'^\s*$', np.nan, regex=True)  # 空字符串替换为NaN
df = df.replace('nan', np.nan)  # 字符串'nan'替换为NaN

print("="*60)
print("消融实验数据基本信息")
print("="*60)
print(f"总样本数: {len(df)}")
print(f"数据形状: {df.shape}")
print(f"\n列名: {df.columns.tolist()}")

# 检查各列缺失值情况
print("\n各列缺失值统计:")
for col in df.columns:
    missing_count = df[col].isna().sum()
    if missing_count > 0:
        print(f"  {col}: {missing_count} 个缺失值 ({missing_count/len(df)*100:.1f}%)")

# 定义模型配置
configs = {
    'baseline': {'col_score': 'score_baseline', 'col_cap': 'cap_baseline', 'name': '基线 (Beam Search)'},
    'beam': {'col_score': 'score_best_beam', 'col_cap': 'cap_best_beam', 'name': 'Beam Search优化'},
    'rope_10000': {'col_score': 'score_rope_10000', 'col_cap': 'cap_rope_10000', 'name': 'RoPE 10000'},
    'rope_5000': {'col_score': 'score_rope_5000', 'col_cap': 'cap_rope_5000', 'name': 'RoPE 5000'},
    'rope_20000': {'col_score': 'score_rope_20000', 'col_cap': 'cap_rope_20000', 'name': 'RoPE 20000'},
    'rope_100000': {'col_score': 'score_rope_100000', 'col_cap': 'cap_rope_100000', 'name': 'RoPE 100000'}
}

# 清理数据：删除任何配置中分数为空的样本
print("\n数据清理: 删除分数缺失的样本...")
original_len = len(df)
for key, cfg in configs.items():
    df = df[df[cfg['col_score']].notna()]
print(f"  清理后样本数: {len(df)} (删除了 {original_len - len(df)} 个无效样本)")

# 重置索引
df = df.reset_index(drop=True)

print("\n" + "="*60)
print("各配置分数统计")
print("="*60)
for key, cfg in configs.items():
    scores = df[cfg['col_score']].dropna()
    print(f"\n{cfg['name']}:")
    print(f"  有效样本数: {len(scores)}")
    print(f"  平均分: {scores.mean():.4f}")
    print(f"  中位数: {scores.median():.4f}")
    print(f"  标准差: {scores.std():.4f}")
    print(f"  最高分: {scores.max():.4f}")
    print(f"  最低分: {scores.min():.4f}")

# ============================================================
# 图1：所有配置箱线图对比
# 说明：展示6种配置的CLIP-Score分布情况，包括中位数、四分位数和异常值
# ============================================================
def plot_all_boxplots(df, configs, save_path):
    """图1：所有配置分数分布箱线图"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    data_to_plot = []
    labels = []
    colors_list = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, (key, cfg) in enumerate(configs.items()):
        scores = df[cfg['col_score']].dropna()
        data_to_plot.append(scores.values)
        labels.append(cfg['name'])
    
    # 绘制箱线图
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                     showmeans=True, meanline=True, 
                     meanprops=dict(color='red', linewidth=2, label='均值'),
                     medianprops=dict(color='darkblue', linewidth=2, label='中位数'))
    
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('CLIP-Score', fontsize=14)
    ax.set_xlabel('配置', fontsize=14)
    ax.set_title('图1：各配置CLIP-Score分布对比', fontsize=16, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=30, ha='right', fontsize=11)
    ax.legend(loc='upper left')
    
    # 添加注释说明
    ax.text(0.02, 0.98, '箱体：Q1-Q3 | 横线：中位数 | 菱形：均值 | 须线：1.5倍IQR', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图1已保存: {save_path}")
    print("   📊 图1说明：箱线图展示了6种配置的CLIP-Score分布，可以直观比较各配置的集中趋势和离散程度")
    plt.close()

# ============================================================
# 图2：相对基线提升热图
# 说明：展示各配置相对于基线的平均提升、中位数提升和改进比例
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
            '最差提升': np.min(imp),
            '标准差提升': np.std(imp)
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 子图1：平均提升和中位数提升
    ax1 = axes[0]
    x = np.arange(len(stats_df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, stats_df['平均提升'], width, label='平均提升', 
                     color='#2ecc71', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, stats_df['中位数提升'], width, label='中位数提升', 
                     color='#3498db', alpha=0.8, edgecolor='black')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats_df['配置'], rotation=30, ha='right', fontsize=10)
    ax1.set_ylabel('CLIP-Score提升值', fontsize=12)
    ax1.set_title('各配置相对基线提升统计', fontsize=13, fontweight='bold')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='零提升线')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars1, stats_df['平均提升']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.001 if val >= 0 else -0.003),
                f'{val:.4f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, stats_df['中位数提升']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.001 if val >= 0 else -0.003),
                f'{val:.4f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
    
    # 子图2：改进比例条形图
    ax2 = axes[1]
    colors_bar = ['#2ecc71' if x >= 50 else '#e74c3c' for x in stats_df['改进比例']]
    bars = ax2.bar(stats_df['配置'], stats_df['改进比例'], color=colors_bar, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('改进比例 (%)', fontsize=12)
    ax2.set_xlabel('配置', fontsize=12)
    ax2.set_title('各配置相对基线的改进比例', fontsize=13, fontweight='bold')
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50%基准线')
    ax2.set_ylim([0, 100])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=30, ha='right', fontsize=10)
    
    # 添加数值标签
    for bar, val in zip(bars, stats_df['改进比例']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('图2：各配置相对基线提升分析', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图2已保存: {save_path}")
    print("   📊 图2说明：左图展示提升幅度，右图展示改进样本比例，正值为优于基线")
    plt.close()
    
    return stats_df

# ============================================================
# 图3：所有配置散点图矩阵
# 说明：每个子图展示一种配置与基线的分数对应关系，点在上半部分表示优于基线
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
            
            # 颜色：绿色表示改进，红色表示下降
            colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
            sizes = [50 if abs(imp) > 0.05 else 30 for imp in improvements]  # 突出大幅改进的点
            
            scatter = ax.scatter(baseline_scores, scores, c=colors, alpha=0.6, 
                                s=sizes, edgecolors='black', linewidth=0.5)
            
            # 添加对角线
            max_val = max(baseline_scores.max(), scores.max())
            min_val = min(baseline_scores.min(), scores.min())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.7, label='y=x (持平线)')
            
            # 添加趋势线
            z = np.polyfit(baseline_scores, scores, 1)
            p = np.poly1d(z)
            ax.plot(np.sort(baseline_scores), p(np.sort(baseline_scores)), 'r-', linewidth=2, alpha=0.7, 
                   label=f'趋势线 (斜率={z[0]:.3f})')
            
            ax.set_xlabel('基线 CLIP-Score', fontsize=11)
            ax.set_ylabel(f'{cfg["name"]} CLIP-Score', fontsize=11)
            ax.set_title(f'{cfg["name"]} vs 基线', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', fontsize=8)
            
            # 添加统计信息
            better_count = (improvements > 0).sum()
            total = len(improvements)
            mean_imp = improvements.mean()
            text = f'改进样本: {better_count}/{total} ({better_count/total*100:.1f}%)\n平均提升: {mean_imp:.4f}'
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            plot_idx += 1
    
    # 隐藏多余的子图
    for i in range(plot_idx, 6):
        axes[i].set_visible(False)
    
    plt.suptitle('图3：各配置与基线对比散点图\n(绿色点表示优于基线，红色点表示劣于基线)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图3已保存: {save_path}")
    print("   📊 图3说明：每个点代表一个样本，点在对角线上方表示该配置优于基线")
    plt.close()

# ============================================================
# 图4：性能对比雷达图
# 说明：多维度综合对比各配置性能，面积越大表示综合性能越好
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
            '最低分': scores.min(),
            '稳定性(1/标准差)': 1 / scores.std() if scores.std() > 0 else 0,
            'Q3四分位': scores.quantile(0.75),
            'Q1四分位': scores.quantile(0.25)
        }
    
    # 选择用于雷达图的指标
    radar_metrics = ['平均分', '中位数', '最高分', '稳定性(1/标准差)', 'Q3四分位']
    
    # 归一化
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    colors_list = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, (name, values) in enumerate(metrics.items()):
        # 归一化
        normalized = []
        for j, cat in enumerate(radar_metrics):
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
    ax.set_xticklabels(radar_metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.set_title('图4：各配置多维度性能对比雷达图\n(数值越大表示性能越好，面积越大综合性能越强)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图4已保存: {save_path}")
    print("   📊 图4说明：雷达图从多个维度对比各配置，覆盖面积越大表示综合性能越好")
    plt.close()

# ============================================================
# 图5：改进幅度分布小提琴图
# 说明：展示各配置相对于基线提升幅度的分布形状，宽度越宽表示该区域数据越集中
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
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    # 设置均值和 median 样式
    parts['cmeans'].set_color('darkred')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('darkblue')
    parts['cmedians'].set_linewidth(2)
    
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=11)
    ax.set_ylabel('CLIP-Score提升值', fontsize=13)
    ax.set_xlabel('配置', fontsize=13)
    ax.set_title('图5：各配置相对基线改进幅度分布\n(小提琴图宽度表示数据密度，横线为中位数，红点为均值)', 
                 fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='零提升线')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right')
    
    # 添加均值标注
    for i, improvements in enumerate(improvements_data):
        mean_val = np.mean(improvements)
        median_val = np.median(improvements)
        ax.text(i + 1, mean_val + (0.003 if mean_val >= 0 else -0.008), 
                f'均值: {mean_val:.4f}', ha='center', va='bottom' if mean_val >= 0 else 'top', 
                fontsize=9, fontweight='bold', color='darkred')
        ax.text(i + 1, median_val + (0.003 if median_val >= 0 else -0.008),
                f'中位数: {median_val:.4f}', ha='center', va='bottom' if median_val >= 0 else 'top',
                fontsize=8, color='darkblue')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图5已保存: {save_path}")
    print("   📊 图5说明：小提琴图展示提升值的分布形态，宽度越大表示该数值区间的样本越多")
    plt.close()

# ============================================================
# 图6：排名热图
# 说明：展示每个样本在不同配置下的性能排名，颜色越绿表示排名越靠前
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
    sample_size = min(100, len(df))
    sample_indices = np.random.choice(len(df), sample_size, replace=False)
    sample_rankings = rankings[sample_indices]
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 使用红绿颜色映射（绿色=好排名，红色=差排名）
    im = ax.imshow(sample_rankings.T, aspect='auto', cmap='RdYlGn_r', 
                   interpolation='nearest', vmin=1, vmax=len(configs))
    
    ax.set_xlabel('样本编号 (随机采样)', fontsize=13)
    ax.set_ylabel('配置', fontsize=13)
    ax.set_title('图6：各样本在不同配置下的性能排名热图\n(1=最佳，颜色越绿表示排名越靠前)', 
                 fontsize=14, fontweight='bold')
    
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xticks(np.arange(0, sample_size, 10))
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('排名 (1=最佳)', fontsize=11)
    
    # 计算每个配置的平均排名
    avg_rankings = rankings.mean(axis=0)
    print("\n各配置平均排名:")
    for label, avg_rank in zip(labels, avg_rankings):
        print(f"  {label}: {avg_rank:.2f}")
    
    # 在图上添加平均排名标注
    for i, (label, avg_rank) in enumerate(zip(labels, avg_rankings)):
        ax.text(sample_size + 2, i, f'平均: {avg_rank:.1f}', 
                va='center', ha='left', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图6已保存: {save_path}")
    print("   📊 图6说明：热图展示每个样本在各配置下的排名，绿色表示该配置对该样本效果更好")
    plt.close()
    
    return avg_rankings

# ============================================================
# 图7：最佳配置分布饼图
# 说明：展示哪种配置在最多的样本上取得最佳效果
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
                                        colors=colors_list, explode=explode, shadow=True,
                                        textprops={'fontsize': 10})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax1.set_title('各配置成为最佳的比例', fontsize=13, fontweight='bold')
    
    # 子图2：条形图
    ax2 = axes[1]
    bars = ax2.bar(labels, best_counts, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('配置', fontsize=12)
    ax2.set_ylabel('成为最佳的次数', fontsize=12)
    ax2.set_title('各配置成为最佳的次数统计', fontsize=13, fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, count in zip(bars, best_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count} 次\n({count/len(df)*100:.1f}%)', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('图7：最佳配置分析\n(哪个配置在最多样本上取得最高分)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图7已保存: {save_path}")
    print("   📊 图7说明：饼图和条形图展示各配置成为最佳选择的比例，帮助识别最可靠的配置")
    plt.close()
    
    return best_counts

# ============================================================
# 图8：相关性矩阵热图
# 说明：展示各配置分数之间的相关性，数值越接近1表示结果越一致
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
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(corr_matrix.columns, fontsize=10)
    
    ax.set_title('图8：各配置CLIP-Score相关性矩阵\n(相关系数越接近1表示结果越一致)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('相关系数', fontsize=11)
    
    # 添加相关系数文本
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text_color = "white" if abs(corr_matrix.iloc[i, j]) > 0.6 else "black"
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                          ha="center", va="center", color=text_color,
                          fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图8已保存: {save_path}")
    print("   📊 图8说明：相关性热图展示各配置结果的一致性，高相关性说明配置间表现相似")
    plt.close()
    
    return corr_matrix

# ============================================================
# 图9：性能排序条形图（新增）
# 说明：直观展示各配置的平均性能排名
# ============================================================
def plot_performance_ranking(df, configs, save_path):
    """图9：各配置平均性能排序"""
    means = []
    stds = []
    labels = []
    
    for key, cfg in configs.items():
        scores = df[cfg['col_score']].dropna()
        means.append(scores.mean())
        stds.append(scores.std())
        labels.append(cfg['name'])
    
    # 按平均值排序
    sorted_indices = np.argsort(means)[::-1]
    sorted_means = [means[i] for i in sorted_indices]
    sorted_stds = [stds[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_labels)))
    bars = ax.barh(range(len(sorted_labels)), sorted_means, xerr=sorted_stds, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5,
                   capsize=5, error_kw={'elinewidth': 2, 'markeredgewidth': 2})
    
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, fontsize=11)
    ax.set_xlabel('平均 CLIP-Score', fontsize=13)
    ax.set_title('图9：各配置平均性能排序\n(误差棒表示标准差，体现稳定性)', 
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # 最高分在上方
    ax.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for i, (bar, mean, std) in enumerate(zip(bars, sorted_means, sorted_stds)):
        ax.text(mean + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{mean:.4f} ± {std:.4f}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图9已保存: {save_path}")
    print("   📊 图9说明：横向条形图按平均分排序，误差棒显示稳定性，越长表示波动越大")
    plt.close()

# ============================================================
# 生成统计报告
# ============================================================
def generate_comprehensive_report(df, configs, stats_df, avg_rankings, best_counts, corr_matrix, save_path):
    """生成完整统计报告"""
    report = []
    report.append("="*80)
    report.append("消融实验完整统计报告")
    report.append("="*80)
    report.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    report.append(f"\n【实验概况】")
    report.append(f"  总样本数: {len(df)}")
    if 'total_time_seconds' in df.columns and pd.notna(df['total_time_seconds'].iloc[0]):
        total_time = df['total_time_seconds'].iloc[0]
        report.append(f"  总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    
    report.append(f"\n【各配置详细统计】")
    for key, cfg in configs.items():
        scores = df[cfg['col_score']].dropna()
        report.append(f"\n  {cfg['name']}:")
        report.append(f"    有效样本数: {len(scores)}")
        report.append(f"    平均分: {scores.mean():.6f}")
        report.append(f"    中位数: {scores.median():.6f}")
        report.append(f"    标准差: {scores.std():.6f}")
        report.append(f"    最高分: {scores.max():.6f}")
        report.append(f"    最低分: {scores.min():.6f}")
        report.append(f"    四分位距(IQR): {scores.quantile(0.75) - scores.quantile(0.25):.6f}")
        ci = 1.96 * scores.std() / np.sqrt(len(scores))
        report.append(f"    95%置信区间: [{scores.mean() - ci:.6f}, {scores.mean() + ci:.6f}]")
    
    report.append(f"\n【相对基线提升统计】")
    for _, row in stats_df.iterrows():
        report.append(f"\n  {row['配置']}:")
        report.append(f"    平均提升: {row['平均提升']:.6f}")
        report.append(f"    中位数提升: {row['中位数提升']:.6f}")
        report.append(f"    改进比例: {row['改进比例']:.2f}%")
        report.append(f"    最佳提升: {row['最佳提升']:.6f}")
        report.append(f"    最差提升: {row['最差提升']:.6f}")
        report.append(f"    提升标准差: {row['标准差提升']:.6f}")
    
    report.append(f"\n【排名分析】")
    report.append(f"  (排名1表示最佳，{len(configs)}表示最差)")
    for label, avg_rank in zip([c['name'] for c in configs.values()], avg_rankings):
        report.append(f"    {label}: 平均排名 = {avg_rank:.2f}")
    
    report.append(f"\n【最佳配置统计】")
    for label, count in zip([c['name'] for c in configs.values()], best_counts):
        report.append(f"    {label}: 成为最佳 {count} 次 ({count/len(df)*100:.1f}%)")
    
    report.append(f"\n【相关性分析】")
    # 获取上三角矩阵的非对角线元素
    corr_upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_pairs = corr_upper.unstack().dropna()
    if len(corr_pairs) > 0:
        report.append(f"    最强正相关: {corr_pairs.idxmax()} = {corr_pairs.max():.4f}")
        report.append(f"    最弱相关: {corr_pairs.idxmin()} = {corr_pairs.min():.4f}")
    
    report.append(f"\n【统计显著性检验】")
    report.append(f"  (配对t检验，检验各配置与基线的差异是否显著)")
    baseline_scores = df[configs['baseline']['col_score']].values
    for key, cfg in configs.items():
        if key != 'baseline':
            scores = df[cfg['col_score']].values
            t_stat, p_value = stats.ttest_rel(baseline_scores, scores)
            report.append(f"\n    {cfg['name']} vs 基线:")
            report.append(f"      t统计量: {t_stat:.4f}")
            report.append(f"      p值: {p_value:.6f}")
            report.append(f"      差异显著性: {'显著 (p < 0.05)' if p_value < 0.05 else '不显著 (p >= 0.05)'}")
            if p_value < 0.01:
                report.append(f"      极显著 (p < 0.01)")
    
    report.append(f"\n【最佳配置推荐】")
    best_by_mean = stats_df.loc[stats_df['平均提升'].idxmax(), '配置']
    best_by_ratio = stats_df.loc[stats_df['改进比例'].idxmax(), '配置']
    best_by_rank = [c['name'] for c in configs.values()][np.argmin(avg_rankings)]
    
    report.append(f"  基于平均提升: {best_by_mean}")
    report.append(f"  基于改进比例: {best_by_ratio}")
    report.append(f"  基于平均排名: {best_by_rank}")
    
    if best_by_mean == best_by_ratio == best_by_rank:
        report.append(f"\n  🏆 综合推荐: {best_by_mean}")
    else:
        report.append(f"\n  ⚠️ 不同指标推荐不同配置，建议根据实际需求选择")
        report.append(f"     追求整体效果: {best_by_mean}")
        report.append(f"     追求稳定改进: {best_by_ratio}")
        report.append(f"     追求排名表现: {best_by_rank}")
    
    # 保存报告
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\n✅ 统计报告已保存: {save_path}")
    
    # 打印摘要
    print("\n" + "="*80)
    print("报告摘要")
    print("="*80)
    for line in report[:40]:
        print(line)
    
    return report

# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    output_dir = 'C:/Users/13113/Desktop/数据分析/ablation_charts'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("开始生成消融实验对比图表...")
    print("="*60)
    
    # 生成所有图表
    plot_all_boxplots(df, configs, f'{output_dir}/1_boxplots.png')
    stats_df = plot_improvement_heatmap(df, configs, f'{output_dir}/2_improvement_heatmap.png')
    plot_scatter_matrix(df, configs, f'{output_dir}/3_scatter_matrix.png')
    plot_radar_comparison(df, configs, f'{output_dir}/4_radar_comparison.png')
    plot_violin_improvements(df, configs, f'{output_dir}/5_violin_improvements.png')
    avg_rankings = plot_ranking_heatmap(df, configs, f'{output_dir}/6_ranking_heatmap.png')
    best_counts = plot_best_config_distribution(df, configs, f'{output_dir}/7_best_config_distribution.png')
    corr_matrix = plot_correlation_matrix(df, configs, f'{output_dir}/8_correlation_matrix.png')
    plot_performance_ranking(df, configs, f'{output_dir}/9_performance_ranking.png')
    
    # 生成统计报告
    generate_comprehensive_report(df, configs, stats_df, avg_rankings, best_counts, corr_matrix, 
                                  f'{output_dir}/ablation_report.txt')
    
    print("\n" + "="*60)
    print("✅ 所有图表生成完成！")
    print(f"📁 输出目录: {output_dir}")
    print("="*60)
    
    # 输出关键发现
    print("\n" + "="*60)
    print("📊 关键发现摘要")
    print("="*60)
    
    baseline_mean = df['score_baseline'].mean()
    print(f"\n基线平均分: {baseline_mean:.4f}")
    print("\n各配置表现:")
    
    results = []
    for key, cfg in configs.items():
        if key != 'baseline':
            mean_score = df[cfg['col_score']].mean()
            improvement = mean_score - baseline_mean
            better_pct = (df[cfg['col_score']] > df['score_baseline']).mean() * 100
            results.append((cfg['name'], mean_score, improvement, better_pct))
            arrow = "↑" if improvement > 0 else "↓"
            print(f"  {cfg['name']}: {mean_score:.4f} ({arrow}{abs(improvement):.4f}), 改进比例={better_pct:.1f}%")
    
    # 找出最佳
    best = max(results, key=lambda x: x[1])
    print(f"\n🏆 最佳配置: {best[0]} (平均分: {best[1]:.4f}, 相对基线提升: {best[2]:+.4f})")
    
    # 输出图表说明汇总
    print("\n" + "="*60)
    print("📖 图表说明汇总")
    print("="*60)
    print("""
    图1 - 箱线图: 展示各配置分数分布，可直观比较中位数、四分位数和异常值
    图2 - 提升热图: 展示各配置相对基线的平均提升和改进比例
    图3 - 散点矩阵: 每个点代表样本，点在对角线上方表示优于基线
    图4 - 雷达图: 多维度综合对比，面积越大综合性能越好
    图5 - 小提琴图: 展示提升值分布形态，宽度越大数据越集中
    图6 - 排名热图: 绿色表示该配置对该样本排名靠前
    图7 - 最佳配置: 展示各配置成为最佳选择的比例
    图8 - 相关性矩阵: 相关系数越接近1表示配置间表现越一致
    图9 - 性能排序: 按平均分排序，误差棒显示稳定性
    """)