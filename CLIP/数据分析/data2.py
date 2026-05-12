"""
ablation_analysis_robust.py
消融实验数据可视化分析 - 鲁棒版（解决数据格式问题）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
import re
warnings.filterwarnings('ignore')

# 设置中文字体和美化样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================
# 第一步：数据诊断和清理
# ============================================================

def diagnose_and_clean_data(file_path):
    """诊断数据问题并清理"""
    print("="*80)
    print("数据诊断与清理")
    print("="*80)
    
    # 尝试多种编码读取
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1', 'iso-8859-1']
    
    df = None
    used_encoding = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"✅ 成功使用编码: {encoding}")
            used_encoding = encoding
            break
        except UnicodeDecodeError:
            print(f"❌ 编码 {encoding} 失败")
            continue
    
    if df is None:
        raise Exception("无法读取文件，请检查文件编码")
    
    print(f"\n原始数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 诊断每一列的内容
    print("\n各列数据类型和样例:")
    for col in df.columns:
        print(f"\n  {col}:")
        print(f"    类型: {df[col].dtype}")
        # 显示前3个非空值
        sample = df[col].dropna().head(3).tolist()
        print(f"    样例: {sample}")
        
        # 检查是否是字符串类型的数字
        if df[col].dtype == 'object':
            # 尝试转换为数值
            test_convert = pd.to_numeric(df[col], errors='coerce')
            if test_convert.notna().sum() > 0:
                print(f"    ⚠️ 包含数值但被识别为字符串，将进行转换")
                df[col] = test_convert
    
    # 清理分数列
    score_columns = ['score_baseline', 'score_best_beam', 'score_rope_10000', 
                     'score_rope_5000', 'score_rope_20000', 'score_rope_100000']
    
    for col in score_columns:
        if col in df.columns:
            # 强制转换为数值
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"\n  {col}: 有效数值 {df[col].notna().sum()}/{len(df)}")
    
    # 清理caption列（如果有的话）
    caption_columns = ['cap_baseline', 'cap_best_beam', 'cap_rope_10000',
                       'cap_rope_5000', 'cap_rope_20000', 'cap_rope_100000']
    
    for col in caption_columns:
        if col in df.columns:
            # 替换空字符串和空白为NaN
            df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)
            df[col] = df[col].replace('nan', np.nan)
            # 转换为字符串
            df[col] = df[col].astype(str)
            df[col] = df[col].replace('nan', np.nan)
    
    # 删除所有分数列都是NaN的行
    score_cols_exist = [col for col in score_columns if col in df.columns]
    df_clean = df.dropna(subset=score_cols_exist, how='all')
    
    print(f"\n清理后数据形状: {df_clean.shape}")
    print(f"删除了 {len(df) - len(df_clean)} 行无效数据")
    
    # 显示清理后的统计
    print("\n清理后各分数列统计:")
    for col in score_columns:
        if col in df_clean.columns:
            valid_count = df_clean[col].notna().sum()
            if valid_count > 0:
                print(f"  {col}: 有效={valid_count}, 均值={df_clean[col].mean():.4f}")
    
    return df_clean, used_encoding

# ============================================================
# 读取数据
# ============================================================

file_path = 'C:/Users/13113/Desktop/数据分析/final_unified_ablation2.csv'
df, encoding_used = diagnose_and_clean_data(file_path)

# 定义模型配置（只包含有效的列）
configs = {}
if 'score_baseline' in df.columns and df['score_baseline'].notna().any():
    configs['baseline'] = {'col_score': 'score_baseline', 'name': '基线 (Beam Search)'}
if 'score_best_beam' in df.columns and df['score_best_beam'].notna().any():
    configs['beam'] = {'col_score': 'score_best_beam', 'name': 'Beam Search优化'}
if 'score_rope_10000' in df.columns and df['score_rope_10000'].notna().any():
    configs['rope_10000'] = {'col_score': 'score_rope_10000', 'name': 'RoPE 10000'}
if 'score_rope_5000' in df.columns and df['score_rope_5000'].notna().any():
    configs['rope_5000'] = {'col_score': 'score_rope_5000', 'name': 'RoPE 5000'}
if 'score_rope_20000' in df.columns and df['score_rope_20000'].notna().any():
    configs['rope_20000'] = {'col_score': 'score_rope_20000', 'name': 'RoPE 20000'}
if 'score_rope_100000' in df.columns and df['score_rope_100000'].notna().any():
    configs['rope_100000'] = {'col_score': 'score_rope_100000', 'name': 'RoPE 100000'}

print(f"\n有效配置: {len(configs)}")
for key, cfg in configs.items():
    print(f"  {cfg['name']}: {df[cfg['col_score']].notna().sum()} 个有效样本")

# 创建输出目录
output_dir = 'C:/Users/13113/Desktop/数据分析/ablation_charts_v2'
os.makedirs(output_dir, exist_ok=True)
print(f"\n输出目录: {output_dir}")

# ============================================================
# 辅助函数
# ============================================================

def safe_mean(values):
    """安全计算均值"""
    valid = values[~np.isnan(values)]
    return np.mean(valid) if len(valid) > 0 else 0

def safe_median(values):
    """安全计算中位数"""
    valid = values[~np.isnan(values)]
    return np.median(valid) if len(valid) > 0 else 0

# ============================================================
# 图1：所有配置箱线图对比
# 说明：展示各配置的CLIP-Score分布情况，包括中位数、四分位数和异常值
# ============================================================
def plot_all_boxplots(df, configs, save_path):
    """图1：所有配置分数分布箱线图"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    data_to_plot = []
    labels = []
    colors_list = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'][:len(configs)]
    
    for i, (key, cfg) in enumerate(configs.items()):
        scores = df[cfg['col_score']].dropna().values
        if len(scores) > 0:
            data_to_plot.append(scores)
            labels.append(cfg['name'])
    
    if not data_to_plot:
        print("⚠️ 无有效数据可绘图")
        return
    
    # 绘制箱线图
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                     showmeans=True, meanline=True, 
                     meanprops=dict(color='red', linewidth=2, label='均值'),
                     medianprops=dict(color='darkblue', linewidth=2, label='中位数'))
    
    for patch, color in zip(bp['boxes'], colors_list[:len(data_to_plot)]):
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
    ax.text(0.02, 0.98, '箱体：Q1-Q3 | 横线：中位数 | 虚线：均值 | 须线：1.5倍IQR', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图1已保存: {save_path}")
    plt.close()

# ============================================================
# 图2：相对基线提升柱状图
# 说明：展示各配置相对于基线的平均提升和改进比例
# ============================================================
def plot_improvement_bars(df, configs, save_path):
    """图2：各配置相对于基线的提升柱状图"""
    if 'baseline' not in configs:
        print("⚠️ 无基线数据，跳过图2")
        return None
    
    baseline_scores = df[configs['baseline']['col_score']].values
    
    stats_data = []
    for key, cfg in configs.items():
        if key != 'baseline':
            scores = df[cfg['col_score']].values
            # 只计算两者都有效的样本
            valid_mask = ~np.isnan(baseline_scores) & ~np.isnan(scores)
            if valid_mask.sum() > 0:
                improvement = scores[valid_mask] - baseline_scores[valid_mask]
                stats_data.append({
                    '配置': cfg['name'],
                    '平均提升': np.mean(improvement),
                    '中位数提升': np.median(improvement),
                    '改进比例': (improvement > 0).mean() * 100,
                    '样本数': valid_mask.sum()
                })
    
    if not stats_data:
        print("⚠️ 无有效提升数据，跳过图2")
        return None
    
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
    plt.close()
    
    return stats_df

# ============================================================
# 图3：散点图对比
# 说明：每个子图展示一种配置与基线的分数对应关系
# ============================================================
def plot_scatter_comparison(df, configs, save_path):
    """图3：各配置与基线的散点对比"""
    if 'baseline' not in configs:
        print("⚠️ 无基线数据，跳过图3")
        return
    
    baseline_scores = df[configs['baseline']['col_score']].values
    
    # 确定子图布局
    other_configs = {k: v for k, v in configs.items() if k != 'baseline'}
    n_plots = len(other_configs)
    if n_plots == 0:
        print("⚠️ 无其他配置数据，跳过图3")
        return
    
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (key, cfg) in enumerate(other_configs.items()):
        ax = axes[idx]
        scores = df[cfg['col_score']].values
        
        # 只使用两者都有效的样本
        valid_mask = ~np.isnan(baseline_scores) & ~np.isnan(scores)
        if valid_mask.sum() == 0:
            ax.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax.transAxes)
            continue
        
        valid_baseline = baseline_scores[valid_mask]
        valid_scores = scores[valid_mask]
        improvements = valid_scores - valid_baseline
        
        # 颜色：绿色表示改进，红色表示下降
        colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
        
        ax.scatter(valid_baseline, valid_scores, c=colors, alpha=0.6, 
                  s=30, edgecolors='black', linewidth=0.5)
        
        # 添加对角线
        max_val = max(valid_baseline.max(), valid_scores.max())
        min_val = min(valid_baseline.min(), valid_scores.min())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.7, label='y=x (持平线)')
        
        # 添加趋势线
        if len(valid_baseline) > 1:
            z = np.polyfit(valid_baseline, valid_scores, 1)
            p = np.poly1d(z)
            ax.plot(np.sort(valid_baseline), p(np.sort(valid_baseline)), 'r-', linewidth=2, alpha=0.7, 
                   label=f'趋势线 (斜率={z[0]:.3f})')
        
        ax.set_xlabel('基线 CLIP-Score', fontsize=11)
        ax.set_ylabel(f'{cfg["name"]} CLIP-Score', fontsize=11)
        ax.set_title(f'{cfg["name"]} vs 基线', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=8)
        
        # 添加统计信息
        better_count = (improvements > 0).sum()
        total = len(improvements)
        mean_imp = np.mean(improvements)
        text = f'改进样本: {better_count}/{total} ({better_count/total*100:.1f}%)\n平均提升: {mean_imp:.4f}'
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 隐藏多余的子图
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('图3：各配置与基线对比散点图\n(绿色点表示优于基线，红色点表示劣于基线)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图3已保存: {save_path}")
    plt.close()

# ============================================================
# 图4：性能对比雷达图
# 说明：多维度综合对比各配置性能
# ============================================================
def plot_radar_comparison(df, configs, save_path):
    """图4：多维度性能对比雷达图"""
    if len(configs) < 3:
        print("⚠️ 配置数量不足，跳过雷达图")
        return
    
    # 计算各项指标
    metrics = {}
    for key, cfg in configs.items():
        scores = df[cfg['col_score']].dropna()
        if len(scores) > 0:
            metrics[cfg['name']] = {
                '平均分': scores.mean(),
                '中位数': scores.median(),
                '最高分': scores.max(),
                '稳定性(1/标准差)': 1 / scores.std() if scores.std() > 0 else 0,
                'Q3四分位': scores.quantile(0.75)
            }
    
    if len(metrics) < 2:
        print("⚠️ 有效指标不足，跳过雷达图")
        return
    
    radar_metrics = list(list(metrics.values())[0].keys())
    
    # 归一化
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    colors_list = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
    
    for i, (name, values) in enumerate(metrics.items()):
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
    ax.set_title('图4：各配置多维度性能对比雷达图\n(数值越大表示性能越好，面积越大综合性能越强)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图4已保存: {save_path}")
    plt.close()

# ============================================================
# 图5：改进幅度分布直方图
# 说明：展示各配置相对于基线提升幅度的分布
# ============================================================
def plot_improvement_histograms(df, configs, save_path):
    """图5：各配置改进幅度分布直方图"""
    if 'baseline' not in configs:
        print("⚠️ 无基线数据，跳过图5")
        return
    
    baseline_scores = df[configs['baseline']['col_score']].values
    
    other_configs = {k: v for k, v in configs.items() if k != 'baseline'}
    n_plots = len(other_configs)
    if n_plots == 0:
        print("⚠️ 无其他配置数据，跳过图5")
        return
    
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (key, cfg) in enumerate(other_configs.items()):
        ax = axes[idx]
        scores = df[cfg['col_score']].values
        
        valid_mask = ~np.isnan(baseline_scores) & ~np.isnan(scores)
        if valid_mask.sum() == 0:
            ax.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax.transAxes)
            continue
        
        improvements = scores[valid_mask] - baseline_scores[valid_mask]
        
        # 绘制直方图
        n, bins, patches = ax.hist(improvements, bins=30, edgecolor='black', alpha=0.7, color='#3498db')
        
        # 根据正负着色
        for patch, left in zip(patches, bins[:-1]):
            if left < 0:
                patch.set_facecolor('#e74c3c')
            else:
                patch.set_facecolor('#2ecc71')
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零提升线')
        ax.axvline(x=np.mean(improvements), color='blue', linestyle='-', linewidth=2, 
                   label=f'均值: {np.mean(improvements):.4f}')
        
        ax.set_xlabel('CLIP-Score提升值', fontsize=11)
        ax.set_ylabel('样本数', fontsize=11)
        ax.set_title(f'{cfg["name"]} 改进幅度分布', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('图5：各配置相对基线改进幅度分布', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图5已保存: {save_path}")
    plt.close()

# ============================================================
# 图6：性能排序条形图
# 说明：直观展示各配置的平均性能排名
# ============================================================
def plot_performance_ranking(df, configs, save_path):
    """图6：各配置平均性能排序"""
    means = []
    stds = []
    labels = []
    
    for key, cfg in configs.items():
        scores = df[cfg['col_score']].dropna()
        if len(scores) > 0:
            means.append(scores.mean())
            stds.append(scores.std())
            labels.append(cfg['name'])
    
    if not means:
        print("⚠️ 无有效数据，跳过图6")
        return
    
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
    ax.set_title('图6：各配置平均性能排序\n(误差棒表示标准差，体现稳定性)', 
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for i, (bar, mean, std) in enumerate(zip(bars, sorted_means, sorted_stds)):
        ax.text(mean + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{mean:.4f} ± {std:.4f}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图6已保存: {save_path}")
    plt.close()

# ============================================================
# 图7：最佳配置分布饼图
# 说明：展示哪种配置在最多的样本上取得最佳效果
# ============================================================
def plot_best_config_distribution(df, configs, save_path):
    """图7：每个样本的最佳配置分布"""
    if len(configs) < 2:
        print("⚠️ 配置数量不足，跳过图7")
        return
    
    # 收集所有分数
    all_scores = []
    labels = []
    
    for key, cfg in configs.items():
        scores = df[cfg['col_score']].values
        all_scores.append(scores)
        labels.append(cfg['name'])
    
    all_scores = np.array(all_scores).T
    
    # 找出每个样本的最佳配置（忽略含NaN的行）
    valid_rows = ~np.isnan(all_scores).any(axis=1)
    if valid_rows.sum() == 0:
        print("⚠️ 无有效样本，跳过图7")
        return
    
    valid_scores = all_scores[valid_rows]
    best_indices = np.argmax(valid_scores, axis=1)
    best_counts = np.bincount(best_indices, minlength=len(labels))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 子图1：饼图
    ax1 = axes[0]
    colors_list = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    explode = [0.05 if count == best_counts.max() else 0 for count in best_counts]
    
    wedges, texts, autotexts = ax1.pie(best_counts, labels=labels, autopct='%1.1f%%',
                                        colors=colors_list, explode=explode, shadow=True)
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
    total_valid = valid_rows.sum()
    for bar, count in zip(bars, best_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count} 次\n({count/total_valid*100:.1f}%)', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('图7：最佳配置分析\n(哪个配置在最多样本上取得最高分)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图7已保存: {save_path}")
    plt.close()
    
    return best_counts

# ============================================================
# 图8：相关性矩阵热图
# 说明：展示各配置分数之间的相关性
# ============================================================
def plot_correlation_matrix(df, configs, save_path):
    """图8：各配置分数相关性矩阵"""
    if len(configs) < 2:
        print("⚠️ 配置数量不足，跳过图8")
        return
    
    # 构建相关性矩阵
    score_df = pd.DataFrame()
    for key, cfg in configs.items():
        score_df[cfg['name']] = df[cfg['col_score']]
    
    # 删除全为NaN的行
    score_df = score_df.dropna(how='all')
    
    if len(score_df) < 2:
        print("⚠️ 有效样本不足，跳过图8")
        return
    
    corr_matrix = score_df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(corr_matrix.columns, fontsize=10)
    
    ax.set_title('图8：各配置CLIP-Score相关性矩阵\n(相关系数越接近1表示结果越一致)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('相关系数', fontsize=11)
    
    # 添加相关系数文本
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text_color = "white" if abs(corr_matrix.iloc[i, j]) > 0.6 else "black"
            ax.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                   ha="center", va="center", color=text_color,
                   fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图8已保存: {save_path}")
    plt.close()
    
    return corr_matrix

# ============================================================
# 生成统计报告
# ============================================================
def generate_report(df, configs, output_dir):
    """生成统计报告"""
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("消融实验统计报告")
    report_lines.append("="*80)
    report_lines.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"总样本数: {len(df)}")
    
    report_lines.append("\n【各配置统计】")
    for key, cfg in configs.items():
        scores = df[cfg['col_score']].dropna()
        if len(scores) > 0:
            report_lines.append(f"\n{cfg['name']}:")
            report_lines.append(f"  有效样本: {len(scores)}")
            report_lines.append(f"  平均分: {scores.mean():.6f}")
            report_lines.append(f"  中位数: {scores.median():.6f}")
            report_lines.append(f"  标准差: {scores.std():.6f}")
            report_lines.append(f"  最高分: {scores.max():.6f}")
            report_lines.append(f"  最低分: {scores.min():.6f}")
    
    # 如果有基线，计算提升
    if 'baseline' in configs:
        baseline_scores = df[configs['baseline']['col_score']].values
        report_lines.append("\n【相对基线提升】")
        for key, cfg in configs.items():
            if key != 'baseline':
                scores = df[cfg['col_score']].values
                valid_mask = ~np.isnan(baseline_scores) & ~np.isnan(scores)
                if valid_mask.sum() > 0:
                    improvements = scores[valid_mask] - baseline_scores[valid_mask]
                    report_lines.append(f"\n{cfg['name']}:")
                    report_lines.append(f"  平均提升: {np.mean(improvements):.6f}")
                    report_lines.append(f"  中位数提升: {np.median(improvements):.6f}")
                    report_lines.append(f"  改进比例: {(improvements > 0).mean()*100:.2f}%")
                    
                    # t检验
                    t_stat, p_value = stats.ttest_rel(baseline_scores[valid_mask], scores[valid_mask])
                    report_lines.append(f"  配对t检验 p值: {p_value:.6f}")
                    report_lines.append(f"  差异显著性: {'显著' if p_value < 0.05 else '不显著'}")
    
    # 保存报告
    report_path = os.path.join(output_dir, 'statistics_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n✅ 统计报告已保存: {report_path}")
    
    # 打印到控制台
    print("\n" + "\n".join(report_lines[:50]))
    
    return report_lines

# ============================================================
# 主函数
# ============================================================

print("\n" + "="*80)
print("开始生成消融实验图表")
print("="*80)

# 生成所有图表
print("\n生成图表...")

# 图1：箱线图
plot_all_boxplots(df, configs, f'{output_dir}/01_boxplots.png')

# 图2：提升柱状图
stats_df = plot_improvement_bars(df, configs, f'{output_dir}/02_improvement_bars.png')

# 图3：散点图
plot_scatter_comparison(df, configs, f'{output_dir}/03_scatter_comparison.png')

# 图4：雷达图
plot_radar_comparison(df, configs, f'{output_dir}/04_radar_comparison.png')

# 图5：改进幅度直方图
plot_improvement_histograms(df, configs, f'{output_dir}/05_improvement_histograms.png')

# 图6：性能排序
plot_performance_ranking(df, configs, f'{output_dir}/06_performance_ranking.png')

# 图7：最佳配置分布
best_counts = plot_best_config_distribution(df, configs, f'{output_dir}/07_best_config_distribution.png')

# 图8：相关性矩阵
corr_matrix = plot_correlation_matrix(df, configs, f'{output_dir}/08_correlation_matrix.png')

# 生成统计报告
generate_report(df, configs, output_dir)

print("\n" + "="*80)
print("✅ 所有图表生成完成！")
print(f"📁 输出目录: {output_dir}")
print("="*80)

# 输出图表说明
print("\n" + "="*80)
print("📖 图表说明")
print("="*80)
print("""
01_boxplots.png        - 箱线图：展示各配置分数分布，可直观比较中位数和离散程度
02_improvement_bars.png - 提升柱状图：展示各配置相对基线的平均提升和改进比例
03_scatter_comparison.png - 散点图：每个点代表样本，对角线上方表示优于基线
04_radar_comparison.png   - 雷达图：多维度综合对比，面积越大性能越好
05_improvement_histograms.png - 改进直方图：展示提升值的分布形态
06_performance_ranking.png   - 性能排序：按平均分排序，误差棒显示稳定性
07_best_config_distribution.png - 最佳配置：各配置成为最佳选择的比例
08_correlation_matrix.png     - 相关性矩阵：配置间结果一致性分析
statistics_report.txt    - 完整统计报告
""")