import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 读取数据
df = pd.read_csv('final_comparison_dataset.csv')

# 数据预处理
df['improvement'] = pd.to_numeric(df['improvement'], errors='coerce')
df['score_baseline'] = pd.to_numeric(df['score_baseline'], errors='coerce')
df['score_ours'] = pd.to_numeric(df['score_ours'], errors='coerce')
df['time_baseline'] = pd.to_numeric(df['time_baseline'], errors='coerce')
df['time_ours'] = pd.to_numeric(df['time_ours'], errors='coerce')

# 移除缺失值
df = df.dropna(subset=['improvement', 'score_baseline', 'score_ours'])

print("=" * 80)
print("RoPE与束搜索组合策略增益效果分析报告")
print("=" * 80)

# ==================== 1. 基本统计量 ====================
print("\n【1. 基本统计量】")
print("-" * 60)

total_samples = len(df)
improved_count = (df['improvement'] > 0).sum()
degraded_count = (df['improvement'] < 0).sum()
unchanged_count = (df['improvement'] == 0).sum()

print(f"总样本数: {total_samples}")
print(f"改进样本数 (improvement > 0): {improved_count} ({improved_count/total_samples*100:.2f}%)")
print(f"退化样本数 (improvement < 0): {degraded_count} ({degraded_count/total_samples*100:.2f}%)")
print(f"不变样本数 (improvement = 0): {unchanged_count} ({unchanged_count/total_samples*100:.2f}%)")

# 分数统计
print(f"\nBaseline分数统计:")
print(f"  均值: {df['score_baseline'].mean():.4f}")
print(f"  中位数: {df['score_baseline'].median():.4f}")
print(f"  标准差: {df['score_baseline'].std():.4f}")
print(f"  最小值: {df['score_baseline'].min():.4f}")
print(f"  最大值: {df['score_baseline'].max():.4f}")

print(f"\nOurs分数统计 (RoPE+束搜索):")
print(f"  均值: {df['score_ours'].mean():.4f}")
print(f"  中位数: {df['score_ours'].median():.4f}")
print(f"  标准差: {df['score_ours'].std():.4f}")
print(f"  最小值: {df['score_ours'].min():.4f}")
print(f"  最大值: {df['score_ours'].max():.4f}")

# 改进统计
print(f"\n改进值统计 (Ours - Baseline):")
print(f"  均值: {df['improvement'].mean():.6f}")
print(f"  中位数: {df['improvement'].median():.6f}")
print(f"  标准差: {df['improvement'].std():.6f}")
print(f"  最小值: {df['improvement'].min():.6f}")
print(f"  最大值: {df['improvement'].max():.6f}")

# 时间统计
print(f"\n时间对比 (秒):")
print(f"  Baseline平均时间: {df['time_baseline'].mean():.4f}s")
print(f"  Ours平均时间: {df['time_ours'].mean():.4f}s")
print(f"  时间变化: {(df['time_ours'].mean() - df['time_baseline'].mean()):.4f}s")
time_change_pct = (df['time_ours'].mean() - df['time_baseline'].mean()) / df['time_baseline'].mean() * 100
print(f"  时间变化百分比: {time_change_pct:.2f}%")

# ==================== 2. 统计检验 ====================
print("\n【2. 统计显著性检验】")
print("-" * 60)

# 配对t检验
t_stat, p_value = stats.ttest_rel(df['score_ours'], df['score_baseline'])
print(f"配对t检验:")
print(f"  t统计量: {t_stat:.4f}")
print(f"  p值: {p_value:.6f}")
if p_value < 0.05:
    print(f"  结论: p < 0.05, 差异具有统计学显著性 ✓")
else:
    print(f"  结论: p >= 0.05, 差异不具有统计学显著性")

# Wilcoxon符号秩检验
w_stat, p_value_w = stats.wilcoxon(df['score_ours'], df['score_baseline'])
print(f"\nWilcoxon符号秩检验:")
print(f"  W统计量: {w_stat:.1f}")
print(f"  p值: {p_value_w:.6f}")
if p_value_w < 0.05:
    print(f"  结论: p < 0.05, 差异具有统计学显著性 ✓")

# Cohen's d 效应量
pooled_std = np.sqrt((df['score_baseline'].std()**2 + df['score_ours'].std()**2) / 2)
cohens_d = (df['score_ours'].mean() - df['score_baseline'].mean()) / pooled_std
print(f"\n效应量 (Cohen's d): {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    print(f"  效应量: 微小")
elif abs(cohens_d) < 0.5:
    print(f"  效应量: 小")
elif abs(cohens_d) < 0.8:
    print(f"  效应量: 中等")
else:
    print(f"  效应量: 大 ✓")

# ==================== 3. 可视化分析 ====================
print("\n【3. 生成可视化图表】")
print("-" * 60)

fig = plt.figure(figsize=(20, 16))

# 3.1 分数分布对比 - 箱线图
ax1 = plt.subplot(3, 3, 1)
data_to_plot = [df['score_baseline'], df['score_ours']]
bp = ax1.boxplot(data_to_plot, labels=['Baseline', 'RoPE+束搜索'], patch_artist=True)
bp['boxes'][0].set_facecolor('#FF9999')
bp['boxes'][1].set_facecolor('#66B3FF')
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Score Distribution Comparison', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 3.2 分数分布直方图
ax2 = plt.subplot(3, 3, 2)
ax2.hist(df['score_baseline'], bins=50, alpha=0.5, label='Baseline', color='#FF9999', edgecolor='black')
ax2.hist(df['score_ours'], bins=50, alpha=0.5, label='RoPE+束搜索', color='#66B3FF', edgecolor='black')
ax2.axvline(df['score_baseline'].mean(), color='#FF9999', linestyle='--', linewidth=2, label=f'Baseline Mean: {df["score_baseline"].mean():.3f}')
ax2.axvline(df['score_ours'].mean(), color='#66B3FF', linestyle='--', linewidth=2, label=f'Ours Mean: {df["score_ours"].mean():.3f}')
ax2.set_xlabel('Score', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Score Distribution Histogram', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3.3 改进值分布
ax3 = plt.subplot(3, 3, 3)
improvement_sorted = df['improvement'].sort_values()
ax3.hist(df['improvement'], bins=50, color='#4ECDC4', edgecolor='black', alpha=0.7)
ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='No Improvement')
ax3.axvline(df['improvement'].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {df["improvement"].mean():.4f}')
ax3.set_xlabel('Improvement (Ours - Baseline)', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Improvement Distribution', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 3.4 改进百分比饼图
ax4 = plt.subplot(3, 3, 4)
labels = ['Improved', 'Degraded', 'Unchanged']
sizes = [improved_count, degraded_count, unchanged_count]
colors = ['#66B3FF', '#FF9999', '#FFE5B4']
explode = (0.05, 0, 0)
ax4.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
ax4.set_title('Improvement Distribution', fontsize=14, fontweight='bold')

# 3.5 时间对比
ax5 = plt.subplot(3, 3, 5)
time_data = [df['time_baseline'].values, df['time_ours'].values]
bp_time = ax5.boxplot(time_data, labels=['Baseline', 'RoPE+束搜索'], patch_artist=True)
bp_time['boxes'][0].set_facecolor('#FF9999')
bp_time['boxes'][1].set_facecolor('#66B3FF')
ax5.set_ylabel('Time (seconds)', fontsize=12)
ax5.set_title('Time Comparison', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 3.6 分数与时间关系散点图
ax6 = plt.subplot(3, 3, 6)
ax6.scatter(df['score_baseline'], df['time_baseline'], alpha=0.5, label='Baseline', color='#FF9999', s=30)
ax6.scatter(df['score_ours'], df['time_ours'], alpha=0.5, label='RoPE+束搜索', color='#66B3FF', s=30)
ax6.set_xlabel('Score', fontsize=12)
ax6.set_ylabel('Time (seconds)', fontsize=12)
ax6.set_title('Score vs Time Scatter', fontsize=14, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 3.7 改进与基线分数的关系
ax7 = plt.subplot(3, 3, 7)
ax7.scatter(df['score_baseline'], df['improvement'], alpha=0.5, color='#4ECDC4', s=30)
ax7.axhline(0, color='red', linestyle='--', linewidth=2)
ax7.set_xlabel('Baseline Score', fontsize=12)
ax7.set_ylabel('Improvement', fontsize=12)
ax7.set_title('Improvement vs Baseline Score', fontsize=14, fontweight='bold')
ax7.grid(True, alpha=0.3)

# 3.8 累积改进分布
ax8 = plt.subplot(3, 3, 8)
sorted_improvement = np.sort(df['improvement'])
cumulative = np.arange(1, len(sorted_improvement) + 1) / len(sorted_improvement)
ax8.plot(sorted_improvement, cumulative, linewidth=2, color='#4ECDC4')
ax8.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Improvement')
ax8.set_xlabel('Improvement', fontsize=12)
ax8.set_ylabel('Cumulative Proportion', fontsize=12)
ax8.set_title('Cumulative Improvement Distribution', fontsize=14, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 3.9 分数改进热力图 - 改进区间分布
ax9 = plt.subplot(3, 3, 9)
bins = np.linspace(-0.2, 0.2, 21)
hist, edges = np.histogram(df['improvement'], bins=bins)
centers = (edges[:-1] + edges[1:]) / 2
colors = ['#FF9999' if x < 0 else '#66B3FF' for x in centers]
ax9.bar(centers, hist, width=0.02, color=colors, edgecolor='black', alpha=0.7)
ax9.axvline(0, color='red', linestyle='--', linewidth=2)
ax9.set_xlabel('Improvement', fontsize=12)
ax9.set_ylabel('Count', fontsize=12)
ax9.set_title('Improvement by Interval', fontsize=14, fontweight='bold')
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('improvement_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("图表已保存至: improvement_analysis.png")

# ==================== 4. 分位数分析 ====================
print("\n【4. 分位数分析】")
print("-" * 60)

percentiles = [5, 25, 50, 75, 95]
print("\nBaseline分数分位数:")
for p in percentiles:
    print(f"  {p}%: {df['score_baseline'].quantile(p/100):.4f}")

print("\nOurs分数分位数:")
for p in percentiles:
    print(f"  {p}%: {df['score_ours'].quantile(p/100):.4f}")

print("\n改进值分位数:")
for p in percentiles:
    print(f"  {p}%: {df['improvement'].quantile(p/100):.6f}")

# ==================== 5. 按改进幅度分组分析 ====================
print("\n【5. 按改进幅度分组分析】")
print("-" * 60)

def categorize_improvement(imp):
    if imp > 0.05:
        return '大幅改进 (>0.05)'
    elif imp > 0:
        return '小幅改进 (0-0.05)'
    elif imp == 0:
        return '无变化'
    elif imp > -0.05:
        return '小幅退化 (-0.05-0)'
    else:
        return '大幅退化 (<-0.05)'

df['category'] = df['improvement'].apply(categorize_improvement)
category_counts = df['category'].value_counts()

print("\n改进幅度分布:")
for cat, count in category_counts.items():
    print(f"  {cat}: {count} ({count/len(df)*100:.2f}%)")

# 各组平均分数
print("\n各组平均分数:")
for cat in category_counts.index:
    group_data = df[df['category'] == cat]
    print(f"  {cat}:")
    print(f"    Baseline: {group_data['score_baseline'].mean():.4f}")
    print(f"    Ours: {group_data['score_ours'].mean():.4f}")
    print(f"    改进: {group_data['improvement'].mean():.6f}")

# ==================== 6. 综合评估指标 ====================
print("\n【6. 综合评估指标】")
print("-" * 60)

# 计算各种评估指标
mean_improvement = df['improvement'].mean()
median_improvement = df['improvement'].median()
success_rate = (df['improvement'] > 0).mean()
win_rate = (df['improvement'] > 0).mean()
tie_rate = (df['improvement'] == 0).mean()
loss_rate = (df['improvement'] < 0).mean()

# 平均改进百分比
mean_improvement_pct = (df['score_ours'] - df['score_baseline']) / df['score_baseline'] * 100
mean_improvement_pct = mean_improvement_pct.replace([np.inf, -np.inf], np.nan).mean()

print(f"平均改进值: {mean_improvement:.6f}")
print(f"中位数改进值: {median_improvement:.6f}")
print(f"平均改进百分比: {mean_improvement_pct:.4f}%")
print(f"胜率 (改进比例): {win_rate*100:.2f}%")
print(f"平率 (不变比例): {tie_rate*100:.2f}%")
print(f"负率 (退化比例): {loss_rate*100:.2f}%")

# 综合得分
composite_score = (win_rate * 1.0 + tie_rate * 0.5) * (1 + abs(mean_improvement) * 10)
print(f"\n综合增益得分: {composite_score:.4f}")

# ==================== 7. 结论 ====================
print("\n【7. 分析结论】")
print("=" * 80)

if mean_improvement > 0:
    print(f"✓ 整体改进: RoPE与束搜索组合策略平均提升了 {mean_improvement:.6f} 分 ({mean_improvement_pct:.2f}%)")
else:
    print(f"✗ 整体退化: RoPE与束搜索组合策略平均降低了 {-mean_improvement:.6f} 分")

if win_rate > 0.5:
    print(f"✓ 胜率优势: {win_rate*100:.2f}% 的样本获得了改进")
else:
    print(f"✗ 胜率劣势: 仅 {win_rate*100:.2f}% 的样本获得改进")

if p_value < 0.05:
    print(f"✓ 统计显著: p={p_value:.6f} < 0.05, 改进具有统计学显著性")
else:
    print(f"✗ 统计不显著: p={p_value:.6f} >= 0.05, 改进不具有统计学显著性")

print("\n【最终评估】")
if mean_improvement > 0 and p_value < 0.05:
    print("✅ RoPE与束搜索组合策略有效提升了图像描述质量")
elif mean_improvement > 0:
    print("⚠️ RoPE与束搜索组合策略有轻微提升，但统计不显著")
else:
    print("❌ RoPE与束搜索组合策略未带来显著改进，建议进一步优化")

print("\n" + "=" * 80)
print("分析完成！")