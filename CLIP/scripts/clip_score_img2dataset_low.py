# analyze_img2dataset.py
"""
img2dataset数据集质量分析工具
韦思怡专用：分析下载的图片集质量，生成报告
"""
import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class Img2DatasetAnalyzer:
    def __init__(self, dataset_root):
        """
        初始化分析器
        :param dataset_root: img2dataset下载的根目录
        """
        self.dataset_root = dataset_root
        self.stats = {}
        
    def scan_dataset(self, sample_ratio=1.0):
        """
        扫描整个数据集，收集基本信息
        :param sample_ratio: 抽样比例（1.0表示全部）
        """
        print(f"正在扫描数据集: {self.dataset_root}")
        
        # 查找所有json文件（包含caption信息）
        json_files = glob.glob(os.path.join(self.dataset_root, "*", "*.json"))
        
        # 如果数据量太大，可以抽样
        if sample_ratio < 1.0:
            import random
            num_samples = int(len(json_files) * sample_ratio)
            json_files = random.sample(json_files, num_samples)
        
        print(f"找到 {len(json_files)} 个样本")
        
        # 收集信息
        data = []
        for i, json_file in enumerate(json_files):
            if i % 1000 == 0:
                print(f"处理中: {i}/{len(json_files)}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    
                # 对应的图片文件
                img_file = json_file.replace('.json', '.jpg')
                img_exists = os.path.exists(img_file)
                
                data.append({
                    'shard': os.path.basename(os.path.dirname(json_file)),
                    'image_id': content.get('key', ''),
                    'caption': content.get('caption', ''),
                    'caption_length': len(content.get('caption', '')),
                    'url': content.get('url', ''),
                    'status': content.get('status', ''),
                    'image_exists': img_exists,
                    'file_path': json_file
                })
            except Exception as e:
                print(f"处理文件出错 {json_file}: {e}")
        
        self.df = pd.DataFrame(data)
        print(f"数据收集完成，有效样本数: {len(self.df)}")
        
        return self.df
    
    def generate_basic_report(self):
        """
        生成基础统计报告
        """
        if not hasattr(self, 'df'):
            print("请先运行 scan_dataset()")
            return
        
        print("\n" + "="*60)
        print("📊 img2dataset 数据集分析报告")
        print("="*60)
        print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"数据集路径: {self.dataset_root}")
        print("-"*60)
        
        # 1. 基本统计
        print("\n📌 基本统计:")
        print(f"  总样本数: {len(self.df)}")
        print(f"  图片存在率: {(self.df['image_exists'].sum() / len(self.df) * 100):.2f}%")
        
        # 2. Caption长度分析
        print("\n📝 Caption长度分析:")
        print(f"  平均长度: {self.df['caption_length'].mean():.2f}")
        print(f"  中位数长度: {self.df['caption_length'].median():.2f}")
        print(f"  最大长度: {self.df['caption_length'].max()}")
        print(f"  最小长度: {self.df['caption_length'].min()}")
        
        # 3. 空caption统计
        empty_captions = self.df[self.df['caption_length'] == 0].shape[0]
        print(f"\n⚠️ 空caption数量: {empty_captions} ({empty_captions/len(self.df)*100:.2f}%)")
        
        # 4. 分片统计
        print("\n📁 分片统计:")
        shard_stats = self.df.groupby('shard').size().sort_values(ascending=False)
        print(f"  分片总数: {len(shard_stats)}")
        print(f"  最大分片: {shard_stats.max()} 张")
        print(f"  最小分片: {shard_stats.min()} 张")
        print(f"  平均分片: {shard_stats.mean():.0f} 张")
        
        # 5. Top 10分片
        print("\n🔝 最大10个分片:")
        for shard, count in shard_stats.head(10).items():
            print(f"  {shard}: {count} 张")
        
        # 保存报告
        self._save_report()
        
    def plot_caption_length_distribution(self):
        """
        绘制caption长度分布图
        """
        plt.figure(figsize=(12, 5))
        
        # 主图
        plt.subplot(1, 2, 1)
        plt.hist(self.df['caption_length'], bins=50, color='skyblue', edgecolor='black')
        plt.title('Caption长度分布')
        plt.xlabel('长度')
        plt.ylabel('频次')
        plt.grid(True, alpha=0.3)
        
        # 累积分布
        plt.subplot(1, 2, 2)
        sorted_lengths = np.sort(self.df['caption_length'])
        yvals = np.arange(len(sorted_lengths)) / float(len(sorted_lengths))
        plt.plot(sorted_lengths, yvals, color='orange', linewidth=2)
        plt.title('Caption长度累积分布')
        plt.xlabel('长度')
        plt.ylabel('累积比例')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('caption_analysis.png', dpi=100)
        print("\n📈 图表已保存: caption_analysis.png")
        
    def export_filtered_dataset(self, min_length=5, output_file='filtered_samples.csv'):
        """
        导出过滤后的样本（供孙霖/王子琛使用）
        :param min_length: 最小caption长度
        """
        # 过滤条件
        filtered = self.df[
            (self.df['caption_length'] >= min_length) & 
            (self.df['image_exists'] == True)
        ]
        
        # 导出必要信息
        export_cols = ['shard', 'image_id', 'caption', 'caption_length']
        filtered[export_cols].to_csv(output_file, index=False)
        
        print(f"\n✅ 已导出过滤后样本: {len(filtered)} 条")
        print(f"  保存到: {output_file}")
        
        return filtered
    
    def _save_report(self):
        """保存报告到文件"""
        with open('dataset_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(f"数据集分析报告\n")
            f.write(f"生成时间: {datetime.now()}\n")
            f.write(f"总样本数: {len(self.df)}\n")
            f.write(f"平均caption长度: {self.df['caption_length'].mean():.2f}\n")
        
        print("\n📄 报告已保存: dataset_analysis_report.txt")

# ================== 使用示例 ==================
if __name__ == "__main__":
    # 配置参数
    DATASET_ROOT = r"C:\Users\13113\Desktop\CLIP\downloaded_images"  # 改成你的实际路径
    
    print("🔍 启动img2dataset分析器")
    print("="*60)
    
    # 创建分析器
    analyzer = Img2DatasetAnalyzer(DATASET_ROOT)
    
    # 1. 扫描数据集（抽样10%快速预览）
    print("\n步骤1: 扫描数据集...")
    df = analyzer.scan_dataset(sample_ratio=0.1)  # 先扫10%看看
    
    # 2. 生成基础报告
    print("\n步骤2: 生成统计报告...")
    analyzer.generate_basic_report()
    
    # 3. 绘制分布图
    print("\n步骤3: 绘制分布图...")
    analyzer.plot_caption_length_distribution()
    
    # 4. 导出过滤后的样本
    print("\n步骤4: 导出过滤样本...")
    filtered = analyzer.export_filtered_dataset(min_length=5)
    
    print("\n✨ 分析完成！")
    print("请查看生成的文件:")
    print("  - dataset_analysis_report.txt (统计报告)")
    print("  - caption_analysis.png (分布图)")
    print("  - filtered_samples.csv (过滤后样本)")