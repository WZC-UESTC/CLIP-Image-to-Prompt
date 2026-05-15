# 🎨 CLIP图生文自动提示词生成系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

基于 CLIP 和 BLIP 模型的智能图像描述生成系统，自动为图像生成高质量的描述性提示词。

</div>

---

## 📋 项目简介

本项目结合了 OpenAI 的 CLIP 模型和 Salesforce 的 BLIP 模型，通过三阶段协同处理，自动生成与图像内容高度匹配的描述性提示词。系统能够识别图像中的物体、场景、风格等元素，并生成自然流畅的文本描述。

### 核心功能

- 🖼️ **智能图像分析**：自动识别图像中的物体、场景、风格、颜色等信息
- 📝 **动态提示词生成**：根据图像内容生成个性化的描述文本
- 🎬 **可视化演示**：动态展示图像处理的完整流程
- ⚡ **批量处理**：支持大规模图片批量推理（已完成12万张测试）
- 🔧 **离线运行**：支持完全离线运行，不依赖网络

---

## 🏗️ 技术架构
图片上传 → 特征提取 → CLIP标签匹配 → 文本生成 → 结果融合

### 三阶段流程

| 阶段 | 模型 | 功能 |
|------|------|------|
| 第一阶段 | BLIP | 生成图像初步文字描述 |
| 第二阶段 | CLIP | 从标签库筛选精准标签 |
| 第三阶段 | Transformer | 语义重构 + 束搜索优化 |

### 技术栈

| 技术 | 用途 |
|------|------|
| PyTorch | 深度学习框架 |
| Transformers | Hugging Face 模型库 |
| OpenCLIP | CLIP 模型实现 |
| Gradio | Web 演示界面 |
| Matplotlib | 可视化绘图 |

---

## 📊 实验数据

| 指标 | 数值 |
|------|------|
| 测试样本 | 12万张图片 |
| 处理效率 | 约 0.16 秒/张 |
| 成功率 | 98%+ |
| 质量提升 | 相比基线提升 14.2% |

### 调参实验结果

| 配置 | CLIP-Score | 耗时/张 | 相对基线提升 |
|------|-----------|---------|-------------|
| 基线（贪心解码） | 0.2562 | 0.14s | - |
| 束搜索 (k=3) | 0.2927 | 0.16s | **+14.2%** |
| 束搜索 (k=5) | 0.2925 | 0.18s | +14.1% |
| RoPE (base=10000) | 0.2815 | 0.17s | +9.9% |

> **结论**：束搜索 (k=3) 在质量和速度间取得最佳平衡，推荐使用。

---

## 🚀 快速开始
### 📁 项目结构
CLIP-Image-to-Prompt/
├── visual_demo_fixed.py      # 主程序（动态演示）

├── visual_demo.py             # 原始演示程序

├── blip_caption.py            # BLIP 模型模块

├── clip_blip_integration.py   # CLIP-BLIP 集成模块

├── requirements.txt           # 依赖清单

├── .gitignore                 # Git 忽略文件

├── uploaded_images/           # 上传图片保存目录

├── examples/                  # 示例图片

└── models/                    # 本地模型文件
### 环境要求

- Python 3.8 或更高版本
- pip 包管理器


### 安装步骤

#### 1. 克隆仓库


git clone https://github.com/WZC-UESTC/CLIP-Image-to-Prompt.git
cd CLIP-Image-to-Prompt

#### 2.📖 使用说明
1. 启动系统
bash
python visual_demo_fixed.py
2. 上传图片
点击"上传图片"选择本地图片

或点击示例图片快速体验

3. 开始处理
点击"开始处理"按钮，观看动态演示：

📸 步骤1/5：图片加载与预处理

🔍 步骤2/5：图像特征提取

🔄 步骤3/5：CLIP 特征比对

📝 步骤4/5：文本生成（逐词动画）

✨ 步骤5/5：结果融合与输出

4. 查看结果
系统会生成最终的提示词，可直接复制使用。

🔧 高级配置
修改生成参数
编辑 blip_caption.py 中的参数：

python
#### 生成参数配置
max_length=20      # 最大生成长度
num_beams=5        # 束搜索宽度
temperature=0.8    # 温度参数（0.5-1.2）
repetition_penalty=1.2  # 重复惩罚
自定义标签库
编辑 clip_blip_integration.py 中的 load_tag_library 方法，添加自定义标签。

### 📈 成果展示
示例输出
<img width="2560" height="1398" alt="8dc5e30bbf940f8799f726ad875cc81c" src="https://github.com/user-attachments/assets/6943f575-2f6c-4740-a205-de552c9b7317" />
👥 团队信息
角色	姓名

项目负责人	叶又豪

团队成员	王子琛

团队成员	周恒凯
团队成员	刘美丹珠
团队成员	陈宝仪

指导教师	陈娟（副教授）
