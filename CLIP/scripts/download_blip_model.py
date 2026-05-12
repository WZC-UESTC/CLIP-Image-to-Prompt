"""
download_blip_model.py
在有网络的环境运行此脚本，下载BLIP模型到本地文件夹
"""

import os
from transformers import BlipProcessor, BlipForConditionalGeneration

# 设置镜像（确保下载成功）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 指定保存路径 - 可以根据需要修改
save_path = "D:/models/blip-image-captioning-base"

print(f"开始下载 BLIP 模型...")
print(f"保存路径: {save_path}")

# 创建目录
os.makedirs(save_path, exist_ok=True)

# 下载并保存processor
print("下载 processor...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
processor.save_pretrained(save_path)
print("✅ processor 保存完成")

# 下载并保存model
print("下载 model...")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.save_pretrained(save_path)
print("✅ model 保存完成")

print(f"\n🎉 所有文件已保存到: {save_path}")
print("现在可以将整个文件夹复制到离线电脑使用")

# 列出下载的文件
print("\n📁 下载的文件列表:")
for file in os.listdir(save_path):
    size = os.path.getsize(os.path.join(save_path, file)) / 1024 / 1024  # MB
    print(f"  - {file}: {size:.2f} MB")