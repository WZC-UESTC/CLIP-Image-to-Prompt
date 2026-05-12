"""
测试本地 BLIP 模型是否能正常加载
"""
import os
import sys

# 设置镜像（备用）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 本地模型路径
model_path = r"C:\Users\13113\Desktop\CLIP\models\blip-image-captioning-base"

print("="*60)
print("测试本地 BLIP 模型")
print("="*60)
print(f"模型路径: {model_path}")

# 检查文件
if not os.path.exists(model_path):
    print(f"❌ 模型目录不存在: {model_path}")
    sys.exit(1)

files = os.listdir(model_path)
print(f"\n找到 {len(files)} 个文件:")
for f in sorted(files):
    size = os.path.getsize(os.path.join(model_path, f)) / (1024*1024)
    if size > 1:
        print(f"  ✅ {f} ({size:.1f} MB)")
    else:
        print(f"  ✅ {f} ({size*1024:.0f} KB)")

# 检查必需文件
required = ['config.json', 'pytorch_model.bin', 'preprocessor_config.json', 'vocab', 'merges']
missing = []
for f in required:
    found = False
    for existing in files:
        if existing.startswith(f):
            found = True
            break
    if not found:
        missing.append(f)

if missing:
    print(f"\n⚠️ 缺少必需文件: {missing}")
    sys.exit(1)

print("\n✅ 所有必需文件都存在")

# 测试加载模型
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image
    import torch
    
    print("\n尝试加载模型...")
    
    # 加载处理器
    processor = BlipProcessor.from_pretrained(model_path)
    print("✅ 处理器加载成功")
    
    # 加载模型
    model = BlipForConditionalGeneration.from_pretrained(model_path)
    print("✅ 模型加载成功")
    
    print("\n🎉 模型完全可用！")
    
except Exception as e:
    print(f"\n❌ 加载失败: {e}")
    sys.exit(1)