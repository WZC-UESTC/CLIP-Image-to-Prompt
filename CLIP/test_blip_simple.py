"""
简单的BLIP测试脚本，用于诊断问题
"""
import sys
import os
import torch
from PIL import Image

print("="*60)
print("BLIP 模型简单测试")
print("="*60)

# 本地模型路径
local_model_path = r"C:\Users\13113\Desktop\CLIP\models\blip-image-captioning-base"

# 1. 检查模型文件
print("\n1. 检查模型文件...")
if not os.path.exists(local_model_path):
    print(f"❌ 模型路径不存在: {local_model_path}")
    sys.exit(1)

files = os.listdir(local_model_path)
print(f"✅ 找到 {len(files)} 个文件")
required_files = ['config.json', 'pytorch_model.bin', 'preprocessor_config.json']
for f in required_files:
    found = any(f in file for file in files)
    if found:
        print(f"   ✅ {f}")
    else:
        print(f"   ❌ {f} 缺失")

# 2. 加载处理器
print("\n2. 加载处理器...")
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained(local_model_path)
    print("✅ 处理器加载成功")
except Exception as e:
    print(f"❌ 处理器加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. 加载模型
print("\n3. 加载模型...")
try:
    model = BlipForConditionalGeneration.from_pretrained(local_model_path)
    print("✅ 模型加载成功")
    
    # 移动到CPU
    model = model.to('cpu')
    model.eval()
    print("✅ 模型设置为评估模式")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 测试图像处理
print("\n4. 测试图像处理...")
test_image = os.path.join(os.path.dirname(__file__), "images", "test.jpg")

if not os.path.exists(test_image):
    print(f"❌ 测试图片不存在: {test_image}")
    print("创建测试图片...")
    
    # 创建一个简单的测试图片
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 174, 174], fill='blue')
    draw.text((100, 100), "Test", fill='black')
    os.makedirs(os.path.dirname(test_image), exist_ok=True)
    img.save(test_image)
    print(f"✅ 创建测试图片: {test_image}")

print(f"\n处理图片: {test_image}")

try:
    # 加载图像
    image = Image.open(test_image).convert('RGB')
    print(f"✅ 图像加载成功，尺寸: {image.size}")
    
    # 预处理
    inputs = processor(image, return_tensors="pt")
    print(f"✅ 预处理成功，输入形状: {inputs['pixel_values'].shape}")
    
    # 生成描述
    print("\n生成描述...")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_length=50,
            num_beams=3,
            temperature=1.0,
            no_repeat_ngram_size=2
        )
    print(f"✅ 生成成功，输出形状: {out.shape}")
    
    # 解码
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(f"\n✅ 生成的描述: {caption}")
    
except Exception as e:
    print(f"❌ 处理失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("🎉 所有测试通过！模型工作正常")
print("="*60)