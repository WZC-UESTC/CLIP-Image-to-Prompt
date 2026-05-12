"""
测试新功能：扩充token到20 + 风格描述
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from blip_caption import BLIPCaptionGenerator

print("="*60)
print("测试新功能")
print("="*60)

local_model_path = r"C:\Users\13113\Desktop\CLIP\models\blip-image-captioning-base"

# 检查模型是否存在
if not os.path.exists(local_model_path):
    print(f"❌ 模型不存在: {local_model_path}")
    sys.exit(1)

# 初始化
print("\n初始化模型...")
generator = BLIPCaptionGenerator(device='cpu', local_model_path=local_model_path)

# 测试图片
test_image = os.path.join(os.path.dirname(__file__), "images", "test.jpg")

if not os.path.exists(test_image):
    print(f"❌ 测试图片不存在: {test_image}")
    print("创建测试图片...")
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (384, 384), color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 284, 284], fill='blue')
    draw.ellipse([150, 150, 234, 234], fill='red')
    draw.text((150, 200), "Test", fill='black')
    os.makedirs(os.path.dirname(test_image), exist_ok=True)
    img.save(test_image)
    print(f"✅ 创建测试图片: {test_image}")

# 测试新功能
print("\n" + "="*60)
print("1. 测试内容描述（20个token）:")
print("="*60)
content = generator.generate_caption(test_image, max_length=20, min_length=10)
print(f"结果: {content}")
if content:
    print(f"Token数估算: {len(content.split())} 个词")

print("\n" + "="*60)
print("2. 测试风格描述:")
print("="*60)
style = generator.generate_style_description(test_image, max_length=15)
print(f"结果: {style}")

print("\n" + "="*60)
print("3. 测试完整描述:")
print("="*60)
complete = generator.generate_complete_description(test_image)
print(f"内容: {complete['content']}")
print(f"风格: {complete['style']}")
print(f"完整: {complete['full']}")

print("\n" + "="*60)
print("✅ 测试完成")
print("="*60)