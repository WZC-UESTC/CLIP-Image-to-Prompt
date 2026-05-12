"""
快速测试 BLIP 模型
"""
import sys
import os

# 添加 scripts 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from blip_caption import BLIPCaptionGenerator

# 本地模型路径
local_model_path = r"C:\Users\13113\Desktop\CLIP\models\blip-image-captioning-base"

print("="*60)
print("快速测试 BLIP 模型")
print("="*60)

# 检查模型是否存在
if not os.path.exists(local_model_path):
    print(f"❌ 模型不存在: {local_model_path}")
    sys.exit(1)

print(f"✅ 找到模型目录: {local_model_path}")

# 初始化模型
print("\n初始化 BLIP 模型...")
try:
    generator = BLIPCaptionGenerator(device='cpu', local_model_path=local_model_path)
    print("\n✅ 模型加载成功！")
except Exception as e:
    print(f"\n❌ 模型加载失败: {e}")
    sys.exit(1)

# 测试图片
test_image = os.path.join(os.path.dirname(__file__), 'images', 'test.jpg')
if os.path.exists(test_image):
    print(f"\n处理图片: {test_image}")
    caption = generator.generate_caption(test_image)
    if caption:
        print(f"✅ 生成的描述: {caption}")
    else:
        print("❌ 生成描述失败")
else:
    print(f"\n⚠️ 测试图片不存在: {test_image}")
    print("请在 images 文件夹中放一张图片进行测试")