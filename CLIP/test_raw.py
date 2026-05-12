"""
原始 transformers 测试 - 直接使用 transformers 库测试 BLIP 模型
"""
import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

print("="*60)
print("原始 Transformers 测试")
print("="*60)

# 本地模型路径
local_model_path = r"C:\Users\13113\Desktop\CLIP\models\blip-image-captioning-base"

# 1. 检查模型路径
print("\n1. 检查模型路径...")
if not os.path.exists(local_model_path):
    print(f"❌ 模型路径不存在: {local_model_path}")
    exit(1)
print(f"✅ 模型路径存在: {local_model_path}")

# 2. 加载模型
print("\n2. 加载处理器和模型...")
try:
    processor = BlipProcessor.from_pretrained(local_model_path)
    print("   ✅ 处理器加载成功")
    
    model = BlipForConditionalGeneration.from_pretrained(local_model_path)
    print("   ✅ 模型加载成功")
    
    model.eval()
    print("   ✅ 模型设置为评估模式")
    
except Exception as e:
    print(f"   ❌ 加载失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 3. 准备测试图片
print("\n3. 准备测试图片...")
test_image = os.path.join(os.path.dirname(__file__), "images", "test.jpg")

if not os.path.exists(test_image):
    print(f"   ⚠️ 测试图片不存在，创建测试图片...")
    from PIL import Image, ImageDraw
    
    # 创建简单的测试图片
    img = Image.new('RGB', (384, 384), color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 284, 284], fill='blue')
    draw.ellipse([150, 150, 234, 234], fill='red')
    draw.text((150, 200), "Test", fill='black')
    
    os.makedirs(os.path.dirname(test_image), exist_ok=True)
    img.save(test_image)
    print(f"   ✅ 创建测试图片: {test_image}")
else:
    print(f"   ✅ 测试图片存在: {test_image}")

# 4. 处理图片
print("\n4. 处理图片并生成描述...")
try:
    # 加载图像
    image = Image.open(test_image).convert('RGB')
    print(f"   📷 图像尺寸: {image.size}")
    
    # 预处理
    inputs = processor(image, return_tensors="pt")
    print(f"   🔧 输入形状: {inputs['pixel_values'].shape}")
    
    # 生成描述 - 使用更多参数
    print(f"   🚀 生成描述中...")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_length=50,
            min_length=10,           # 添加最小长度
            num_beams=5,             # 增加beam search数量
            temperature=0.8,         # 添加温度参数
            repetition_penalty=1.2,  # 重复惩罚
            do_sample=True,          # 启用采样
            top_p=0.9,              # nucleus sampling
            num_return_sequences=1
        )
    
    print(f"   📊 输出形状: {out.shape}")
    print(f"   🔢 输出tokens: {out[0].tolist()}")
    
    # 解码 - 显示不同的解码方式
    caption1 = processor.decode(out[0], skip_special_tokens=True)
    caption2 = processor.decode(out[0], skip_special_tokens=False)
    
    print(f"\n   📝 解码结果 (skip_special_tokens=True): '{caption1}'")
    print(f"   📝 解码结果 (skip_special_tokens=False): '{caption2}'")
    
    # 尝试手动解码每个token
    print(f"\n   🔍 手动解码每个token:")
    for i, token_id in enumerate(out[0].tolist()):
        token = processor.decode([token_id], skip_special_tokens=False)
        print(f"      token {i}: {token_id} -> '{token}'")
    
    if caption1 and caption1.strip():
        print("\n" + "="*60)
        print(f"✅ 成功！生成的描述:")
        print(f"   {caption1}")
        print("="*60)
    else:
        print("\n⚠️ 生成的描述为空，尝试使用不同的生成参数...")
        
        # 尝试另一种生成方式
        print("\n   尝试使用贪婪搜索...")
        with torch.no_grad():
            out2 = model.generate(
                **inputs,
                max_length=50,
                do_sample=False,  # 使用贪婪搜索
                num_beams=1
            )
        
        caption3 = processor.decode(out2[0], skip_special_tokens=True)
        print(f"   贪婪搜索结果: '{caption3}'")
        
        if caption3 and caption3.strip():
            print(f"\n✅ 使用贪婪搜索成功！描述: {caption3}")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()