"""
修复BLIP processor问题
"""
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import BertTokenizer

local_model_path = r"C:\Users\13113\Desktop\CLIP\models\blip-image-captioning-base"

print("="*60)
print("修复BLIP Processor")
print("="*60)

# 1. 检查模型目录中的文件
print("\n1. 检查模型目录中的文件:")
files = os.listdir(local_model_path)
for f in sorted(files):
    print(f"   - {f}")

# 2. 尝试不同的加载方式
print("\n2. 尝试不同的加载方式...")

# 方式1：使用from_pretrained
print("\n   方式1: 标准加载")
try:
    processor1 = BlipProcessor.from_pretrained(local_model_path)
    print(f"   ✅ 加载成功")
    print(f"   Tokenizer类型: {type(processor1.tokenizer)}")
    
    # 测试编码解码
    test_ids = [101, 2054, 102]
    decoded = processor1.decode(test_ids)
    print(f"   测试解码 {test_ids}: '{decoded}'")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# 方式2：分别加载image processor和tokenizer
print("\n   方式2: 分别加载")
try:
    from transformers import BlipImageProcessor
    
    image_processor = BlipImageProcessor.from_pretrained(local_model_path)
    tokenizer = BertTokenizer.from_pretrained(local_model_path)
    
    from transformers import BlipProcessor
    
    processor2 = BlipProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer
    )
    print(f"   ✅ 自定义processor创建成功")
    
    test_ids = [101, 2054, 102]
    decoded = processor2.decode(test_ids)
    print(f"   测试解码 {test_ids}: '{decoded}'")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# 3. 检查vocab文件
print("\n3. 检查vocab文件:")
vocab_files = ['vocab.json', 'vocab.txt', 'tokenizer.json']
for vf in vocab_files:
    path = os.path.join(local_model_path, vf)
    if os.path.exists(path):
        print(f"   ✅ {vf} 存在")
    else:
        print(f"   ❌ {vf} 不存在")

print("\n" + "="*60)
print("建议：重新下载模型文件，确保包含所有必要的tokenizer文件")
print("需要的文件: vocab.json, merges.txt, tokenizer_config.json")
print("="*60)