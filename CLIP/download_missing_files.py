"""
下载缺失的tokenizer文件
"""
import os
import requests
from tqdm import tqdm

print("="*60)
print("下载缺失的tokenizer文件")
print("="*60)

# 本地模型路径
local_model_path = r"C:\Users\13113\Desktop\CLIP\models\blip-image-captioning-base"

# 缺失的文件列表及其下载URL
missing_files = {
    "tokenizer.json": "https://hf-mirror.com/Salesforce/blip-image-captioning-base/resolve/main/tokenizer.json",
    "tokenizer_config.json": "https://hf-mirror.com/Salesforce/blip-image-captioning-base/resolve/main/tokenizer_config.json"
}

print(f"\n模型路径: {local_model_path}")
print(f"需要下载 {len(missing_files)} 个文件")

def download_file(url, save_path):
    """下载文件并显示进度"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(save_path)) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"   下载失败: {e}")
        return False

# 下载缺失的文件
print("\n开始下载...")
success_count = 0

for filename, url in missing_files.items():
    save_path = os.path.join(local_model_path, filename)
    
    if os.path.exists(save_path):
        print(f"\n✅ {filename} 已存在")
        success_count += 1
        continue
    
    print(f"\n📥 下载: {filename}")
    if download_file(url, save_path):
        print(f"   ✅ 下载完成")
        success_count += 1
    else:
        print(f"   ❌ 下载失败")

print("\n" + "="*60)
print(f"下载完成: {success_count}/{len(missing_files)}")
print("="*60)

# 验证文件
print("\n验证模型目录文件:")
files = os.listdir(local_model_path)
required = ['config.json', 'pytorch_model.bin', 'preprocessor_config.json', 
            'vocab.json', 'merges.txt', 'tokenizer.json', 'tokenizer_config.json']

for f in required:
    if os.path.exists(os.path.join(local_model_path, f)):
        size = os.path.getsize(os.path.join(local_model_path, f)) / 1024
        print(f"  ✅ {f} ({size:.1f} KB)")
    else:
        # 检查是否有类似文件
        similar = [x for x in files if f.replace('.json', '') in x or f.replace('.txt', '') in x]
        if similar:
            print(f"  ⚠️ {f} -> 找到相似文件: {similar}")
        else:
            print(f"  ❌ {f} 缺失")