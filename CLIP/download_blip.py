"""
使用 requests 库直接下载，支持断点续传，绕过 SSL 验证问题
"""
import os
import requests
from tqdm import tqdm
import time
import sys

# 禁用 SSL 警告（仅用于解决网络问题）
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_file_with_resume(url, save_path, max_retries=3):
    """支持断点续传和重试的下载"""
    for attempt in range(max_retries):
        try:
            # 获取已下载的大小
            downloaded_size = 0
            if os.path.exists(save_path):
                downloaded_size = os.path.getsize(save_path)
                if downloaded_size > 0:
                    print(f"已下载 {downloaded_size/(1024**3):.2f} GB，继续下载...")
            
            # 设置请求头，支持断点续传
            headers = {}
            if downloaded_size > 0:
                headers['Range'] = f'bytes={downloaded_size}-'
            
            # 发送请求，禁用 SSL 验证
            response = requests.get(
                url, 
                stream=True, 
                headers=headers,
                verify=False,  # 禁用 SSL 验证，解决 SSL 错误
                timeout=60
            )
            response.raise_for_status()
            
            # 获取文件总大小
            total_size = int(response.headers.get('content-length', 0)) + downloaded_size
            
            # 确定写入模式
            mode = 'ab' if downloaded_size > 0 else 'wb'
            
            # 下载文件
            with open(save_path, mode) as f:
                with tqdm(total=total_size, initial=downloaded_size, unit='B', unit_scale=True, desc=os.path.basename(save_path)) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # 验证文件大小
            final_size = os.path.getsize(save_path)
            if final_size == total_size:
                print(f"✅ 下载成功: {os.path.basename(save_path)}")
                return True
            else:
                print(f"⚠️ 文件大小不匹配: {final_size} != {total_size}")
                # 继续重试
                
        except Exception as e:
            print(f"❌ 下载失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                return False
    
    return False

def main():
    # 模型保存路径
    model_path = os.path.join(os.path.dirname(__file__), "models", "blip-image-captioning-base")
    os.makedirs(model_path, exist_ok=True)
    
    print("="*60)
    print("BLIP 模型下载工具 (支持断点续传)")
    print("="*60)
    print(f"保存路径: {os.path.abspath(model_path)}")
    print("="*60)
    
    # 文件列表
    files = {
        "config.json": "https://hf-mirror.com/Salesforce/blip-image-captioning-base/resolve/main/config.json",
        "preprocessor_config.json": "https://hf-mirror.com/Salesforce/blip-image-captioning-base/resolve/main/preprocessor_config.json",
        "vocab.json": "https://hf-mirror.com/Salesforce/blip-image-captioning-base/resolve/main/vocab.json",
        "merges.txt": "https://hf-mirror.com/Salesforce/blip-image-captioning-base/resolve/main/merges.txt",
        "special_tokens_map.json": "https://hf-mirror.com/Salesforce/blip-image-captioning-base/resolve/main/special_tokens_map.json",
        "pytorch_model.bin": "https://hf-mirror.com/Salesforce/blip-image-captioning-base/resolve/main/pytorch_model.bin"
    }
    
    success_count = 0
    for filename, url in files.items():
        save_path = os.path.join(model_path, filename)
        
        # 检查文件是否已存在且完整
        if os.path.exists(save_path):
            size = os.path.getsize(save_path) / (1024*1024)
            if filename == "pytorch_model.bin" and size > 900:
                print(f"\n✅ {filename} 已存在 ({size:.1f} MB)")
                success_count += 1
                continue
            elif filename != "pytorch_model.bin" and size > 0:
                print(f"\n✅ {filename} 已存在 ({size:.1f} KB)")
                success_count += 1
                continue
        
        print(f"\n下载: {filename}")
        if download_file_with_resume(url, save_path):
            success_count += 1
        else:
            print(f"❌ {filename} 下载失败")
            print(f"   手动下载链接: {url}")
    
    print("\n" + "="*60)
    print(f"下载完成: {success_count}/{len(files)}")
    print("="*60)
    
    if success_count == len(files):
        print("🎉 所有文件下载成功！")
        return True
    else:
        print("\n⚠️ 部分文件下载失败")
        print("可以使用浏览器下载缺失的文件，或尝试:")
        print(f"cd {model_path}")
        print("curl -L -O https://hf-mirror.com/Salesforce/blip-image-captioning-base/resolve/main/pytorch_model.bin")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)