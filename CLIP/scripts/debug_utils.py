"""
debug_utils.py
项目调试工具集
"""

import torch
import time
import os
import sys
from typing import Any, Dict, List
import traceback


class Debugger:
    """调试器类"""
    
    @staticmethod
    def print_model_info(model, name="Model"):
        """打印模型信息"""
        print(f"\n{'='*50}")
        print(f"{name} 信息")
        print(f"{'='*50}")
        print(f"类型: {type(model)}")
        
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
            print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        
        if hasattr(model, 'device'):
            print(f"设备: {model.device}")
    
    @staticmethod
    def print_tensor_info(tensor, name="Tensor"):
        """打印张量信息"""
        print(f"\n{name}:")
        print(f"  形状: {tensor.shape}")
        print(f"  类型: {tensor.dtype}")
        print(f"  设备: {tensor.device}")
        print(f"  最小值: {tensor.min().item():.6f}")
        print(f"  最大值: {tensor.max().item():.6f}")
        print(f"  均值: {tensor.mean().item():.6f}")
        print(f"  标准差: {tensor.std().item():.6f}")
    
    @staticmethod
    def timeit(func):
        """计时装饰器"""
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"⏱️ {func.__name__} 耗时: {elapsed:.4f}s")
            return result
        return wrapper
    
    @staticmethod
    def check_gpu():
        """检查GPU状态"""
        print("\n" + "="*50)
        print("GPU 状态检查")
        print("="*50)
        
        if torch.cuda.is_available():
            print(f"✅ CUDA可用")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            print(f"   当前GPU: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"   当前显存使用: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        else:
            print("❌ CUDA不可用，使用CPU")
    
    @staticmethod
    def trace_call(func):
        """追踪函数调用"""
        def wrapper(*args, **kwargs):
            print(f"🔍 调用: {func.__name__}")
            print(f"   参数: {args[1:] if len(args) > 1 else '无'}")
            result = func(*args, **kwargs)
            print(f"   返回: {type(result)}")
            return result
        return wrapper


class ValidationChecker:
    """验证检查器"""
    
    @staticmethod
    def check_file_exists(file_path: str):
        """检查文件是否存在"""
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"✅ {file_path} ({size:.1f} KB)")
            return True
        else:
            print(f"❌ 文件不存在: {file_path}")
            return False
    
    @staticmethod
    def check_folder_exists(folder_path: str):
        """检查文件夹是否存在"""
        if os.path.exists(folder_path):
            try:
                files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                print(f"✅ {folder_path} ({len(files)} 个文件)")
            except:
                print(f"✅ {folder_path}")
            return True
        else:
            print(f"❌ 文件夹不存在: {folder_path}")
            return False
    
    @staticmethod
    def check_model_cache(model_name: str):
        """检查模型是否已缓存"""
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        if os.path.exists(cache_dir):
            # 查找包含模型名的文件夹
            matching = [f for f in os.listdir(cache_dir) if model_name.replace('/', '--') in f]
            if matching:
                print(f"✅ 模型已缓存: {matching[0]}")
                return True
        print(f"⚠️ 模型未缓存: {model_name}")
        return False
    
    @staticmethod
    def test_image_load(image_path: str):
        """测试图像加载"""
        from PIL import Image
        
        try:
            img = Image.open(image_path)
            print(f"✅ 图像加载成功: {image_path}")
            print(f"   尺寸: {img.size}")
            print(f"   模式: {img.mode}")
            return True
        except Exception as e:
            print(f"❌ 图像加载失败: {e}")
            return False


class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
    
    def add_test(self, name: str, func):
        """添加测试"""
        self.tests.append((name, func))
    
    def run_all(self):
        """运行所有测试"""
        print("\n" + "="*60)
        print("开始运行测试")
        print("="*60)
        
        for name, func in self.tests:
            try:
                print(f"\n📝 测试: {name}")
                result = func()
                if result is not False:
                    print(f"   ✅ 通过")
                    self.passed += 1
                else:
                    print(f"   ❌ 失败")
                    self.failed += 1
            except Exception as e:
                print(f"   ❌ 异常: {e}")
                traceback.print_exc()
                self.failed += 1
        
        print("\n" + "="*60)
        print(f"测试结果: {self.passed} 通过, {self.failed} 失败")
        print("="*60)
        
        return self.failed == 0