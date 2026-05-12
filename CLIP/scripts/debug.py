"""
debug.py
项目调试脚本 - 适配当前项目结构（所有代码在 scripts 文件夹）
"""

import os
import sys

# 添加当前目录到路径（确保能找到同级模块）
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

from debug_utils import Debugger, ValidationChecker


def check_environment():
    """检查运行环境"""
    print("\n" + "="*60)
    print("1. 环境检查")
    print("="*60)
    
    # Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查GPU
    Debugger.check_gpu()
    
    # 检查虚拟环境
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"虚拟环境: {'✅ 已激活' if in_venv else '⚠️ 未激活'}")


def check_files():
    """检查必要文件"""
    print("\n" + "="*60)
    print("2. 文件检查")
    print("="*60)
    
    # 项目根目录是 scripts 的上一级
    project_root = os.path.dirname(current_dir)
    
    print(f"项目根目录: {project_root}")
    
    # 检查各个文件夹
    folders = [
        ('images文件夹', os.path.join(project_root, 'images')),
        ('scripts文件夹', current_dir),
        ('outputs文件夹', os.path.join(project_root, 'outputs')),
    ]
    
    for name, path in folders:
        ValidationChecker.check_folder_exists(path)
    
    # 检查测试图片
    test_image = os.path.join(project_root, 'images', 'test.jpg')
    if os.path.exists(test_image):
        from PIL import Image
        img = Image.open(test_image)
        print(f"✅ 测试图片: {test_image} ({img.size[0]}x{img.size[1]})")
    else:
        print(f"⚠️ 测试图片不存在，请放入: {test_image}")
        print("   可以任意放一张图片，重命名为 test.jpg")


def test_imports():
    """测试模块导入"""
    print("\n" + "="*60)
    print("3. 模块导入测试")
    print("="*60)
    
    modules = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('PIL', 'Pillow'),
        ('open_clip', 'OpenCLIP'),
        ('matplotlib', 'Matplotlib'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('tqdm', 'TQDM'),
    ]
    
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError as e:
            print(f"❌ {display_name}: {e}")
    
    # 检查自定义模块
    print("\n自定义模块:")
    custom_modules = [
        'blip_caption',
        'rope_module',
        'transformer_model',
        'clip_interrogator',
        'tag_library',
        'postprocessor',
        'clip_score',
        'model_comparison',
    ]
    
    for module_name in custom_modules:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name}")
        except ImportError as e:
            print(f"  ❌ {module_name}: {e}")


def test_blip():
    """测试BLIP模块"""
    print("\n" + "="*60)
    print("4. BLIP模块测试")
    print("="*60)
    
    try:
        from blip_caption import BLIPCaptionGenerator
        print("✅ BLIP模块导入成功")
        
        project_root = os.path.dirname(current_dir)
        test_image = os.path.join(project_root, 'images', 'test.jpg')
        
        if os.path.exists(test_image):
            print("加载BLIP模型...")
            blip = BLIPCaptionGenerator()
            caption = blip.generate_caption(test_image)
            print(f"✅ 图像描述: {caption[:100]}...")
        else:
            print(f"⚠️ 测试图像不存在，跳过")
        
        return True
    except Exception as e:
        print(f"❌ BLIP测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rope():
    """测试RoPE模块"""
    print("\n" + "="*60)
    print("5. RoPE模块测试")
    print("="*60)
    
    try:
        from rope_module import RotaryPositionalEncoding
        print("✅ RoPE模块导入成功")
        
        # 测试RoPE
        batch, seq_len, dim = 2, 10, 64
        q = torch.randn(batch, seq_len, dim)
        k = torch.randn(batch, seq_len, dim)
        
        rope = RotaryPositionalEncoding(dim=dim, max_seq_len=20)
        q_rot, k_rot = rope(q, k)
        
        print(f"  输入形状: q={q.shape}")
        print(f"  输出形状: q_rot={q_rot.shape}")
        print(f"  RoPE生效: {not torch.allclose(q, q_rot)}")
        print("✅ RoPE测试通过")
        
        return True
    except Exception as e:
        print(f"❌ RoPE测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tag_library():
    """测试标签库模块"""
    print("\n" + "="*60)
    print("6. 标签库模块测试")
    print("="*60)
    
    try:
        from tag_library import TagLibrary
        print("✅ 标签库模块导入成功")
        
        # 创建标签库（自动生成默认库）
        library = TagLibrary(auto_create=True)
        print(f"✅ 标签库创建: {library.total_tags()} 个标签")
        print(f"   类别: {list(library.library.keys())[:5]}...")
        
        # 测试获取标签
        all_tags = library.get_all_tags()
        print(f"   总标签数: {len(all_tags)}")
        print(f"   前5个: {all_tags[:5]}")
        
        return True
    except Exception as e:
        print(f"❌ 标签库测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clip_interrogator():
    """测试CLIP Interrogator模块"""
    print("\n" + "="*60)
    print("7. CLIP Interrogator测试")
    print("="*60)
    
    try:
        from tag_library import TagLibrary
        from clip_interrogator import CLIPInterrogator
        print("✅ 模块导入成功")
        
        # 创建标签库
        library = TagLibrary(auto_create=True)
        print(f"✅ 标签库: {library.total_tags()} 个标签")
        
        # 创建Interrogator
        print("加载CLIP模型...")
        ci = CLIPInterrogator(
            model_name='ViT-B-32',
            device='cpu',
            tag_library=library.library
        )
        print("✅ CLIP Interrogator创建成功")
        
        # 测试匹配
        project_root = os.path.dirname(current_dir)
        test_image = os.path.join(project_root, 'images', 'test.jpg')
        
        if os.path.exists(test_image):
            print(f"\n测试图像: {test_image}")
            matches = ci.match_tags(test_image, top_k=5)
            print("✅ 标签匹配结果:")
            for tag, score, cat in matches:
                print(f"   [{cat}] {tag}: {score:.4f}")
        else:
            print(f"⚠️ 测试图像不存在，跳过")
        
        return True
    except Exception as e:
        print(f"❌ CLIP Interrogator测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_postprocessor():
    """测试后处理模块"""
    print("\n" + "="*60)
    print("8. 后处理模块测试")
    print("="*60)
    
    try:
        from postprocessor import TextPostProcessor
        print("✅ 后处理模块导入成功")
        
        processor = TextPostProcessor()
        
        # 测试去重
        tags = ['dog', 'cat', 'dog', 'bird', 'cat']
        unique = processor.deduplicate(tags)
        print(f"  去重: {tags} -> {unique}")
        
        # 测试融合
        merged = processor.merge_tags(['beautiful', 'sunset', 'golden'])
        print(f"  融合: {merged}")
        
        # 测试格式化
        prompt = processor.format_prompt(
            blip_caption="A beautiful sunset over the ocean",
            tags=['sunset', 'ocean', 'golden hour'],
            format_type='standard'
        )
        print(f"  格式化: {prompt}")
        
        # 测试语法修正
        fixed = processor.fix_grammar("a beautiful sunset over the ocean")
        print(f"  语法修正: {fixed}")
        
        print("✅ 后处理模块测试通过")
        return True
    except Exception as e:
        print(f"❌ 后处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clip_score():
    """测试CLIP评分模块"""
    print("\n" + "="*60)
    print("9. CLIP评分模块测试")
    print("="*60)
    
    try:
        from clip_score import CLIPScorer
        print("✅ CLIP评分模块导入成功")
        
        # 创建评分器
        scorer = CLIPScorer()
        print("✅ CLIP评分器创建成功")
        
        # 测试评分
        project_root = os.path.dirname(current_dir)
        test_image = os.path.join(project_root, 'images', 'test.jpg')
        
        if os.path.exists(test_image):
            score = scorer.compute_score(test_image, "a photo")
            print(f"  测试评分: {score:.4f}")
        else:
            print(f"⚠️ 测试图像不存在，跳过")
        
        return True
    except Exception as e:
        print(f"❌ CLIP评分测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主调试函数"""
    print("="*60)
    print("🔧 CLIP图生文项目 - 调试工具")
    print("="*60)
    
    # 运行所有检查
    check_environment()
    check_files()
    test_imports()
    
    # 模块测试（按依赖顺序）
    test_blip()
    test_rope()
    test_tag_library()
    test_clip_interrogator()
    test_postprocessor()
    test_clip_score()
    
    print("\n" + "="*60)
    print("调试完成！")
    print("="*60)


if __name__ == "__main__":
    # 导入torch用于测试
    import torch
    main()