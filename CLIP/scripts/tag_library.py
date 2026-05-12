"""
tag_library.py - 增强版
自动生成标签库，无需外部文件
"""

import json
import os
from typing import Dict, List, Optional
import random


class TagLibrary:
    """
    提示词库管理类 - 自带默认标签库
    """
    
    def __init__(self, library_path: str = None, auto_create: bool = True):
        """
        初始化标签库
        
        Args:
            library_path: 标签库JSON文件路径
            auto_create: 如果没有文件，是否自动创建默认标签库
        """
        self.library = {}
        
        # 尝试从文件加载
        if library_path and os.path.exists(library_path):
            self.load(library_path)
        elif auto_create:
            # 自动创建默认标签库
            self._create_default_library()
            # 如果提供了路径，自动保存
            if library_path:
                self.save(library_path)
        else:
            self.library = {}
        
        print(f"✅ 标签库加载完成: {self.total_tags()} 个标签，{len(self.library)} 个类别")
    
    def _create_default_library(self):
        """创建完整的默认标签库"""
        self.library = {
            # ========== 物体类 ==========
            'animals': [
                'dog', 'cat', 'bird', 'fish', 'horse', 'cow', 'sheep', 'rabbit',
                'elephant', 'lion', 'tiger', 'bear', 'monkey', 'fox', 'wolf',
                'butterfly', 'bee', 'ladybug', 'spider', 'snake', 'deer', 'squirrel',
                'panda', 'koala', 'kangaroo', 'giraffe', 'zebra', 'hippo', 'rhino'
            ],
            'vehicles': [
                'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'train', 'airplane',
                'boat', 'ship', 'helicopter', 'tractor', 'ambulance', 'fire truck',
                'police car', 'taxi', 'van', 'scooter', 'skateboard', 'motorboat'
            ],
            'nature': [
                'tree', 'flower', 'grass', 'mountain', 'river', 'lake', 'ocean',
                'forest', 'desert', 'beach', 'sunset', 'sunrise', 'cloud', 'rainbow',
                'snow', 'ice', 'rock', 'stone', 'leaf', 'petal', 'waterfall', 'cave'
            ],
            'buildings': [
                'house', 'building', 'castle', 'bridge', 'tower', 'church', 'temple',
                'skyscraper', 'cabin', 'barn', 'lighthouse', 'windmill', 'pyramid',
                'palace', 'museum', 'library', 'school', 'hospital', 'factory', 'hotel'
            ],
            'people': [
                'person', 'man', 'woman', 'child', 'baby', 'family', 'couple',
                'group', 'crowd', 'artist', 'worker', 'soldier', 'doctor', 'teacher',
                'athlete', 'dancer', 'singer', 'photographer', 'businessman'
            ],
            'food': [
                'apple', 'banana', 'orange', 'strawberry', 'pizza', 'burger', 'cake',
                'coffee', 'tea', 'wine', 'bread', 'cheese', 'salad', 'soup', 'rice'
            ],
            
            # ========== 场景类 ==========
            'scenes': [
                'beach', 'forest', 'desert', 'mountain', 'city', 'countryside',
                'urban', 'rural', 'indoor', 'outdoor', 'street', 'park', 'garden',
                'farm', 'office', 'classroom', 'kitchen', 'bedroom', 'living room',
                'bathroom', 'balcony', 'rooftop', 'underwater', 'space', 'night',
                'day', 'sunny', 'rainy', 'snowy', 'foggy', 'stormy', 'sunset'
            ],
            
            # ========== 风格类 ==========
            'art_styles': [
                'realistic', 'photorealistic', 'hyperrealistic', 'cartoon',
                'anime', 'manga', 'sketch', 'drawing', 'illustration',
                'oil painting', 'watercolor', 'acrylic', 'pastel', 'charcoal',
                'digital art', 'vector art', 'pixel art', 'low poly', '3d render',
                'vintage', 'retro', 'modern', 'minimalist', 'abstract'
            ],
            'art_movements': [
                'impressionism', 'expressionism', 'cubism', 'surrealism',
                'pop art', 'abstract expressionism', 'art nouveau', 'baroque',
                'renaissance', 'gothic', 'romanticism', 'realism'
            ],
            'photography': [
                'photography', 'macro photography', 'long exposure', 'bokeh',
                'hdr', 'black and white', 'sepia', 'polaroid', 'film grain',
                'portrait', 'landscape', 'street photography', 'wildlife'
            ],
            
            # ========== 颜色类 ==========
            'colors': [
                'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
                'brown', 'black', 'white', 'gray', 'gold', 'silver', 'bronze',
                'cyan', 'magenta', 'teal', 'indigo', 'violet', 'crimson', 'scarlet',
                'emerald', 'sapphire', 'ruby', 'amber', 'ivory', 'beige', 'lavender'
            ],
            
            # ========== 质量/属性类 ==========
            'qualities': [
                'beautiful', 'cute', 'ugly', 'scary', 'funny', 'sad', 'happy',
                'bright', 'dark', 'sharp', 'blurry', 'detailed', 'simple',
                'colorful', 'monochrome', 'vibrant', 'muted', 'soft', 'harsh',
                'warm', 'cool', 'mysterious', 'magical', 'dreamy', 'eerie',
                'peaceful', 'chaotic', 'elegant', 'rustic', 'modern'
            ],
            
            # ========== 动作/状态类 ==========
            'actions': [
                'running', 'walking', 'sitting', 'standing', 'lying', 'jumping',
                'flying', 'swimming', 'eating', 'drinking', 'sleeping', 'working',
                'playing', 'reading', 'writing', 'thinking', 'smiling', 'crying',
                'dancing', 'singing', 'talking', 'fighting', 'driving', 'flying'
            ],
            
            # ========== 时间/季节类 ==========
            'time': [
                'morning', 'afternoon', 'evening', 'night', 'dawn', 'dusk',
                'spring', 'summer', 'autumn', 'fall', 'winter'
            ]
        }
    
    def load(self, file_path: str):
        """从JSON文件加载标签库"""
        with open(file_path, 'r', encoding='utf-8') as f:
            self.library = json.load(f)
        print(f"📂 从文件加载标签库: {file_path}")
    
    def save(self, file_path: str):
        """保存标签库到JSON文件"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.library, f, ensure_ascii=False, indent=2)
        print(f"💾 标签库已保存: {file_path}")
    
    def total_tags(self) -> int:
        """获取标签总数"""
        total = 0
        for value in self.library.values():
            if isinstance(value, list):
                total += len(value)
            elif isinstance(value, dict):
                for sublist in value.values():
                    total += len(sublist)
        return total
    
    def get_all_tags(self) -> List[str]:
        """获取所有标签（扁平列表）"""
        tags = []
        for value in self.library.values():
            if isinstance(value, list):
                tags.extend(value)
            elif isinstance(value, dict):
                for sublist in value.values():
                    tags.extend(sublist)
        return tags
    
    def get_tags_by_category(self, category: str) -> List[str]:
        """按类别获取标签"""
        if category in self.library:
            if isinstance(self.library[category], list):
                return self.library[category]
            elif isinstance(self.library[category], dict):
                all_tags = []
                for sublist in self.library[category].values():
                    all_tags.extend(sublist)
                return all_tags
        return []
    
    def add_tag(self, category: str, tag: str, subcategory: str = None):
        """添加标签"""
        if category not in self.library:
            if subcategory:
                self.library[category] = {subcategory: []}
            else:
                self.library[category] = []
        
        target = self.library[category]
        if subcategory:
            if subcategory not in target:
                target[subcategory] = []
            if tag not in target[subcategory]:
                target[subcategory].append(tag)
        else:
            if tag not in target:
                target.append(tag)
    
    def show_summary(self):
        """显示标签库摘要"""
        print("\n" + "="*50)
        print("标签库摘要")
        print("="*50)
        for category, tags in self.library.items():
            if isinstance(tags, list):
                count = len(tags)
                print(f"  {category}: {count} 个标签")
            elif isinstance(tags, dict):
                total = sum(len(v) for v in tags.values())
                print(f"  {category}: {total} 个标签 (子分类: {list(tags.keys())})")
        print(f"\n总计: {self.total_tags()} 个标签")
        print("="*50)


# 快速测试
if __name__ == "__main__":
    print("="*50)
    print("标签库测试 - 无文件自动创建")
    print("="*50)
    
    # 自动创建默认标签库
    library = TagLibrary(auto_create=True)
    library.show_summary()
    
    # 测试获取所有标签
    all_tags = library.get_all_tags()
    print(f"\n前10个标签: {all_tags[:10]}")
    
    # 测试按类别获取
    animals = library.get_tags_by_category('animals')
    print(f"动物类标签: {animals[:5]}...")