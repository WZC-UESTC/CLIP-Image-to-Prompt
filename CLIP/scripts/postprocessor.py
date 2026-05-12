"""
postprocessor.py
文本后处理模块
包含：排序、去重、融合、语法修正等
"""

import re
from typing import List, Dict, Tuple, Optional
from collections import Counter


class TextPostProcessor:
    """
    文本后处理器
    对生成的提示词进行优化处理
    """
    
    def __init__(self):
        """初始化后处理器"""
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> set:
        """加载停用词"""
        return {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
                'of', 'to', 'for', 'with', 'by', 'from', 'as', 'be', 'this', 'that'}
    
    def rank_by_clip_score(self, 
                           candidates: List[Tuple[str, float]], 
                           clip_scorer=None,
                           image_path: str = None) -> List[Tuple[str, float]]:
        """
        基于CLIP得分排序
        
        Args:
            candidates: [(text, original_score), ...]
            clip_scorer: CLIPScorer实例
            image_path: 图像路径（用于计算CLIP得分）
        
        Returns:
            排序后的列表
        """
        if not clip_scorer or not image_path:
            # 如果没有CLIP评分器，按原分数排序
            return sorted(candidates, key=lambda x: x[1], reverse=True)
        
        # 重新计算CLIP得分
        scored = []
        for text, _ in candidates:
            clip_score = clip_scorer.compute_score(image_path, text)
            scored.append((text, clip_score))
        
        return sorted(scored, key=lambda x: x[1], reverse=True)
    
    def rank_by_frequency(self, 
                          candidates: List[Tuple[str, float]],
                          frequency_dict: Dict[str, int] = None) -> List[Tuple[str, float]]:
        """
        基于词频排序（高频词优先）
        
        Args:
            candidates: [(text, original_score), ...]
            frequency_dict: 词频字典，如 {'dog': 100, 'cat': 80}
        
        Returns:
            排序后的列表
        """
        if not frequency_dict:
            return candidates
        
        def get_frequency(text):
            words = text.lower().split()
            freq = sum(frequency_dict.get(word, 0) for word in words)
            return freq
        
        # 综合得分：原始分数 + 词频权重
        scored = []
        for text, orig_score in candidates:
            freq_score = get_frequency(text) / 100  # 归一化
            combined = orig_score + 0.1 * freq_score
            scored.append((text, combined))
        
        return sorted(scored, key=lambda x: x[1], reverse=True)
    
    def rank_by_combined(self,
                         candidates: List[Tuple[str, float]],
                         clip_scorer=None,
                         image_path: str = None,
                         frequency_dict: Dict[str, int] = None,
                         weights: Dict[str, float] = None) -> List[Tuple[str, float]]:
        """
        综合排序（CLIP得分 + 词频 + 长度）
        
        Args:
            candidates: [(text, original_score), ...]
            clip_scorer: CLIP评分器
            image_path: 图像路径
            frequency_dict: 词频字典
            weights: 权重 {'clip': 0.6, 'freq': 0.2, 'length': 0.2}
        
        Returns:
            排序后的列表
        """
        if weights is None:
            weights = {'clip': 0.6, 'freq': 0.2, 'length': 0.2}
        
        scored = []
        for text, _ in candidates:
            total_score = 0
            
            # CLIP得分
            if clip_scorer and image_path:
                clip_score = clip_scorer.compute_score(image_path, text)
                total_score += weights['clip'] * clip_score
            else:
                total_score += weights['clip'] * 0.5  # 默认分数
            
            # 词频得分
            if frequency_dict:
                words = text.lower().split()
                freq_score = sum(frequency_dict.get(word, 0) for word in words) / 100
                total_score += weights['freq'] * min(freq_score, 1.0)
            
            # 长度得分（适中长度得分高）
            length = len(text.split())
            length_score = 1.0 - abs(length - 10) / 20
            length_score = max(0, min(1, length_score))
            total_score += weights['length'] * length_score
            
            scored.append((text, total_score))
        
        return sorted(scored, key=lambda x: x[1], reverse=True)
    
    def deduplicate(self, tags: List[str]) -> List[str]:
        """
        去重（保留第一次出现）
        
        Args:
            tags: 标签列表
        
        Returns:
            去重后的列表
        """
        seen = set()
        result = []
        for tag in tags:
            if tag.lower() not in seen:
                seen.add(tag.lower())
                result.append(tag)
        return result
    
    def merge_tags(self, tags: List[str], separator: str = ', ') -> str:
        """
        融合标签为字符串
        
        Args:
            tags: 标签列表
            separator: 分隔符
        
        Returns:
            融合后的字符串
        """
        return separator.join(tags)
    
    def format_prompt(self, 
                      blip_caption: str, 
                      tags: List[str],
                      format_type: str = 'standard') -> str:
        """
        格式化最终提示词
        
        Args:
            blip_caption: BLIP生成的描述
            tags: CLIP匹配的标签列表
            format_type: 格式类型 ('standard', 'comma', 'with_category')
        
        Returns:
            格式化后的提示词
        """
        if format_type == 'standard':
            # 标准格式：描述 + 标签
            tag_str = ', '.join(tags[:5])
            return f"{blip_caption}, {tag_str}"
        
        elif format_type == 'comma':
            # 纯逗号分隔
            parts = [blip_caption] + tags
            return ', '.join(parts)
        
        elif format_type == 'with_category':
            # 带类别标签
            return f"{blip_caption} featuring {', '.join(tags)}"
        
        else:
            return f"{blip_caption} {', '.join(tags)}"
    
    def fix_grammar(self, text: str) -> str:
        """
        简单语法修正
        
        Args:
            text: 原始文本
        
        Returns:
            修正后的文本
        """
        # 首字母大写
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # 确保以句号结尾
        if text and not text[-1] in '.!?':
            text += '.'
        
        # 修复常见缩写
        text = text.replace(" 's", "'s")
        text = text.replace(" n't", "n't")
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def remove_stopwords(self, tags: List[str]) -> List[str]:
        """
        移除停用词
        
        Args:
            tags: 标签列表
        
        Returns:
            过滤后的标签列表
        """
        return [tag for tag in tags if tag.lower() not in self.stopwords]
    
    def truncate(self, text: str, max_length: int = 100) -> str:
        """
        截断文本
        
        Args:
            text: 原始文本
            max_length: 最大长度
        
        Returns:
            截断后的文本
        """
        if len(text) <= max_length:
            return text
        
        # 在最后一个单词处截断
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        if last_space > 0:
            truncated = truncated[:last_space]
        
        return truncated + "..."
    
    def build_frequency_dict(self, tag_list: List[str]) -> Dict[str, int]:
        """
        构建词频字典
        
        Args:
            tag_list: 标签列表
        
        Returns:
            {word: frequency}
        """
        all_words = []
        for tag in tag_list:
            all_words.extend(tag.lower().split())
        
        return dict(Counter(all_words))


# 测试代码
if __name__ == "__main__":
    print("="*50)
    print("文本后处理器测试")
    print("="*50)
    
    processor = TextPostProcessor()
    
    # 测试去重
    tags = ['dog', 'cat', 'dog', 'bird', 'cat']
    unique = processor.deduplicate(tags)
    print(f"去重前: {tags}")
    print(f"去重后: {unique}")
    
    # 测试融合
    merged = processor.merge_tags(['beautiful', 'sunset', 'ocean'])
    print(f"融合: {merged}")
    
    # 测试格式化
    prompt = processor.format_prompt(
        blip_caption="A beautiful sunset over the ocean",
        tags=['sunset', 'ocean', 'golden hour', 'peaceful'],
        format_type='standard'
    )
    print(f"格式化: {prompt}")
    
    # 测试语法修正
    fixed = processor.fix_grammar("a beautiful sunset over the ocean")
    print(f"语法修正: {fixed}")