"""
相关性评估器 - 评估检索结果与查询的相关性
"""
from typing import List, Dict, Any
from langchain.schema import Document
from logging_setup import logger
from config import RELEVANCE_THRESHOLD


class RelevanceEvaluator:
    """相关性评估器"""
    
    def __init__(self, threshold: float = RELEVANCE_THRESHOLD):
        """
        初始化相关性评估器
        
        Args:
            threshold: 相关性阈值（0-1之间），默认从配置读取
        """
        self.threshold = threshold
        logger.info(f"RelevanceEvaluator初始化，阈值: {self.threshold}")
    
    def evaluate(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """
        评估文档与查询的相关性
        
        Args:
            query: 用户查询
            documents: 检索到的文档列表
        
        Returns:
            评估结果字典，包含：
            - is_relevant: 是否相关（bool）
            - relevant_docs: 相关文档列表
            - max_score: 最高相似度分数
            - avg_score: 平均相似度分数
            - relevant_count: 相关文档数量
        """
        if not documents:
            return {
                'is_relevant': False,
                'relevant_docs': [],
                'max_score': 0.0,
                'avg_score': 0.0,
                'relevant_count': 0,
                'relevant_ratio': 0.0
            }
        
        # 提取相似度分数
        scores = []
        for doc in documents:
            # 从metadata中获取分数，如果没有则使用默认值
            score = doc.metadata.get('score', 0.0) if hasattr(doc, 'metadata') else 0.0
            scores.append(score)
        
        # 计算统计指标
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # 筛选相关文档
        relevant_docs = []
        for doc, score in zip(documents, scores):
            if score >= self.threshold:
                relevant_docs.append(doc)
        
        relevant_count = len(relevant_docs)
        relevant_ratio = relevant_count / len(documents) if documents else 0.0
        
        # 判断是否相关
        # 策略：最高分超过阈值 或 平均分超过阈值*0.8
        is_relevant = (max_score >= self.threshold) or (avg_score >= self.threshold * 0.8)
        
        logger.info(
            f"相关性评估完成: is_relevant={is_relevant}, "
            f"max_score={max_score:.3f}, avg_score={avg_score:.3f}, "
            f"relevant_count={relevant_count}/{len(documents)}, "
            f"relevant_ratio={relevant_ratio:.2f}"
        )
        
        return {
            'is_relevant': is_relevant,
            'relevant_docs': relevant_docs if is_relevant else [],
            'max_score': max_score,
            'avg_score': avg_score,
            'relevant_count': relevant_count,
            'relevant_ratio': relevant_ratio
        }
    
    def filter_by_threshold(self, documents: List[Document]) -> List[Document]:
        """
        根据阈值过滤文档
        
        Args:
            documents: 文档列表
        
        Returns:
            过滤后的文档列表
        """
        filtered = []
        for doc in documents:
            score = doc.metadata.get('score', 0.0) if hasattr(doc, 'metadata') else 0.0
            if score >= self.threshold:
                filtered.append(doc)
        
        logger.info(f"筛选相关文档: 总数={len(documents)}, 相关={len(filtered)}, 阈值={self.threshold}")
        return filtered
    
    def adjust_threshold(self, new_threshold: float):
        """动态调整阈值"""
        if 0.0 <= new_threshold <= 1.0:
            old_threshold = self.threshold
            self.threshold = new_threshold
            logger.info(f"阈值已调整: {old_threshold} -> {new_threshold}")
        else:
            logger.warning(f"无效的阈值: {new_threshold}，必须在0-1之间")


def create_relevance_evaluator(threshold: float = None) -> RelevanceEvaluator:
    """创建相关性评估器的工厂函数"""
    return RelevanceEvaluator(threshold=threshold or RELEVANCE_THRESHOLD)
