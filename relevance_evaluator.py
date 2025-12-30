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
        
        # 先进行语义相关性判断
        if not self._is_hr_related_query(query):
            logger.info(f"查询'{query}'与人事领域无关，判定为不相关")
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
        # 策略：最高分必须超过阈值，且至少有一个文档相关
        # 如果最高分很低（< 0.3），直接判定为不相关
        if max_score < 0.3:
            is_relevant = False
            logger.info(f"最高分过低({max_score:.3f} < 0.3)，判定为不相关")
        else:
            is_relevant = (max_score >= self.threshold) and (relevant_count > 0)
        
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
    
    def _is_hr_related_query(self, query: str) -> bool:
        """
        判断查询是否与人事领域相关
        
        Args:
            query: 用户查询
        
        Returns:
            是否与人事相关
        """
        # 人事相关关键词
        hr_keywords = [
            '考勤', '打卡', '请假', '休假', '迟到', '早退', '出勤',
            '薪资', '工资', '薪酬', '奖金', '发放', '薪水',
            '入职', '新员工', '报到', '入职手续',
            '离职', '辞职', '退休', '离职手续', '离职流程',
            '培训', '学习', '发展', '课程', '培训计划',
            '福利', '待遇', '补贴', '津贴', '福利待遇',
            '流程', '步骤', '程序', '办理',
            '政策', '制度', '规定', '条例',
            '资产', '领用', '归还', '报废',
            '绩效', '考核', '评估',
            '招聘', '面试', '录用',
            '合同', '协议', '劳动合同',
            '社保', '公积金', '五险一金',
            '加班', '调休', '值班',
            '部门', '岗位', '职位',
            '员工', '人员', '职工'
        ]
        
        # 非人事相关的明显标志
        non_hr_patterns = [
            # 数学计算
            r'^\d+[\+\-\*/]\d+',  # 1+2, 3-1, 4*5, 6/2
            r'=\?$',  # 以=?结尾
            # 技术问题
            'python', 'java', 'javascript', 'c\+\+', 'golang', 'rust',
            'qt', 'react', 'vue', 'angular',
            '编程', '代码', '算法', '数据结构',
            # 常识问题
            '什么是', '谁是', '哪里', '为什么',
            '天气', '新闻', '股票', '比赛'
        ]
        
        query_lower = query.lower()
        
        # 检查是否匹配非人事模式
        import re
        for pattern in non_hr_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"查询匹配非人事模式: {pattern}")
                return False
        
        # 检查是否包含人事关键词
        for keyword in hr_keywords:
            if keyword in query:
                logger.info(f"查询包含人事关键词: {keyword}")
                return True
        
        # 如果查询很短（<5个字符）且不包含人事关键词，可能是非人事问题
        if len(query) < 5:
            logger.info(f"查询过短且不包含人事关键词: {query}")
            return False
        
        # 默认认为可能与人事相关（保守策略）
        return True
    
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
