"""
查询路由器 - 智能路由查询到合适的回答策略
"""
from typing import List, Dict, Any
from langchain.schema import Document
from logging_setup import logger
from config import RELEVANCE_THRESHOLD, MAX_SEARCH_RESULTS
from relevance_evaluator import RelevanceEvaluator


class QueryRouter:
    """查询路由器"""
    
    def __init__(
        self,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
        max_results: int = MAX_SEARCH_RESULTS
    ):
        """
        初始化查询路由器
        
        Args:
            relevance_threshold: 相关性阈值
            max_results: 最大检索结果数
        """
        self.relevance_threshold = relevance_threshold
        self.max_results = max_results
        self.evaluator = RelevanceEvaluator(threshold=relevance_threshold)
        logger.info(f"QueryRouter初始化，相关性阈值: {self.relevance_threshold}")
    
    def route(
        self,
        query: str,
        vector_store,
        user_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        路由查询到合适的回答策略
        
        Args:
            query: 用户查询
            vector_store: 向量存储实例
            user_context: 用户上下文
        
        Returns:
            路由结果字典，包含：
            - strategy: 回答策略 ('document_based' 或 'general_knowledge')
            - documents: 相关文档列表
            - evaluation: 相关性评估结果
        """
        # 1. 向量检索
        logger.info(f"执行向量检索: query='{query}', k={self.max_results}")
        documents = self._search_documents(vector_store, query)
        
        if not documents:
            logger.info("未找到任何文档，使用通用知识回答")
            return {
                'strategy': 'general_knowledge',
                'documents': [],
                'evaluation': {
                    'is_relevant': False,
                    'max_score': 0.0,
                    'avg_score': 0.0
                }
            }
        
        logger.info(f"向量检索完成: 找到 {len(documents)} 个文档")
        
        # 2. 相关性评估
        evaluation = self.evaluator.evaluate(query, documents)
        
        # 3. 决策路由策略
        if evaluation['is_relevant']:
            strategy = 'document_based'
            relevant_docs = evaluation['relevant_docs']
            logger.info(f"选择回答策略: {strategy}（使用 {len(relevant_docs)} 个相关文档）")
        else:
            strategy = 'general_knowledge'
            relevant_docs = []
            logger.info(f"选择回答策略: {strategy}")
        
        return {
            'strategy': strategy,
            'documents': relevant_docs,
            'evaluation': evaluation
        }
    
    def _search_documents(
        self,
        vector_store,
        query: str,
        filter_dict: Dict = None
    ) -> List[Document]:
        """
        执行向量搜索
        
        Args:
            vector_store: 向量存储实例
            query: 查询文本
            filter_dict: 过滤条件
        
        Returns:
            文档列表
        """
        try:
            # 使用similarity_search_with_score获取带分数的结果
            results = vector_store.similarity_search_with_score(
                query,
                k=self.max_results,
                filter=filter_dict
            )
            
            # 转换为Document对象，并添加分数到metadata
            documents = []
            for doc, score in results:
                # 将距离转换为相似度分数（假设使用余弦距离）
                # 余弦距离范围[0, 2]，转换为相似度[0, 1]
                similarity_score = max(0.0, 1.0 - (score / 2.0))
                
                # 添加分数到metadata
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata['score'] = similarity_score
                
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []


def create_query_router(
    relevance_threshold: float = None,
    max_results: int = None
) -> QueryRouter:
    """创建查询路由器的工厂函数"""
    return QueryRouter(
        relevance_threshold=relevance_threshold or RELEVANCE_THRESHOLD,
        max_results=max_results or MAX_SEARCH_RESULTS
    )
