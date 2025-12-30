"""
查询路由器 - 智能路由查询到合适的回答策略
集成混合检索、查询重述和Rerank（dev-mix分支）
"""
from typing import List, Dict, Any
from langchain.schema import Document
from logging_setup import logger
from config import (
    RELEVANCE_THRESHOLD, MAX_SEARCH_RESULTS,
    USE_HYBRID_SEARCH, USE_QUERY_REPHRASE, USE_RERANK
)
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
        路由查询到合适的回答策略（集成混合检索、查询重述和Rerank）
        
        Args:
            query: 用户查询
            vector_store: 向量存储实例
            user_context: 用户上下文
        
        Returns:
            路由结果字典，包含：
            - strategy: 回答策略 ('document_based' 或 'general_knowledge')
            - documents: 相关文档列表
            - evaluation: 相关性评估结果
            - rephrased_query: 重述后的查询（如果启用）
        """
        # 1. 查询重述（dev-mix新增）
        rephrased_query = query
        if USE_QUERY_REPHRASE:
            try:
                from query_rephrase import rephrase_query
                rephrased_query = rephrase_query(query)
                logger.info(f"查询重述: '{query}' -> '{rephrased_query}'")
            except Exception as e:
                logger.warning(f"查询重述失败，使用原始查询: {e}")
        
        # 2. 混合检索（dev-mix新增）
        logger.info(f"执行{'混合' if USE_HYBRID_SEARCH else '向量'}检索: query='{rephrased_query}', k={self.max_results}")
        documents = self._search_documents(vector_store, rephrased_query)
        
        if not documents:
            logger.info("未找到任何文档，使用通用知识回答")
            return {
                'strategy': 'general_knowledge',
                'documents': [],
                'evaluation': {
                    'is_relevant': False,
                    'max_score': 0.0,
                    'avg_score': 0.0
                },
                'rephrased_query': rephrased_query
            }
        
        logger.info(f"检索完成: 找到 {len(documents)} 个文档")
        
        # 3. Rerank重排序（dev-mix新增）
        if USE_RERANK and len(documents) > 0:
            try:
                from reranker import rerank_documents
                documents = rerank_documents(rephrased_query, documents)
                logger.info(f"Rerank完成: 保留 {len(documents)} 个文档")
            except Exception as e:
                logger.warning(f"Rerank失败，使用原始排序: {e}")
        
        # 4. 相关性评估
        evaluation = self.evaluator.evaluate(rephrased_query, documents)
        
        # 5. 决策路由策略
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
            'evaluation': evaluation,
            'rephrased_query': rephrased_query
        }
    
    def _search_documents(
        self,
        vector_store,
        query: str,
        filter_dict: Dict = None
    ) -> List[Document]:
        """
        执行文档搜索（支持混合检索 - dev-mix新增）
        
        Args:
            vector_store: 向量存储实例
            query: 查询文本
            filter_dict: 过滤条件
        
        Returns:
            文档列表
        """
        try:
            # 如果启用混合检索
            if USE_HYBRID_SEARCH:
                try:
                    from hybrid_retriever import create_hybrid_retriever
                    from knowledge_base import get_vector_manager
                    
                    # 获取所有文档用于BM25
                    vector_manager = get_vector_manager()
                    if vector_manager:
                        all_documents = vector_manager.get_all_documents()
                        
                        if all_documents:
                            # 创建混合检索器
                            hybrid_retriever = create_hybrid_retriever(
                                vector_store=vector_store,
                                documents=all_documents,
                                k=self.max_results
                            )
                            
                            # 执行混合检索
                            documents = hybrid_retriever.get_relevant_documents(query)
                            
                            # 添加分数到metadata
                            for i, doc in enumerate(documents):
                                if not hasattr(doc, 'metadata'):
                                    doc.metadata = {}
                                # 基于排名的分数
                                doc.metadata['score'] = 1.0 - (i / len(documents)) if documents else 0.0
                            
                            logger.info(f"混合检索完成: 找到 {len(documents)} 个文档")
                            return documents
                        else:
                            logger.warning("文档缓存为空，降级到向量检索")
                    else:
                        logger.warning("向量管理器不可用，降级到向量检索")
                        
                except Exception as e:
                    logger.warning(f"混合检索失败，降级到向量检索: {e}")
            
            # 降级到纯向量检索
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
            logger.error(f"文档搜索失败: {e}")
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
