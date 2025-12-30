"""
混合检索器 - 结合向量检索和BM25检索
参考 General_Doc_QA_System/knowledge.py 的实现
"""
from typing import List, Optional
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from logging_setup import logger
from config import (
    USE_HYBRID_SEARCH, VECTOR_SEARCH_WEIGHT, BM25_SEARCH_WEIGHT,
    MAX_SEARCH_RESULTS
)


class HybridRetriever:
    """混合检索器 - 向量检索 + BM25检索"""
    
    def __init__(
        self,
        vector_store,
        documents: List[Document],
        vector_weight: float = VECTOR_SEARCH_WEIGHT,
        bm25_weight: float = BM25_SEARCH_WEIGHT,
        k: int = MAX_SEARCH_RESULTS
    ):
        """
        初始化混合检索器
        
        Args:
            vector_store: 向量存储实例
            documents: 所有文档列表（用于BM25）
            vector_weight: 向量检索权重
            bm25_weight: BM25检索权重
            k: 返回结果数量
        """
        self.vector_store = vector_store
        self.documents = documents
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.k = k
        
        # 初始化检索器
        self._init_retrievers()
        
        logger.info(
            f"混合检索器初始化完成: "
            f"向量权重={vector_weight}, BM25权重={bm25_weight}, "
            f"文档数={len(documents)}, k={k}"
        )
    
    def _init_retrievers(self):
        """初始化向量检索器和BM25检索器"""
        try:
            # 1. 向量检索器
            self.vector_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.k}
            )
            logger.info("向量检索器初始化成功")
            
            # 2. BM25检索器
            if self.documents:
                self.bm25_retriever = BM25Retriever.from_documents(
                    self.documents,
                    k=self.k
                )
                logger.info(f"BM25检索器初始化成功，文档数: {len(self.documents)}")
            else:
                logger.warning("文档列表为空，BM25检索器未初始化")
                self.bm25_retriever = None
            
            # 3. 集成检索器
            if self.bm25_retriever:
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[self.vector_retriever, self.bm25_retriever],
                    weights=[self.vector_weight, self.bm25_weight]
                )
                logger.info("集成检索器初始化成功（向量+BM25）")
            else:
                # 如果BM25不可用，只使用向量检索
                self.ensemble_retriever = self.vector_retriever
                logger.warning("BM25不可用，仅使用向量检索")
                
        except Exception as e:
            logger.error(f"检索器初始化失败: {e}", exc_info=True)
            # 降级到仅向量检索
            self.ensemble_retriever = self.vector_retriever
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        获取相关文档
        
        Args:
            query: 查询文本
        
        Returns:
            相关文档列表
        """
        try:
            logger.info(f"执行混合检索: query='{query[:50]}...'")
            
            # 使用集成检索器
            documents = self.ensemble_retriever.get_relevant_documents(query)
            
            logger.info(f"混合检索完成: 找到 {len(documents)} 个文档")
            return documents
            
        except Exception as e:
            logger.error(f"混合检索失败: {e}", exc_info=True)
            # 降级到向量检索
            try:
                logger.warning("降级到纯向量检索")
                documents = self.vector_retriever.get_relevant_documents(query)
                return documents
            except Exception as fallback_e:
                logger.error(f"向量检索也失败: {fallback_e}")
                return []
    
    def search_with_score(self, query: str) -> List[tuple]:
        """
        带分数的搜索
        
        Args:
            query: 查询文本
        
        Returns:
            (Document, score) 元组列表
        """
        try:
            # 向量检索带分数
            vector_results = self.vector_store.similarity_search_with_score(
                query, k=self.k
            )
            
            # BM25检索（不带分数，需要手动添加）
            if self.bm25_retriever:
                bm25_docs = self.bm25_retriever.get_relevant_documents(query)
                # 为BM25结果添加默认分数
                bm25_results = [(doc, 0.5) for doc in bm25_docs]
            else:
                bm25_results = []
            
            # 合并结果（简单合并，可以优化）
            all_results = vector_results + bm25_results
            
            # 去重（基于内容）
            seen_contents = set()
            unique_results = []
            for doc, score in all_results:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    unique_results.append((doc, score))
            
            # 按分数排序并限制数量
            unique_results.sort(key=lambda x: x[1], reverse=True)
            return unique_results[:self.k]
            
        except Exception as e:
            logger.error(f"带分数搜索失败: {e}")
            # 降级到向量检索
            return self.vector_store.similarity_search_with_score(query, k=self.k)


def create_hybrid_retriever(
    vector_store,
    documents: List[Document],
    use_hybrid: bool = USE_HYBRID_SEARCH,
    vector_weight: float = VECTOR_SEARCH_WEIGHT,
    bm25_weight: float = BM25_SEARCH_WEIGHT,
    k: int = MAX_SEARCH_RESULTS
) -> HybridRetriever:
    """
    创建混合检索器的工厂函数
    
    Args:
        vector_store: 向量存储实例
        documents: 所有文档列表
        use_hybrid: 是否使用混合检索
        vector_weight: 向量检索权重
        bm25_weight: BM25检索权重
        k: 返回结果数量
    
    Returns:
        HybridRetriever实例
    """
    if not use_hybrid:
        logger.info("混合检索已禁用，仅使用向量检索")
    
    return HybridRetriever(
        vector_store=vector_store,
        documents=documents,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        k=k
    )
