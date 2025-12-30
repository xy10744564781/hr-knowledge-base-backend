"""
重排序模块 - 使用阿里云Rerank模型
参考 General_Doc_QA_System/models.py 的 get_ali_rerank
"""
from typing import List
from langchain.schema import Document
from langchain_community.document_compressors import DashScopeRerank
from logging_setup import logger
from config import (
    DASHSCOPE_API_KEY, RERANK_MODEL, USE_RERANK, RERANK_TOP_N
)


class DocumentReranker:
    """文档重排序器 - 使用阿里云Rerank模型"""
    
    def __init__(
        self,
        model: str = RERANK_MODEL,
        top_n: int = RERANK_TOP_N,
        api_key: str = DASHSCOPE_API_KEY
    ):
        """
        初始化重排序器
        
        Args:
            model: Rerank模型名称
            top_n: 返回的文档数量
            api_key: API密钥
        """
        self.model = model
        self.top_n = top_n
        self.api_key = api_key
        
        try:
            # 初始化阿里云Rerank
            self.reranker = DashScopeRerank(
                model=model,
                dashscope_api_key=api_key,
                top_n=top_n
            )
            logger.info(f"Rerank模型初始化成功: {model}, top_n={top_n}")
        except Exception as e:
            logger.error(f"Rerank模型初始化失败: {e}")
            self.reranker = None
    
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        重排序文档
        
        Args:
            query: 查询文本
            documents: 待排序的文档列表
        
        Returns:
            重排序后的文档列表
        """
        if not self.reranker:
            logger.warning("Rerank模型未初始化，跳过重排序")
            return documents[:self.top_n]
        
        if not documents:
            return []
        
        try:
            logger.info(f"开始重排序: query='{query[:50]}...', 文档数={len(documents)}")
            
            # 使用Rerank模型重排序
            reranked_docs = self.reranker.compress_documents(
                documents=documents,
                query=query
            )
            
            logger.info(f"重排序完成: 返回 {len(reranked_docs)} 个文档")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            # 失败时返回原始文档（限制数量）
            return documents[:self.top_n]
    
    def rerank_with_scores(
        self, 
        query: str, 
        documents: List[Document]
    ) -> List[tuple]:
        """
        重排序并返回分数
        
        Args:
            query: 查询文本
            documents: 待排序的文档列表
        
        Returns:
            (Document, score) 元组列表
        """
        reranked_docs = self.rerank(query, documents)
        
        # 为重排序后的文档添加分数（基于排名）
        results = []
        for i, doc in enumerate(reranked_docs):
            # 分数从1.0递减到0.0
            score = 1.0 - (i / len(reranked_docs)) if reranked_docs else 0.0
            
            # 将分数添加到metadata
            if not hasattr(doc, 'metadata'):
                doc.metadata = {}
            doc.metadata['rerank_score'] = score
            doc.metadata['rerank_position'] = i
            
            results.append((doc, score))
        
        return results


def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int = RERANK_TOP_N,
    use_rerank: bool = USE_RERANK
) -> List[Document]:
    """
    重排序文档的便捷函数
    
    Args:
        query: 查询文本
        documents: 待排序的文档列表
        top_n: 返回的文档数量
        use_rerank: 是否使用重排序
    
    Returns:
        重排序后的文档列表
    """
    if not use_rerank:
        logger.info("重排序已禁用")
        return documents[:top_n]
    
    if not documents:
        return []
    
    reranker = DocumentReranker(top_n=top_n)
    return reranker.rerank(query, documents)


def create_reranker(
    model: str = RERANK_MODEL,
    top_n: int = RERANK_TOP_N
) -> DocumentReranker:
    """
    创建重排序器的工厂函数
    
    Args:
        model: Rerank模型名称
        top_n: 返回的文档数量
    
    Returns:
        DocumentReranker实例
    """
    return DocumentReranker(model=model, top_n=top_n)
