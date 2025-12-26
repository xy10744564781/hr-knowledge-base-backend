"""
阿里云百炼Embeddings适配器
直接使用OpenAI SDK，避免LangChain兼容性问题
"""
from typing import List
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from logging_setup import logger


class DashScopeEmbeddings(Embeddings):
    """阿里云百炼Embeddings适配器"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v3",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        **kwargs
    ):
        """
        初始化阿里云百炼Embeddings
        
        Args:
            api_key: 阿里云API密钥
            model: 模型名称
            base_url: API基础URL
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        logger.info(f"DashScopeEmbeddings初始化: model={model}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为文档列表生成embeddings
        
        Args:
            texts: 文本列表
            
        Returns:
            embeddings列表
        """
        if not texts:
            return []
        
        try:
            # 确保所有文本都是字符串
            clean_texts = [str(text).strip() for text in texts if text and str(text).strip()]
            
            if not clean_texts:
                logger.warning("没有有效的文本可以生成embedding")
                return []
            
            logger.info(f"生成embeddings: {len(clean_texts)} 个文本")
            
            # 调用OpenAI SDK
            response = self.client.embeddings.create(
                model=self.model,
                input=clean_texts,
                encoding_format="float"
            )
            
            # 提取embeddings
            embeddings = [data.embedding for data in response.data]
            
            logger.info(f"成功生成 {len(embeddings)} 个embeddings，维度: {len(embeddings[0])}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"生成embeddings失败: {e}", exc_info=True)
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        为单个查询文本生成embedding
        
        Args:
            text: 查询文本
            
        Returns:
            embedding向量
        """
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []
