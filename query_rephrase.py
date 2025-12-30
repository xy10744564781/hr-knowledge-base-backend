"""
查询重述模块 - 使用LLM优化查询
参考 General_Doc_QA_System/knowledge.py 的 RePhraseQueryRetriever
"""
from typing import Optional
from langchain_openai import ChatOpenAI
from logging_setup import logger
from config import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, LLM_MODEL,
    USE_QUERY_REPHRASE
)


class QueryRephraser:
    """查询重述器 - 使用LLM优化查询文本"""
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        初始化查询重述器
        
        Args:
            llm: LLM实例，如果为None则创建新实例
        """
        if llm is None:
            self.llm = ChatOpenAI(
                model=LLM_MODEL,
                openai_api_key=DASHSCOPE_API_KEY,
                openai_api_base=DASHSCOPE_BASE_URL,
                temperature=0.1,  # 低温度，保持稳定性
                max_tokens=200  # 查询重述不需要太长
            )
        else:
            self.llm = llm
        
        logger.info("查询重述器初始化完成")
    
    def rephrase(self, query: str) -> str:
        """
        重述查询
        
        Args:
            query: 原始查询
        
        Returns:
            重述后的查询
        """
        try:
            logger.info(f"开始重述查询: '{query}'")
            
            # 构建重述提示词
            prompt = self._build_rephrase_prompt(query)
            
            # 调用LLM
            response = self.llm.invoke(prompt)
            rephrased_query = response.content.strip()
            
            # 验证重述结果
            if not rephrased_query or len(rephrased_query) < 2:
                logger.warning("重述结果无效，使用原始查询")
                return query
            
            logger.info(f"查询重述完成: '{rephrased_query}'")
            return rephrased_query
            
        except Exception as e:
            logger.error(f"查询重述失败: {e}")
            # 失败时返回原始查询
            return query
    
    def _build_rephrase_prompt(self, query: str) -> str:
        """构建重述提示词"""
        return f"""你是一个专业的查询优化助手。请将用户的查询重新表述为更适合检索的形式。

要求：
1. 提取查询的核心关键词
2. 扩展同义词和相关术语
3. 使用更专业的人事领域术语
4. 保持查询简洁，不要添加无关内容
5. 只输出重述后的查询文本，不要解释

原始查询：{query}

重述后的查询："""


def rephrase_query(query: str, llm: Optional[ChatOpenAI] = None) -> str:
    """
    重述查询的便捷函数
    
    Args:
        query: 原始查询
        llm: LLM实例（可选）
    
    Returns:
        重述后的查询
    """
    if not USE_QUERY_REPHRASE:
        logger.info("查询重述已禁用")
        return query
    
    rephraser = QueryRephraser(llm)
    return rephraser.rephrase(query)


def rephrase_query_batch(queries: list, llm: Optional[ChatOpenAI] = None) -> list:
    """
    批量重述查询
    
    Args:
        queries: 查询列表
        llm: LLM实例（可选）
    
    Returns:
        重述后的查询列表
    """
    if not USE_QUERY_REPHRASE:
        return queries
    
    rephraser = QueryRephraser(llm)
    return [rephraser.rephrase(q) for q in queries]
