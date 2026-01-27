"""
LangChain链式调用模块

本模块实现智能多部门知识库检索系统的LangChain链式调用架构
"""

from .base_chain import BaseKnowledgeChain, ChainManager
from .user_context_chain import UserContextChain
from .query_intent_chain import QueryIntentChain
from .retrieval_strategy_chain import RetrievalStrategyChain
from .document_retrieval_chain import DocumentRetrievalChain
from .answer_generation_chain import AnswerGenerationChain
from .models import (
    UserContext, IntentAnalysis, RetrievalStrategy, 
    ChainInput, ChainOutput
)

__all__ = [
    'BaseKnowledgeChain',
    'ChainManager', 
    'UserContextChain',
    'QueryIntentChain',
    'RetrievalStrategyChain',
    'DocumentRetrievalChain',
    'AnswerGenerationChain',
    'UserContext',
    'IntentAnalysis', 
    'RetrievalStrategy',
    'ChainInput',
    'ChainOutput'
]