"""
LangChain链式调用的数据模型

定义链之间传递的数据结构
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class UserRole(str, Enum):
    """用户角色枚举"""
    EMPLOYEE = "employee"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


@dataclass
class UserContext:
    """用户上下文信息"""
    user_id: str
    username: str
    department: str
    department_id: str
    role: UserRole
    accessible_folders: List[str]
    can_upload: bool
    permissions: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = {}


@dataclass
class IntentAnalysis:
    """查询意图分析结果"""
    primary_intent: str  # 主要意图 (人事/质量/技术等)
    confidence: float    # 置信度 (0.0-1.0)
    keywords: List[str]  # 关键词
    domain_scores: Dict[str, float]  # 各领域得分
    detected_department: Optional[str] = None  # 检测到的部门


@dataclass
class RetrievalStrategy:
    """检索策略"""
    primary_folders: List[str]    # 主要检索文件夹
    secondary_folders: List[str]  # 备选检索文件夹
    search_filters: Dict[str, Any]  # 搜索过滤条件
    max_results: int = 5         # 最大结果数
    relevance_threshold: float = 0.3  # 相关性阈值
    detected_department: Optional[str] = None  # 检测到的目标部门
    has_permission: bool = True  # 用户是否有权限访问检测到的部门
    
    def get_all_folders(self) -> List[str]:
        """获取所有检索文件夹"""
        return self.primary_folders + self.secondary_folders
    
    def build_chroma_filters(self) -> Dict[str, Any]:
        """构建ChromaDB过滤条件"""
        all_folders = self.get_all_folders()
        if all_folders:
            return {"department": {"$in": all_folders}}
        return {}


@dataclass
class ChainInput:
    """链输入的基础数据结构"""
    query: str
    user_id: str
    session_id: Optional[str] = None
    additional_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_data is None:
            self.additional_data = {}


@dataclass
class ChainOutput:
    """链输出的基础数据结构"""
    success: bool
    data: Dict[str, Any]
    error_message: Optional[str] = None
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}


@dataclass
class DocumentResult:
    """文档检索结果"""
    content: str
    metadata: Dict[str, Any]
    score: float
    document_id: str
    title: str
    department: str
    
    @classmethod
    def from_langchain_document(cls, doc, score: float = 0.0):
        """从LangChain Document对象创建"""
        metadata = getattr(doc, 'metadata', {})
        return cls(
            content=doc.page_content,
            metadata=metadata,
            score=score,
            document_id=metadata.get('document_id', ''),
            title=metadata.get('title', ''),
            department=metadata.get('department', '')
        )


@dataclass
class QueryResult:
    """完整查询结果"""
    answer: str
    user_context: UserContext
    intent_analysis: IntentAnalysis
    retrieval_strategy: RetrievalStrategy
    documents: List[DocumentResult]
    confidence: float
    processing_time: float
    source_type: str  # 'document_based' or 'general_knowledge'