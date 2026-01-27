from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class Department(str, Enum):
    """部门枚举"""
    HR = "人事"
    QUALITY = "质量"
    TECH = "技术"
    FINANCE = "财务"
    SALES = "销售"
    OPERATIONS = "运营"
    PUBLIC = "公共"  # 匿名用户和公共访问

class UserContext(BaseModel):
    """简化的用户上下文，专为人事部门设计"""
    department: Department = Department.HR
    user_role: str = "employee"  # 改为字符串类型，从数据库动态获取角色
    user_id: Optional[str] = None
    
    class Config:
        use_enum_values = True

class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str = Field(..., min_length=1, max_length=500, description="用户查询问题")
    user_ctx: UserContext = Field(default_factory=UserContext, description="用户上下文")
    session_id: Optional[str] = Field(default=None, description="会话ID（用于对话历史）")  # dev-mix新增

class DocumentCategory(str, Enum):
    """文档分类枚举"""
    POLICY = "政策制度"
    PROCESS = "流程指南"
    HANDBOOK = "员工手册"
    TRAINING = "培训资料"
    FORM = "表格模板"
    OTHER = "其他"

class AccessLevel(str, Enum):
    """访问权限枚举"""
    PUBLIC = "全员"
    HR_ONLY = "人事专用"
    MANAGER_ONLY = "管理层"
    CONFIDENTIAL = "机密"

class DocumentUploadRequest(BaseModel):
    """文档上传请求模型"""
    title: str = Field(..., min_length=1, max_length=100, description="文档标题")
    category: DocumentCategory = Field(default=DocumentCategory.OTHER, description="文档分类")
    access_level: AccessLevel = Field(default=AccessLevel.PUBLIC, description="访问权限")
    description: Optional[str] = Field(None, max_length=500, description="文档描述")
    user_ctx: UserContext = Field(default_factory=UserContext, description="用户上下文")

class DocumentInfo(BaseModel):
    """文档信息模型"""
    id: str
    title: str
    category: DocumentCategory
    access_level: AccessLevel
    filename: str
    file_size: Optional[int] = None
    upload_time: datetime
    uploader: str
    chunks_count: int
    department: Optional[str] = None
    description: Optional[str] = None

class QueryResponse(BaseModel):
    """查询响应模型"""
    answer: str = Field(..., description="AI生成的回答")
    source_data: List[Dict] = Field(default_factory=list, description="参考来源数据")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="回答置信度")
    processing_time: Optional[float] = Field(None, description="处理时间（秒）")

class DocumentUploadResponse(BaseModel):
    """文档上传响应模型"""
    status: str = Field(..., description="上传状态")
    document_id: Optional[str] = Field(None, description="文档ID")
    filename: str = Field(..., description="文件名")
    chunks: int = Field(..., description="文档分块数量")
    first_chunk: Optional[str] = Field(None, description="首个文档块预览")
    message: Optional[str] = Field(None, description="状态消息")

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    service: str = Field(..., description="服务名称")
    timestamp: datetime = Field(default_factory=datetime.now, description="检查时间")
    detail: Optional[Dict] = Field(None, description="详细信息")

class VectorStoreStatus(BaseModel):
    """向量存储状态模型"""
    status: str = Field(..., description="向量库状态")
    documents: int = Field(..., description="文档数量")
    collection_name: str = Field(..., description="集合名称")
    last_updated: Optional[datetime] = Field(None, description="最后更新时间")
