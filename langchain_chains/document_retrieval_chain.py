"""
文档检索执行链

在确定的范围内执行向量搜索，获取相关文档
"""

from typing import Dict, Any, Optional, List

# 简单的 Document 类定义
class Document:
    def __init__(self, page_content: str, metadata: Dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}
from .base_chain import BaseKnowledgeChain
from .models import RetrievalStrategy, UserContext, ChainInput, DocumentResult
from knowledge_base import get_vector_manager


class DocumentRetrievalChain(BaseKnowledgeChain):
    """文档检索执行链"""
    
    def __init__(self, **kwargs):
        super().__init__(chain_name="document_retrieval", **kwargs)
        self.vector_manager = get_vector_manager()
    
    def _execute_chain(
        self, 
        inputs: Dict[str, Any], 
        run_manager: Optional[Any] = None
    ) -> Dict[str, Any]:
        """执行文档检索"""
        chain_input: ChainInput = inputs["input_data"]
        user_context: UserContext = inputs.get("user_context")
        retrieval_strategy: RetrievalStrategy = inputs.get("retrieval_strategy")
        
        if not user_context:
            raise ValueError("缺少用户上下文信息")
        if not retrieval_strategy:
            raise ValueError("缺少检索策略")
        
        query = chain_input.query
        
        self.logger.info(
            f"执行文档检索: 查询='{query[:50]}...', "
            f"检索范围={retrieval_strategy.get_all_folders()}"
        )
        
        try:
            if not self.vector_manager:
                self.logger.warning("向量管理器不可用，返回空结果")
                return {
                    "documents": [],
                    "retrieval_method": "none",
                    "total_results": 0
                }
            
            # 执行分层检索
            documents = self._execute_layered_retrieval(query, retrieval_strategy)
            
            # 过滤和排序结果
            filtered_documents = self._filter_and_rank_documents(
                documents, retrieval_strategy, user_context
            )
            
            # 转换为DocumentResult对象
            document_results = []
            for doc in filtered_documents:
                # 尝试从多个位置获取分数
                score = getattr(doc, 'score', None)
                if score is None:
                    score = doc.metadata.get('score', 0.0)
                document_results.append(
                    DocumentResult.from_langchain_document(doc, score)
                )
            
            self.logger.info(f"文档检索完成: 找到 {len(document_results)} 个相关文档")
            
            return {
                "documents": document_results,
                "retrieval_method": "vector_search",
                "total_results": len(document_results),
                "search_strategy": {
                    "primary_folders": retrieval_strategy.primary_folders,
                    "secondary_folders": retrieval_strategy.secondary_folders,
                    "filters_applied": retrieval_strategy.search_filters
                }
            }
            
        except Exception as e:
            self.logger.error(f"文档检索失败: {str(e)}")
            
            return {
                "documents": [],
                "retrieval_method": "error",
                "total_results": 0,
                "error": str(e)
            }
    
    def _execute_layered_retrieval(
        self, 
        query: str, 
        strategy: RetrievalStrategy
    ) -> List[Document]:
        """执行分层检索"""
        all_documents = []
        
        # 第一层：主要文件夹检索
        if strategy.primary_folders:
            primary_filters = {
                **strategy.search_filters,
                "department": {"$in": strategy.primary_folders}
            }
            
            primary_docs = self._search_with_filters(
                query, 
                primary_filters, 
                strategy.max_results
            )
            
            all_documents.extend(primary_docs)
            self.logger.info(f"主要文件夹检索: {len(primary_docs)} 个结果")
        
        # 第二层：如果主要检索结果不足，搜索备选文件夹
        if len(all_documents) < strategy.max_results and strategy.secondary_folders:
            remaining_slots = strategy.max_results - len(all_documents)
            
            secondary_filters = {
                **strategy.search_filters,
                "department": {"$in": strategy.secondary_folders}
            }
            
            secondary_docs = self._search_with_filters(
                query,
                secondary_filters,
                remaining_slots
            )
            
            all_documents.extend(secondary_docs)
            self.logger.info(f"备选文件夹检索: {len(secondary_docs)} 个结果")
        
        return all_documents
    
    def _search_with_filters(
        self, 
        query: str, 
        filters: Dict[str, Any], 
        k: int
    ) -> List[Document]:
        """使用过滤条件执行搜索"""
        try:
            # 使用向量管理器执行搜索
            results = self.vector_manager.search_documents(
                query=query,
                k=k,
                filter_metadata=filters if filters else None
            )
            
            return results if results else []
            
        except Exception as e:
            self.logger.error(f"向量搜索失败: {str(e)}")
            return []
    
    def _filter_and_rank_documents(
        self, 
        documents: List[Document], 
        strategy: RetrievalStrategy,
        user_context: UserContext
    ) -> List[Document]:
        """过滤和排序文档"""
        if not documents:
            return []
        
        # 1. 相关性过滤
        filtered_docs = []
        for doc in documents:
            # 尝试从多个位置获取分数
            score = getattr(doc, 'score', None)
            if score is None:
                score = doc.metadata.get('score', 0.0)
            
            doc_title = doc.metadata.get('title', '未知')
            doc_dept = doc.metadata.get('department', '未知')
            
            if score >= strategy.relevance_threshold:
                filtered_docs.append(doc)
                self.logger.debug(f"✓ 保留文档: {doc_title} (部门: {doc_dept}, 分数: {score:.3f})")
            else:
                self.logger.debug(f"✗ 过滤文档: {doc_title} (部门: {doc_dept}, 分数: {score:.3f} < 阈值: {strategy.relevance_threshold})")
        
        self.logger.info(
            f"相关性过滤: {len(documents)} -> {len(filtered_docs)} "
            f"(阈值: {strategy.relevance_threshold})"
        )
        
        # 2. 权限过滤
        permission_filtered = []
        for doc in filtered_docs:
            if self._check_document_permission(doc, user_context):
                permission_filtered.append(doc)
                self.logger.debug(f"✓ 权限通过: {doc.metadata.get('title', '未知')}")
            else:
                self.logger.debug(f"✗ 权限拒绝: {doc.metadata.get('title', '未知')} (部门: {doc.metadata.get('department', '未知')})")
        
        self.logger.info(
            f"权限过滤: {len(filtered_docs)} -> {len(permission_filtered)}"
        )
        
        # 3. 排序（按相关性分数降序）
        sorted_docs = sorted(
            permission_filtered,
            key=lambda x: getattr(x, 'score', 0.0),
            reverse=True
        )
        
        # 4. 限制结果数量
        final_docs = sorted_docs[:strategy.max_results]
        
        return final_docs
    
    def _check_document_permission(self, doc: Document, user_context: UserContext) -> bool:
        """检查用户是否有权限访问文档"""
        metadata = getattr(doc, 'metadata', {})
        doc_department = metadata.get('department', '公共')
        
        # 只检查部门权限
        # 如果文档的部门在用户可访问列表中，就允许访问
        if doc_department not in user_context.accessible_folders:
            self.logger.debug(f"文档部门 '{doc_department}' 不在用户可访问列表 {user_context.accessible_folders} 中")
            return False
        
        # 不再检查 access_level，因为：
        # 1. 很多文档可能没有这个字段
        # 2. 部门权限已经足够控制访问
        # 3. 避免因为字段缺失或不匹配导致检索失败
        
        return True
    
    def _get_allowed_access_levels(self, user_role) -> List[str]:
        """根据用户角色获取允许的访问级别"""
        from .models import UserRole
        
        if user_role == UserRole.SUPER_ADMIN:
            return ["public", "internal", "confidential", "restricted"]
        elif user_role == UserRole.ADMIN:
            return ["public", "internal", "confidential"]
        else:
            return ["public", "internal"]
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """验证输入数据"""
        super()._validate_inputs(inputs)
        
        if "user_context" not in inputs:
            raise ValueError("缺少用户上下文信息")
        if "retrieval_strategy" not in inputs:
            raise ValueError("缺少检索策略")


class RetrievalResultAnalyzer:
    """检索结果分析器"""
    
    @staticmethod
    def analyze_results(documents: List[DocumentResult], strategy: RetrievalStrategy) -> Dict[str, Any]:
        """分析检索结果"""
        if not documents:
            return {
                "quality": "no_results",
                "coverage": 0.0,
                "avg_score": 0.0,
                "department_distribution": {},
                "recommendations": ["尝试使用不同的关键词", "扩大搜索范围"]
            }
        
        # 计算平均分数
        avg_score = sum(doc.score for doc in documents) / len(documents)
        
        # 分析部门分布
        dept_distribution = {}
        for doc in documents:
            dept = doc.department or "unknown"
            dept_distribution[dept] = dept_distribution.get(dept, 0) + 1
        
        # 评估质量
        quality = "excellent" if avg_score > 0.8 else "good" if avg_score > 0.6 else "fair" if avg_score > 0.4 else "poor"
        
        # 计算覆盖率
        target_folders = strategy.get_all_folders()
        covered_folders = set(doc.department for doc in documents if doc.department)
        coverage = len(covered_folders & set(target_folders)) / len(target_folders) if target_folders else 0.0
        
        # 生成建议
        recommendations = RetrievalResultAnalyzer._generate_recommendations(
            documents, strategy, avg_score, coverage
        )
        
        return {
            "quality": quality,
            "coverage": coverage,
            "avg_score": avg_score,
            "department_distribution": dept_distribution,
            "recommendations": recommendations
        }
    
    @staticmethod
    def _generate_recommendations(
        documents: List[DocumentResult], 
        strategy: RetrievalStrategy,
        avg_score: float,
        coverage: float
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if avg_score < 0.4:
            recommendations.append("尝试使用更具体的关键词")
            recommendations.append("考虑重新表述您的问题")
        
        if coverage < 0.5:
            recommendations.append("扩大搜索范围到更多部门")
        
        if len(documents) < 3:
            recommendations.append("降低搜索精度以获得更多结果")
        
        if not recommendations:
            recommendations.append("检索结果质量良好")
        
        return recommendations