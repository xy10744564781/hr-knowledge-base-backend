"""
智能多部门知识库检索路由器

整合所有LangChain链，提供完整的查询处理流程
"""

import time
from typing import Dict, Any, Optional, List
from .base_chain import ChainManager
from .user_context_chain import UserContextChain
from .query_intent_chain import QueryIntentChain
from .retrieval_strategy_chain import RetrievalStrategyChain
from .document_retrieval_chain import DocumentRetrievalChain
from .answer_generation_chain import AnswerGenerationChain
from .models import ChainInput, QueryResult, UserContext, IntentAnalysis, RetrievalStrategy, DocumentResult
from logging_setup import logger


class IntelligentMultiDepartmentRouter:
    """智能多部门知识库检索路由器"""
    
    def __init__(self):
        self.chain_manager = ChainManager()
        self.logger = logger
        self._initialize_chains()
    
    def _initialize_chains(self):
        """初始化所有链"""
        try:
            # 注册所有链
            self.chain_manager.register_chain(UserContextChain())
            self.chain_manager.register_chain(QueryIntentChain())
            self.chain_manager.register_chain(RetrievalStrategyChain())
            self.chain_manager.register_chain(DocumentRetrievalChain())
            # 注意：不再注册 AnswerGenerationChain，因为流式查询会直接生成答案
            # self.chain_manager.register_chain(AnswerGenerationChain())
            
            # 设置执行顺序 - 只执行到文档检索，不生成答案
            self.chain_manager.set_execution_order([
                "user_context",
                "query_intent", 
                "retrieval_strategy",
                "document_retrieval"
                # 移除 "answer_generation"，让流式查询服务直接生成答案
            ])
            
            self.logger.info("智能路由器初始化成功（流式优化模式：跳过答案生成）")
            
        except Exception as e:
            self.logger.error(f"智能路由器初始化失败: {str(e)}")
            raise
    
    def route_query(
        self, 
        query: str, 
        user_id: str, 
        session_id: Optional[str] = None
    ) -> QueryResult:
        """执行完整的查询路由流程"""
        start_time = time.time()
        
        self.logger.info(f"开始智能查询路由: user_id={user_id}, query='{query[:50]}...'")
        
        try:
            # 构建初始输入
            chain_input = ChainInput(
                query=query,
                user_id=user_id,
                session_id=session_id
            )
            
            # 执行链管道
            pipeline_result = self.chain_manager.execute_pipeline(chain_input)
            
            if not pipeline_result["success"]:
                raise RuntimeError(f"链管道执行失败: {pipeline_result['error']}")
            
            # 提取结果
            results = pipeline_result["results"]
            processing_time = time.time() - start_time
            
            # 构建查询结果
            query_result = self._build_query_result(results, processing_time)
            
            self.logger.info(
                f"智能查询路由完成: 耗时={processing_time:.3f}s, "
                f"策略={query_result.source_type}, "
                f"文档数={len(query_result.documents)}"
            )
            
            return query_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"智能查询路由失败: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # 返回错误结果
            return self._create_error_result(error_msg, processing_time)
    
    
    def _build_query_result(self, results: Dict[str, Any], processing_time: float) -> QueryResult:
        """构建查询结果"""
        # 提取各链的结果
        user_context_result = results["user_context"]
        intent_result = results["query_intent"]
        strategy_result = results["retrieval_strategy"]
        retrieval_result = results["document_retrieval"]
        # 注意：不再有 answer_generation 结果
        
        # 提取数据
        user_context = user_context_result.data.get("user_context")
        intent_analysis = intent_result.data.get("intent_analysis")
        retrieval_strategy = strategy_result.data.get("retrieval_strategy")
        documents = retrieval_result.data.get("documents", [])
        
        # 确定来源类型（基于是否有文档）
        if documents:
            source_type = "document_based"
            confidence = self._calculate_document_confidence(documents)
        else:
            source_type = "general_knowledge"
            confidence = 0.6  # 通用知识的默认置信度
        
        return QueryResult(
            answer="",  # 不再生成答案，由流式查询服务生成
            user_context=user_context,
            intent_analysis=intent_analysis,
            retrieval_strategy=retrieval_strategy,
            documents=documents,
            confidence=confidence,
            processing_time=processing_time,
            source_type=source_type
        )
    
    def _calculate_document_confidence(self, documents: List) -> float:
        """计算文档置信度"""
        if not documents:
            return 0.0
        
        # 基于文档数量和相关性分数计算置信度
        avg_score = sum(doc.score for doc in documents) / len(documents)
        doc_count_factor = min(len(documents) / 5.0, 1.0)
        
        confidence = (avg_score * 0.7 + doc_count_factor * 0.3)
        return round(confidence, 3)
    
    def _create_error_result(self, error_msg: str, processing_time: float) -> QueryResult:
        """创建错误结果"""
        # 创建默认的用户上下文
        default_user_context = UserContext(
            user_id="error",
            username="系统错误",
            department="system",
            department_id="",
            role="employee",
            accessible_folders=["public"],
            can_upload=False
        )
        
        # 创建默认的意图分析
        default_intent = IntentAnalysis(
            primary_intent="error",
            confidence=0.0,
            keywords=[],
            domain_scores={}
        )
        
        # 创建默认的检索策略
        default_strategy = RetrievalStrategy(
            primary_folders=["public"],
            secondary_folders=[],
            search_filters={}
        )
        
        return QueryResult(
            answer=f"抱歉，系统处理您的查询时遇到了问题：{error_msg}\n\n请稍后重试或联系技术支持。",
            user_context=default_user_context,
            intent_analysis=default_intent,
            retrieval_strategy=default_strategy,
            documents=[],
            confidence=0.0,
            processing_time=processing_time,
            source_type="error"
        )
    
    def get_chain_status(self) -> Dict[str, Any]:
        """获取链状态信息"""
        return {
            "registered_chains": self.chain_manager.list_chains(),
            "execution_order": self.chain_manager.execution_order,
            "total_chains": len(self.chain_manager.chains)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试简单查询
            test_result = self.route_query(
                query="测试查询",
                user_id="health_check"
            )
            
            return {
                "status": "healthy",
                "chains_count": len(self.chain_manager.chains),
                "test_processing_time": test_result.processing_time,
                "last_check": time.time()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": time.time()
            }


# 全局路由器实例
_global_router = None

def get_intelligent_router() -> IntelligentMultiDepartmentRouter:
    """获取全局智能路由器实例"""
    global _global_router
    if _global_router is None:
        _global_router = IntelligentMultiDepartmentRouter()
    return _global_router


def create_intelligent_router() -> IntelligentMultiDepartmentRouter:
    """创建新的智能路由器实例"""
    return IntelligentMultiDepartmentRouter()


class RouterPerformanceMonitor:
    """路由器性能监控器"""
    
    def __init__(self):
        self.query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_processing_time": 0.0,
            "total_processing_time": 0.0
        }
        self.chain_stats = {}
    
    def record_query(self, success: bool, processing_time: float, chain_results: Dict[str, Any]):
        """记录查询统计"""
        self.query_stats["total_queries"] += 1
        self.query_stats["total_processing_time"] += processing_time
        
        if success:
            self.query_stats["successful_queries"] += 1
        else:
            self.query_stats["failed_queries"] += 1
        
        # 更新平均处理时间
        self.query_stats["avg_processing_time"] = (
            self.query_stats["total_processing_time"] / self.query_stats["total_queries"]
        )
        
        # 记录各链的性能
        for chain_name, result in chain_results.items():
            if chain_name not in self.chain_stats:
                self.chain_stats[chain_name] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "total_time": 0.0,
                    "avg_time": 0.0
                }
            
            stats = self.chain_stats[chain_name]
            stats["total_calls"] += 1
            stats["total_time"] += result.processing_time
            
            if result.success:
                stats["successful_calls"] += 1
            
            stats["avg_time"] = stats["total_time"] / stats["total_calls"]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        success_rate = (
            self.query_stats["successful_queries"] / self.query_stats["total_queries"]
            if self.query_stats["total_queries"] > 0 else 0.0
        )
        
        return {
            "query_stats": {
                **self.query_stats,
                "success_rate": success_rate
            },
            "chain_stats": self.chain_stats,
            "recommendations": self._generate_performance_recommendations()
        }
    
    def _generate_performance_recommendations(self) -> List[str]:
        """生成性能优化建议"""
        recommendations = []
        
        # 检查平均处理时间
        avg_time = self.query_stats["avg_processing_time"]
        if avg_time > 5.0:
            recommendations.append("平均处理时间较长，建议优化链执行效率")
        
        # 检查成功率
        success_rate = (
            self.query_stats["successful_queries"] / self.query_stats["total_queries"]
            if self.query_stats["total_queries"] > 0 else 1.0
        )
        if success_rate < 0.9:
            recommendations.append("查询成功率较低，建议检查错误处理机制")
        
        # 检查各链性能
        for chain_name, stats in self.chain_stats.items():
            if stats["avg_time"] > 2.0:
                recommendations.append(f"{chain_name} 链处理时间较长，建议优化")
        
        if not recommendations:
            recommendations.append("系统性能良好")
        
        return recommendations