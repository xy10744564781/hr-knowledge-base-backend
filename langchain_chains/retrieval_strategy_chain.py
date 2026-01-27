"""
检索策略确定链

根据意图分析和用户权限确定最优的检索策略
"""

from typing import Dict, Any, Optional, List
from .base_chain import BaseKnowledgeChain
from .models import RetrievalStrategy, IntentAnalysis, UserContext, ChainInput


class RetrievalStrategyChain(BaseKnowledgeChain):
    """检索策略确定链"""
    
    def __init__(self, **kwargs):
        super().__init__(chain_name="retrieval_strategy", **kwargs)
    
    def _execute_chain(
        self, 
        inputs: Dict[str, Any], 
        run_manager: Optional[Any] = None
    ) -> Dict[str, Any]:
        """执行检索策略确定"""
        chain_input: ChainInput = inputs["input_data"]
        user_context: UserContext = inputs.get("user_context")
        intent_analysis: IntentAnalysis = inputs.get("intent_analysis")
        
        if not user_context:
            raise ValueError("缺少用户上下文信息")
        if not intent_analysis:
            raise ValueError("缺少意图分析结果")
        
        self.logger.info(
            f"确定检索策略: 用户={user_context.username}, "
            f"意图={intent_analysis.primary_intent}, "
            f"置信度={intent_analysis.confidence:.2f}"
        )
        
        try:
            # 确定检索策略
            strategy = self._determine_strategy(
                user_context, 
                intent_analysis, 
                chain_input.query
            )
            
            self.logger.info(
                f"检索策略确定: 主要文件夹={strategy.primary_folders}, "
                f"备选文件夹={strategy.secondary_folders}"
            )
            
            return {
                "retrieval_strategy": strategy,
                "strategy_reasoning": self._get_strategy_reasoning(
                    user_context, intent_analysis, strategy
                )
            }
            
        except Exception as e:
            self.logger.error(f"检索策略确定失败: {str(e)}")
            
            # 返回默认策略
            fallback_strategy = self._create_fallback_strategy(user_context)
            
            return {
                "retrieval_strategy": fallback_strategy,
                "strategy_reasoning": "使用默认策略（发生错误）",
                "error": str(e)
            }
    
    def _determine_strategy(
        self, 
        user_context: UserContext, 
        intent_analysis: IntentAnalysis,
        query: str
    ) -> RetrievalStrategy:
        """确定检索策略
        
        核心原则：
        1. 用户查询无权限部门时，优先在公共文件夹搜索
        2. 如果公共文件夹有结果，正常返回
        3. 只有公共文件夹也没有结果时，才提示权限问题
        """
        
        # 获取用户可访问的文件夹
        accessible_folders = user_context.accessible_folders
        detected_department = intent_analysis.detected_department
        confidence = intent_analysis.confidence
        
        # 策略1: 高置信度的特定部门意图
        if detected_department and confidence > 0.7:
            if detected_department in accessible_folders:
                # 用户有权限访问检测到的部门
                primary_folders = [detected_department]
                secondary_folders = [f for f in accessible_folders if f != detected_department]
            else:
                # 用户无权限访问检测到的部门
                # 只在公共文件夹中搜索，因为：
                # 1. 如果公共文件夹有相关文档，可以正常回答
                # 2. 如果公共文件夹没有相关文档，后续会提示权限问题
                # 3. 不应该在用户自己的部门（如技术）搜索人事部门的问题
                primary_folders = []
                if "公共" in accessible_folders:
                    primary_folders.append("公共")
                secondary_folders = []
        
        # 策略2: 中等置信度的部门意图
        elif detected_department and confidence > 0.4:
            if detected_department in accessible_folders:
                # 同时搜索检测到的部门和公共文件夹
                primary_folders = [detected_department]
                if "公共" in accessible_folders and "公共" not in primary_folders:
                    primary_folders.append("公共")
                secondary_folders = [f for f in accessible_folders if f not in primary_folders]
            else:
                # 用户无权限，只搜索公共文件夹
                primary_folders = []
                if "公共" in accessible_folders:
                    primary_folders.append("公共")
                secondary_folders = []
        
        # 策略3: 低置信度或通用查询
        else:
            # 优先搜索用户自己的部门
            if user_context.department in accessible_folders and user_context.department != "公共":
                primary_folders = [user_context.department]
                secondary_folders = [f for f in accessible_folders if f != user_context.department]
            else:
                # 搜索所有可访问的文件夹
                primary_folders = accessible_folders[:2] if len(accessible_folders) > 1 else accessible_folders
                secondary_folders = accessible_folders[2:] if len(accessible_folders) > 2 else []
        
        # 根据查询复杂度调整参数
        max_results = self._calculate_max_results(query, confidence)
        relevance_threshold = self._calculate_relevance_threshold(confidence)
        
        # 构建搜索过滤条件
        search_filters = self._build_search_filters(
            primary_folders + secondary_folders,
            user_context,
            intent_analysis
        )
        
        return RetrievalStrategy(
            primary_folders=primary_folders,
            secondary_folders=secondary_folders,
            search_filters=search_filters,
            max_results=max_results,
            relevance_threshold=relevance_threshold,
            detected_department=detected_department,  # 保存检测到的部门，用于后续判断
            has_permission=detected_department in accessible_folders if detected_department else True
        )
    
    def _calculate_max_results(self, query: str, confidence: float) -> int:
        """根据查询复杂度计算最大结果数"""
        base_results = 5
        
        # 查询长度影响
        query_length = len(query.split())
        if query_length > 10:
            base_results += 2  # 复杂查询需要更多结果
        elif query_length < 5:
            base_results -= 1  # 简单查询减少结果
        
        # 置信度影响
        if confidence > 0.8:
            base_results += 1  # 高置信度可以多检索一些
        elif confidence < 0.3:
            base_results += 2  # 低置信度需要更多候选
        
        return max(3, min(base_results, 10))  # 限制在3-10之间
    
    def _calculate_relevance_threshold(self, confidence: float) -> float:
        """根据意图置信度计算相关性阈值
        
        降低阈值以确保能找到更多相关文档
        """
        if confidence > 0.8:
            return 0.25  # 高置信度，适中阈值
        elif confidence > 0.5:
            return 0.2   # 中等置信度，较低阈值
        else:
            return 0.15  # 低置信度，很低阈值
    
    def _build_search_filters(
        self, 
        target_folders: List[str],
        user_context: UserContext,
        intent_analysis: IntentAnalysis
    ) -> Dict[str, Any]:
        """构建搜索过滤条件"""
        filters = {}
        
        # 部门过滤（必须的）
        if target_folders:
            filters["department"] = {"$in": target_folders}
        
        # 注意：不添加 access_level 和 category 过滤
        # 因为很多文档可能没有这些字段，会导致检索失败
        # 权限控制已经通过 department 过滤实现
        
        return filters
    
    def _get_allowed_access_levels(self, user_role) -> List[str]:
        """根据用户角色获取允许的访问级别"""
        from .models import UserRole
        
        if user_role == UserRole.SUPER_ADMIN:
            return ["public", "internal", "confidential", "restricted"]
        elif user_role == UserRole.ADMIN:
            return ["public", "internal", "confidential"]
        else:
            return ["public", "internal"]
    
    def _get_relevant_doc_types(self, intent_analysis: IntentAnalysis) -> List[str]:
        """根据意图分析获取相关的文档类型"""
        intent_to_doc_types = {
            "人事": ["政策制度", "流程指南", "员工手册", "表格模板"],
            "质量": ["政策制度", "流程指南", "标准规范"],
            "技术": ["技术文档", "操作手册", "流程指南"],
            "财务": ["政策制度", "流程指南", "表格模板"],
            "销售": ["流程指南", "培训资料", "表格模板"],
            "运营": ["流程指南", "政策制度", "培训资料"]
        }
        
        primary_intent = intent_analysis.primary_intent
        if primary_intent in intent_to_doc_types:
            return intent_to_doc_types[primary_intent]
        
        return []  # 不限制文档类型
    
    def _create_fallback_strategy(self, user_context: UserContext) -> RetrievalStrategy:
        """创建默认的检索策略"""
        accessible_folders = user_context.accessible_folders
        
        return RetrievalStrategy(
            primary_folders=accessible_folders if "公共" in accessible_folders else [],
            secondary_folders=accessible_folders[1:] if len(accessible_folders) > 1 else [],
            search_filters={"department": {"$in": accessible_folders}} if accessible_folders else {},
            max_results=5,
            relevance_threshold=0.3
        )
    
    def _get_strategy_reasoning(
        self, 
        user_context: UserContext, 
        intent_analysis: IntentAnalysis,
        strategy: RetrievalStrategy
    ) -> str:
        """获取策略推理说明"""
        reasoning_parts = []
        
        # 用户权限说明
        reasoning_parts.append(f"用户 {user_context.username} 可访问文件夹: {', '.join(user_context.accessible_folders)}")
        
        # 意图分析说明
        if intent_analysis.detected_department:
            reasoning_parts.append(
                f"检测到查询意图: {intent_analysis.detected_department} "
                f"(置信度: {intent_analysis.confidence:.2f})"
            )
        else:
            reasoning_parts.append("未检测到明确的部门意图，使用通用策略")
        
        # 策略说明
        if strategy.primary_folders:
            reasoning_parts.append(f"主要检索范围: {', '.join(strategy.primary_folders)}")
        if strategy.secondary_folders:
            reasoning_parts.append(f"备选检索范围: {', '.join(strategy.secondary_folders)}")
        
        reasoning_parts.append(f"最大结果数: {strategy.max_results}, 相关性阈值: {strategy.relevance_threshold}")
        
        return "; ".join(reasoning_parts)
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """验证输入数据"""
        super()._validate_inputs(inputs)
        
        if "user_context" not in inputs:
            raise ValueError("缺少用户上下文信息")
        if "intent_analysis" not in inputs:
            raise ValueError("缺少意图分析结果")


class StrategyOptimizer:
    """检索策略优化器"""
    
    @staticmethod
    def optimize_for_performance(strategy: RetrievalStrategy) -> RetrievalStrategy:
        """为性能优化策略"""
        # 限制最大结果数以提高性能
        if strategy.max_results > 8:
            strategy.max_results = 8
        
        # 如果有太多文件夹，限制搜索范围
        all_folders = strategy.get_all_folders()
        if len(all_folders) > 4:
            # 只保留前4个最重要的文件夹
            strategy.primary_folders = strategy.primary_folders[:2]
            strategy.secondary_folders = strategy.secondary_folders[:2]
        
        return strategy
    
    @staticmethod
    def optimize_for_accuracy(strategy: RetrievalStrategy) -> RetrievalStrategy:
        """为准确性优化策略"""
        # 提高相关性阈值
        strategy.relevance_threshold = max(strategy.relevance_threshold, 0.4)
        
        # 增加结果数以获得更多候选
        strategy.max_results = min(strategy.max_results + 2, 10)
        
        return strategy