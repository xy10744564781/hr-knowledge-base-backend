"""
回答生成链

基于检索结果和用户上下文生成个性化回答
"""

from typing import Dict, Any, Optional, List

# 尝试导入 LangChain 的 ChatOpenAI
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain.llms import OpenAI as ChatOpenAI
    except ImportError:
        # 如果都失败，使用现有的 llm_agent 作为降级
        from llm_agent import get_hr_agent
        ChatOpenAI = None

from .base_chain import BaseKnowledgeChain
from .models import UserContext, ChainInput, DocumentResult
from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, LLM_MODEL


class AnswerGenerationChain(BaseKnowledgeChain):
    """回答生成链"""
    
    def __init__(self, **kwargs):
        super().__init__(chain_name="answer_generation", **kwargs)
        self.llm = self._initialize_llm()
        self.prompt_builder = PromptBuilder()
    
    def _initialize_llm(self) -> ChatOpenAI:
        """初始化LLM"""
        try:
            return ChatOpenAI(
                model=LLM_MODEL,
                openai_api_key=DASHSCOPE_API_KEY,
                openai_api_base=DASHSCOPE_BASE_URL,
                temperature=0.3,  # 适中的温度，保持一致性但允许一定创造性
                max_tokens=1500
            )
        except Exception as e:
            self.logger.error(f"LLM初始化失败: {str(e)}")
            return None
    
    def _execute_chain(
        self, 
        inputs: Dict[str, Any], 
        run_manager: Optional[Any] = None
    ) -> Dict[str, Any]:
        """执行回答生成"""
        chain_input: ChainInput = inputs["input_data"]
        user_context: UserContext = inputs.get("user_context")
        documents: List[DocumentResult] = inputs.get("documents", [])
        intent_analysis = inputs.get("intent_analysis")  # 获取意图分析结果
        retrieval_strategy = inputs.get("retrieval_strategy")  # 获取检索策略
        
        if not user_context:
            raise ValueError("缺少用户上下文信息")
        
        query = chain_input.query
        
        self.logger.info(
            f"生成回答: 用户={user_context.username}, "
            f"文档数={len(documents)}, 查询='{query[:50]}...'"
        )
        
        try:
            if not self.llm:
                # 降级到模板回答
                return self._generate_template_answer(query, documents, user_context)
            
            # 验证文档相关性
            if documents and intent_analysis:
                relevant_docs = self._filter_relevant_documents(
                    documents, intent_analysis, retrieval_strategy
                )
                
                # 如果过滤后没有相关文档
                if not relevant_docs:
                    self.logger.warning(
                        f"检索到{len(documents)}个文档，但都与查询意图不相关。"
                        f"检测部门: {intent_analysis.detected_department}"
                    )
                    
                    # 判断是否是权限问题
                    if (intent_analysis.detected_department and 
                        retrieval_strategy and 
                        not retrieval_strategy.has_permission):
                        # 用户查询的是无权限部门的内容，且公共文件夹没有相关文档
                        return self._generate_permission_denied_answer(
                            query, intent_analysis.detected_department, user_context
                        )
                    else:
                        # 其他情况：没有找到相关文档
                        return self._generate_no_relevant_docs_answer(query, user_context)
                
                # 使用过滤后的相关文档
                documents = relevant_docs
            
            # 判断回答类型
            if documents:
                # 基于文档的回答
                answer_data = self._generate_document_based_answer(
                    query, documents, user_context
                )
            else:
                # 通用知识回答
                answer_data = self._generate_general_knowledge_answer(
                    query, user_context
                )
            
            self.logger.info(f"回答生成完成: 类型={answer_data['answer_type']}")
            
            return answer_data
            
        except Exception as e:
            self.logger.error(f"回答生成失败: {str(e)}")
            
            # 返回错误回答
            return self._generate_error_answer(str(e), user_context)
    
    def _generate_document_based_answer(
        self, 
        query: str, 
        documents: List[DocumentResult], 
        user_context: UserContext
    ) -> Dict[str, Any]:
        """生成基于文档的回答"""
        # 构建提示词
        prompt = self.prompt_builder.build_document_based_prompt(
            query, documents, user_context
        )
        
        # 调用LLM生成回答
        response = self.llm.invoke(prompt)
        answer = response.content.strip()
        
        # 后处理回答
        processed_answer = self._post_process_answer(answer, documents)
        
        return {
            "answer": processed_answer,
            "answer_type": "document_based",
            "source_documents": [
                {
                    "title": doc.title,
                    "department": doc.department,
                    "score": doc.score,
                    "document_id": doc.document_id
                }
                for doc in documents
            ],
            "confidence": self._calculate_confidence(documents),
            "source_info": "基于公司文档"
        }
    
    def _generate_general_knowledge_answer(
        self, 
        query: str, 
        user_context: UserContext
    ) -> Dict[str, Any]:
        """生成通用知识回答"""
        # 构建通用知识提示词
        prompt = self.prompt_builder.build_general_knowledge_prompt(
            query, user_context
        )
        
        # 调用LLM生成回答
        response = self.llm.invoke(prompt)
        answer = response.content.strip()
        
        # 添加来源标识
        if not answer.startswith("💡"):
            answer = "💡 **信息来源：AI通用知识（非公司文档）**\n\n" + answer
        
        # 添加温馨提示
        if not "温馨提示" in answer:
            answer += "\n\n【温馨提示】\n此回答基于AI通用知识。如需了解公司具体政策，请联系相关部门或查阅公司文档。"
        
        # 添加免责声明
        disclaimer = "⚠️ 免责声明：以上回答由AI基于文档内容分析生成，仅供参考。如有疑问或需要准确信息，请与公司相关部门负责人确认。"
        if disclaimer not in answer:
            answer += f"\n\n{disclaimer}"
        
        return {
            "answer": answer,
            "answer_type": "general_knowledge",
            "source_documents": [],
            "confidence": 0.6,  # 通用知识的默认置信度
            "source_info": "AI通用知识"
        }
    
    def _generate_template_answer(
        self, 
        query: str, 
        documents: List[DocumentResult], 
        user_context: UserContext
    ) -> Dict[str, Any]:
        """生成模板回答（LLM不可用时的降级处理）"""
        if documents:
            # 基于文档的模板回答
            doc_titles = [doc.title for doc in documents[:3]]
            answer = f"""【问题理解】
关于「{query}」的查询，我在公司文档中找到了以下相关信息：

【相关文档】
{chr(10).join(f'• {title}' for title in doc_titles)}

【建议操作】
1. 请查阅上述相关文档获取详细信息
2. 如需具体指导，请联系{user_context.department}部门
3. 如有疑问，可咨询部门管理员

【注意事项】
以上信息来源于公司内部文档，请以最新版本为准。"""
        else:
            # 通用回答模板
            answer = f"""💡 **信息来源：系统提示（AI服务暂时不可用）**

关于「{query}」的查询，系统暂时无法提供详细回答。

【建议操作】
1. 请联系{user_context.department}部门获取相关信息
2. 查阅公司内部文档或政策手册
3. 咨询部门管理员或相关同事

【温馨提示】
系统正在维护中，请稍后再试或通过其他方式获取帮助。"""
        
        # 添加免责声明
        disclaimer = "⚠️ 免责声明：以上回答由AI基于文档内容分析生成，仅供参考。如有疑问或需要准确信息，请与公司相关部门负责人确认。"
        answer += f"\n\n{disclaimer}"
        
        return {
            "answer": answer,
            "answer_type": "template",
            "source_documents": [],
            "confidence": 0.3,
            "source_info": "系统模板"
        }
    
    def _generate_error_answer(self, error_msg: str, user_context: UserContext) -> Dict[str, Any]:
        """生成错误回答"""
        answer = f"""抱歉，系统在处理您的查询时遇到了问题。

【错误信息】
{error_msg}

【建议操作】
1. 请稍后重试
2. 联系{user_context.department}部门获取帮助
3. 如问题持续，请联系技术支持

【联系方式】
如需紧急帮助，请直接联系相关部门或系统管理员。"""
        
        # 添加免责声明
        disclaimer = "⚠️ 免责声明：以上回答由AI基于文档内容分析生成，仅供参考。如有疑问或需要准确信息，请与公司相关部门负责人确认。"
        answer += f"\n\n{disclaimer}"
        
        return {
            "answer": answer,
            "answer_type": "error",
            "source_documents": [],
            "confidence": 0.0,
            "source_info": "系统错误"
        }
    
    def _filter_relevant_documents(
        self,
        documents: List[DocumentResult],
        intent_analysis,
        retrieval_strategy
    ) -> List[DocumentResult]:
        """过滤相关文档
        
        检查文档是否与查询意图匹配：
        1. 如果用户查询特定部门的内容，检查文档是否来自该部门或相关
        2. 使用文档标题和部门信息进行匹配
        """
        if not intent_analysis or not intent_analysis.detected_department:
            # 没有检测到特定部门，返回所有文档
            return documents
        
        detected_dept = intent_analysis.detected_department
        keywords = intent_analysis.keywords
        
        self.logger.info(
            f"过滤文档相关性: 检测部门={detected_dept}, "
            f"关键词={keywords}, 文档数={len(documents)}"
        )
        
        relevant_docs = []
        for doc in documents:
            # 检查1：文档部门是否匹配
            if doc.department == detected_dept:
                relevant_docs.append(doc)
                self.logger.debug(f"✓ 文档相关（部门匹配）: {doc.title} ({doc.department})")
                continue
            
            # 检查2：文档标题是否包含检测到的部门名称
            if detected_dept in doc.title:
                relevant_docs.append(doc)
                self.logger.debug(f"✓ 文档相关（标题包含部门）: {doc.title}")
                continue
            
            # 检查3：文档标题是否包含查询关键词
            if keywords:
                title_lower = doc.title.lower()
                if any(keyword.lower() in title_lower for keyword in keywords):
                    relevant_docs.append(doc)
                    self.logger.debug(f"✓ 文档相关（标题包含关键词）: {doc.title}")
                    continue
            
            # 文档不相关
            self.logger.debug(
                f"✗ 文档不相关: {doc.title} (部门: {doc.department}, "
                f"检测部门: {detected_dept})"
            )
        
        self.logger.info(
            f"文档相关性过滤完成: {len(documents)} -> {len(relevant_docs)}"
        )
        
        return relevant_docs
    
    def _generate_permission_denied_answer(
        self,
        query: str,
        detected_department: str,
        user_context: UserContext
    ) -> Dict[str, Any]:
        """生成权限拒绝回答"""
        answer = f"""【权限提示】

您查询的内容涉及「{detected_department}」部门，但您当前无权访问该部门的文档。

【您的权限】
- 当前部门：{user_context.department}
- 可访问部门：{', '.join(user_context.accessible_folders)}

【建议操作】
1. 如需查看{detected_department}部门的文档，请联系{detected_department}部门负责人申请权限
2. 您可以在公共文件夹中查找相关信息
3. 联系系统管理员了解权限申请流程

【温馨提示】
公司文档按部门进行权限管理，以保护敏感信息。如有业务需要，请通过正规流程申请访问权限。"""
        
        # 添加免责声明
        disclaimer = "⚠️ 免责声明：以上回答由AI基于文档内容分析生成，仅供参考。如有疑问或需要准确信息，请与公司相关部门负责人确认。"
        answer += f"\n\n{disclaimer}"
        
        return {
            "answer": answer,
            "answer_type": "permission_denied",
            "source_documents": [],
            "confidence": 1.0,  # 权限判断是确定的
            "source_info": "权限控制"
        }
    
    def _generate_no_relevant_docs_answer(
        self,
        query: str,
        user_context: UserContext
    ) -> Dict[str, Any]:
        """生成未找到相关文档的回答"""
        answer = f"""【查询结果】

抱歉，未能在您有权访问的文档中找到与「{query}」相关的内容。

【您的权限范围】
- 当前部门：{user_context.department}
- 可访问部门：{', '.join(user_context.accessible_folders)}

【建议操作】
1. 尝试使用不同的关键词重新查询
2. 检查查询内容是否属于您有权访问的部门范围
3. 联系{user_context.department}部门同事或管理员获取帮助
4. 如需访问其他部门的文档，请申请相应权限

【温馨提示】
如果您确定相关文档应该存在，可能是：
- 文档尚未上传到系统
- 文档的关键词与您的查询不匹配
- 文档存储在您无权访问的部门中"""
        
        # 添加免责声明
        disclaimer = "⚠️ 免责声明：以上回答由AI基于文档内容分析生成，仅供参考。如有疑问或需要准确信息，请与公司相关部门负责人确认。"
        answer += f"\n\n{disclaimer}"
        
        return {
            "answer": answer,
            "answer_type": "no_relevant_docs",
            "source_documents": [],
            "confidence": 0.8,  # 未找到文档是确定的
            "source_info": "未找到相关文档"
        }
    
    def _post_process_answer(self, answer: str, documents: List[DocumentResult]) -> str:
        """后处理回答"""
        # 确保回答包含文档依据部分
        if "【文档依据】" not in answer and documents:
            doc_list = "\n".join(f"• {doc.title}" for doc in documents[:3])
            answer = f"【文档依据】\n{doc_list}\n\n{answer}"
        
        # 添加参考文档列表
        if documents and "📌 参考文档" not in answer:
            ref_docs = "\n".join(
                f"• {doc.title} ({doc.department})"
                for doc in documents[:5]
            )
            answer += f"\n\n📌 参考文档\n{ref_docs}"
        
        # 添加免责声明（如果还没有）
        disclaimer = "⚠️ 免责声明：以上回答由AI基于文档内容分析生成，仅供参考。如有疑问或需要准确信息，请与公司相关部门负责人确认。"
        if disclaimer not in answer:
            answer += f"\n\n{disclaimer}"
        
        return answer
    
    def _calculate_confidence(self, documents: List[DocumentResult]) -> float:
        """计算回答置信度"""
        if not documents:
            return 0.0
        
        # 基于文档数量和相关性分数计算置信度
        avg_score = sum(doc.score for doc in documents) / len(documents)
        doc_count_factor = min(len(documents) / 5.0, 1.0)  # 文档数量因子
        
        confidence = (avg_score * 0.7 + doc_count_factor * 0.3)
        return round(confidence, 3)
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """验证输入数据"""
        super()._validate_inputs(inputs)
        
        if "user_context" not in inputs:
            raise ValueError("缺少用户上下文信息")


class PromptBuilder:
    """提示词构建器"""
    
    def build_document_based_prompt(
        self, 
        query: str, 
        documents: List[DocumentResult], 
        user_context: UserContext
    ) -> str:
        """构建基于文档的提示词"""
        # 格式化文档内容
        formatted_docs = self._format_documents(documents)
        
        # 获取角色特定指令
        role_instructions = self._get_role_instructions(user_context.role)
        
        prompt = f"""你是一个专业的企业知识库助手。请基于提供的公司文档回答用户问题。

用户信息：
- 姓名：{user_context.username}
- 部门：{user_context.department}
- 角色：{user_context.role}
- 可访问部门：{', '.join(user_context.accessible_folders)}

角色要求：{role_instructions}

用户问题：{query}

相关文档：
{formatted_docs}

回答要求：
1. 严格基于提供的文档内容回答
2. 在回答开头添加【文档依据】部分，列出引用的文档
3. 提供详细的操作步骤和注意事项
4. 如果文档信息不完整，明确说明缺失的内容
5. 在回答末尾添加"📌 参考文档"列表
6. 在回答最后必须添加免责声明："⚠️ 免责声明：以上回答由AI基于文档内容分析生成，仅供参考。如有疑问或需要准确信息，请与公司相关部门负责人确认。"
7. 保持专业、准确、有帮助的语调

请开始回答："""
        
        return prompt
    
    def build_general_knowledge_prompt(
        self, 
        query: str, 
        user_context: UserContext
    ) -> str:
        """构建通用知识提示词"""
        role_instructions = self._get_role_instructions(user_context.role)
        
        prompt = f"""你是一个友好、专业的AI助手。

用户信息：
- 部门：{user_context.department}
- 角色：{user_context.role}

角色要求：{role_instructions}

用户问题：{query}

当前情况：知识库中没有找到相关的公司文档。

请直接回答用户的问题。如果是简单的常识问题，请简洁明了地回答。
如果是复杂的专业问题，请提供有帮助的建议。

重要提示：
1. 请在回答开头添加"💡 **信息来源：AI通用知识（非公司文档）**"
2. 直接回答问题，保持友好、自然的语气
3. 在回答末尾添加温馨提示，说明这是基于AI通用知识的回答

请开始回答："""
        
        return prompt
    
    def _format_documents(self, documents: List[DocumentResult]) -> str:
        """格式化文档内容"""
        if not documents:
            return "未找到相关文档。"
        
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            formatted_doc = f"""
【文档{i}】{doc.title} ({doc.department})
相关性：{doc.score:.2f}
内容：{doc.content[:500]}{'...' if len(doc.content) > 500 else ''}
"""
            formatted_docs.append(formatted_doc)
        
        return "\n".join(formatted_docs)
    
    def _get_role_instructions(self, user_role) -> str:
        """获取角色特定指令"""
        from .models import UserRole
        
        role_instructions = {
            UserRole.EMPLOYEE: "作为员工，请提供易懂的政策解释和操作指导",
            UserRole.ADMIN: "作为部门管理员，请提供详细的管理指导和操作步骤",
            UserRole.SUPER_ADMIN: "作为超级管理员，请提供全面的系统分析和战略建议"
        }
        
        return role_instructions.get(user_role, "请提供专业的指导")


class AnswerQualityEvaluator:
    """回答质量评估器"""
    
    @staticmethod
    def evaluate_answer(answer: str, documents: List[DocumentResult], query: str) -> Dict[str, Any]:
        """评估回答质量"""
        metrics = {
            "completeness": AnswerQualityEvaluator._evaluate_completeness(answer, query),
            "accuracy": AnswerQualityEvaluator._evaluate_accuracy(answer, documents),
            "clarity": AnswerQualityEvaluator._evaluate_clarity(answer),
            "helpfulness": AnswerQualityEvaluator._evaluate_helpfulness(answer)
        }
        
        # 计算总体质量分数
        overall_score = sum(metrics.values()) / len(metrics)
        
        return {
            "overall_score": overall_score,
            "metrics": metrics,
            "quality_level": AnswerQualityEvaluator._get_quality_level(overall_score)
        }
    
    @staticmethod
    def _evaluate_completeness(answer: str, query: str) -> float:
        """评估回答完整性"""
        # 简单的完整性评估
        if len(answer) < 50:
            return 0.3
        elif len(answer) < 200:
            return 0.6
        else:
            return 0.9
    
    @staticmethod
    def _evaluate_accuracy(answer: str, documents: List[DocumentResult]) -> float:
        """评估回答准确性"""
        if not documents:
            return 0.5  # 通用知识回答的默认准确性
        
        # 检查是否包含文档依据
        if "【文档依据】" in answer:
            return 0.9
        elif "参考文档" in answer:
            return 0.7
        else:
            return 0.5
    
    @staticmethod
    def _evaluate_clarity(answer: str) -> float:
        """评估回答清晰度"""
        # 检查结构化元素
        structure_indicators = ["【", "】", "•", "1.", "2.", "3."]
        structure_score = sum(1 for indicator in structure_indicators if indicator in answer)
        
        return min(structure_score / 5.0, 1.0)
    
    @staticmethod
    def _evaluate_helpfulness(answer: str) -> float:
        """评估回答有用性"""
        helpful_indicators = ["建议", "操作", "步骤", "注意", "联系", "查阅"]
        helpful_score = sum(1 for indicator in helpful_indicators if indicator in answer)
        
        return min(helpful_score / 4.0, 1.0)
    
    @staticmethod
    def _get_quality_level(score: float) -> str:
        """获取质量等级"""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"