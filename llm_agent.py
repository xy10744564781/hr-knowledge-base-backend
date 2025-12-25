import json
import os
from typing import List, Dict, Optional
from langchain_ollama import ChatOllama
from logging_setup import logger
from config import OLLAMA_MODEL, OLLAMA_BASE_URL

class HRKnowledgeAgent:
    """人事知识库智能代理 - 优化版本"""
    
    def __init__(self):
        self.llm = None
        self.system_prompt = self._load_system_prompt()
        self._initialize_llm()
    
    def _load_system_prompt(self) -> str:
        """加载系统提示词"""
        try:
            prompt_path = os.path.join(os.path.dirname(__file__), 'prompt', 'hr_prompt.txt')
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"加载系统提示词失败: {e}")
            return "你是一个专业的人事知识库助手。"
    
    def _initialize_llm(self):
        """初始化LLM - 针对qwen3:8b优化"""
        try:
            # qwen3:8b模型优化配置
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.1,  # 保持一致性
                top_p=0.9,
                num_ctx=2048,  # 增加上下文长度
                num_predict=2048,  # 增加生成长度
                repeat_penalty=1.1,  # 减少重复
                top_k=30,  # qwen3:8b优化的候选词数量
                num_thread=6,  # 针对qwen3:8b优化的线程数
                # 移除可能导致截断的停止词
            )
            logger.info(f"LLM初始化成功 (完整回答模式): {OLLAMA_MODEL}")
            
        except Exception as e:
            logger.error(f"LLM初始化失败: {e}")
            self.llm = None
    
    def _format_context_documents(self, vector_results: List) -> str:
        """格式化上下文文档"""
        if not vector_results:
            return "未找到相关文档。"
        
        formatted_docs = []
        for i, doc in enumerate(vector_results, 1):
            content = doc.page_content.strip()
            metadata = getattr(doc, 'metadata', {})
            
            # 提取文档信息
            title = metadata.get('title', f'文档{i}')
            category = metadata.get('category', '未分类')
            
            formatted_doc = f"""
【文档{i}】{title} ({category})
内容：{content}
"""
            formatted_docs.append(formatted_doc)
        
        return "\n".join(formatted_docs)
    
    def _build_enhanced_prompt(self, question: str, context_docs: str, user_ctx: Dict) -> str:
        """构建增强的提示词 - 优化版本"""
        user_role = user_ctx.get('user_role', 'hr_staff')
        department = user_ctx.get('department', 'HR')
        
        # 根据用户角色调整回答风格
        role_context = {
            'hr_staff': '作为人事专员，请提供详细的操作指导',
            'hr_manager': '作为人事经理，请提供管理层面的建议',
            'hr_director': '作为人事总监，请提供战略层面的分析',
            'employee': '作为员工，请提供易懂的政策解释'
        }.get(user_role, '请提供专业的人事指导')
        
        # 简化的提示词，避免过长导致截断
        prompt = f"""你是一个专业的人事知识库助手。请基于提供的文档内容回答用户问题。

用户角色：{user_role}
用户问题：{question}

相关文档：
{context_docs}

请按以下格式回答：

【问题理解】
简要确认问题要点

【详细解答】
基于文档内容提供完整详细的回答，包括：
- 具体流程步骤
- 重要注意事项  
- 相关政策要求
- 实操建议

【操作指导】
如果涉及具体操作，请提供分步骤指导

【注意事项】
列出重要的注意点和风险提示

请确保回答完整、准确，基于文档内容，不要编造信息。"""
        
        return prompt
    
    def _generate_fallback_response(self, vector_results: List, question: str) -> str:
        """生成降级响应"""
        if not vector_results:
            return """【问题理解】
您咨询的问题我已收到。

【当前状态】
抱歉，暂时未能在知识库中找到直接相关的政策文档。

【建议操作】
1. 请尝试使用不同的关键词重新查询
2. 联系人事部门获取最新政策信息
3. 查阅公司内部人事管理系统

【联系方式】
如需进一步帮助，请直接联系人事部门。"""
        
        # 提供文档摘要
        summary_parts = ["【问题理解】", f"关于「{question}」的查询，我找到了以下相关信息：", "", "【相关信息】"]
        
        for i, doc in enumerate(vector_results[:3], 1):
            content = doc.page_content[:150].strip()
            metadata = getattr(doc, 'metadata', {})
            title = metadata.get('title', f'相关文档{i}')
            
            summary_parts.append(f"{i}. {title}")
            summary_parts.append(f"   {content}...")
            summary_parts.append("")
        
        summary_parts.extend([
            "【建议操作】",
            "1. 查阅上述相关文档获取详细信息",
            "2. 如需具体指导，请联系人事部门",
            "",
            "【注意事项】",
            "以上信息仅供参考，具体执行请以最新政策为准。"
        ])
        
        return "\n".join(summary_parts)
    
    def generate_response(self, question: str, vector_results: List, user_ctx: Dict) -> str:
        """生成人事知识库回答 - 优化版本"""
        try:
            if not self.llm:
                logger.error("LLM未初始化，使用降级响应")
                return self._generate_fallback_response(vector_results, question)
            
            # 格式化上下文文档
            context_docs = self._format_context_documents(vector_results)
            
            # 构建增强提示词
            prompt = self._build_enhanced_prompt(question, context_docs, user_ctx)
            
            # 生成回答
            response = self.llm.invoke(prompt)
            
            if response and response.content:
                logger.info("LLM回答生成成功")
                return response.content.strip()
            else:
                logger.warning("LLM返回空响应，使用降级处理")
                return self._generate_fallback_response(vector_results, question)
        
        except Exception as e:
            logger.error(f"LLM回答生成失败: {str(e)}")
            return self._generate_fallback_response(vector_results, question)
    
    def analyze_query_intent(self, question: str) -> Dict:
        """分析查询意图"""
        try:
            # 改进的意图分析 - 按优先级和权重排序
            intent_keywords = {
                'attendance': ['考勤', '打卡', '请假', '休假', '迟到', '早退', '出勤'],
                'salary': ['薪资', '工资', '薪酬', '奖金', '发放', '薪水'],
                'onboarding': ['入职', '新员工', '报到', '入职手续'],
                'offboarding': ['离职', '辞职', '退休', '离职手续', '离职流程'],
                'training': ['培训', '学习', '发展', '课程', '培训计划'],
                'benefit': ['福利', '待遇', '补贴', '津贴', '福利待遇'],
                'process': ['流程', '步骤', '程序', '办理', '怎么办', '如何'],
                'policy': ['政策', '制度', '规定', '条例', '政策制度']
            }
            
            detected_intents = []
            intent_scores = {}
            
            for intent, keywords in intent_keywords.items():
                score = 0
                for keyword in keywords:
                    if keyword in question:
                        # 给更具体的关键词更高的权重
                        if len(keyword) > 2:
                            score += 2
                        else:
                            score += 1
                
                if score > 0:
                    detected_intents.append(intent)
                    intent_scores[intent] = score
            
            # 按分数排序，选择最高分的作为主要意图
            if detected_intents:
                primary_intent = max(detected_intents, key=lambda x: intent_scores[x])
            else:
                primary_intent = 'general'
            
            return {
                'intents': detected_intents,
                'primary_intent': primary_intent,
                'confidence': len(detected_intents) / len(intent_keywords)
            }
        
        except Exception as e:
            logger.error(f"查询意图分析失败: {e}")
            return {'intents': ['general'], 'primary_intent': 'general', 'confidence': 0.0}

# 全局代理实例
hr_agent = None

def get_hr_agent() -> HRKnowledgeAgent:
    """获取人事知识库代理实例"""
    global hr_agent
    if hr_agent is None:
        hr_agent = HRKnowledgeAgent()
    return hr_agent

def integrate_results(vector_results: List, sql_results: List, question: str, user_ctx: Dict) -> str:
    """
    整合向量检索结果，生成人事相关的回答
    保持向后兼容性的接口
    """
    agent = get_hr_agent()
    return agent.generate_response(question, vector_results, user_ctx)

def analyze_query(question: str) -> Dict:
    """分析查询意图"""
    agent = get_hr_agent()
    return agent.analyze_query_intent(question)