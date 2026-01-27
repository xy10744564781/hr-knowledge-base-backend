import json
import os
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from logging_setup import logger
from config import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, LLM_MODEL,
    LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_TOKENS
)

class HRKnowledgeAgent:
    """人事知识库智能代理 - 基于阿里云百炼API"""
    
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
        """初始化LLM - 使用阿里云百炼API"""
        try:
            self.llm = ChatOpenAI(
                model=LLM_MODEL,
                openai_api_key=DASHSCOPE_API_KEY,
                openai_api_base=DASHSCOPE_BASE_URL,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                max_tokens=LLM_MAX_TOKENS
            )
            logger.info(f"LLM初始化成功（阿里云百炼）: {LLM_MODEL}")
            
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
        """构建增强的提示词 - 使用hr_prompt.txt作为基础，不在开头添加信息来源"""
        user_role = user_ctx.get('user_role', 'hr_staff')
        department = user_ctx.get('department', 'HR')
        
        # 根据用户角色调整回答风格
        role_context = {
            'hr_staff': '作为人事专员，请提供详细的操作指导',
            'hr_manager': '作为人事经理，请提供管理层面的建议',
            'hr_director': '作为人事总监，请提供战略层面的分析',
            'employee': '作为员工，请提供易懂的政策解释'
        }.get(user_role, '请提供专业的人事指导')
        
        # 使用加载的系统提示词作为基础
        prompt = f"""{self.system_prompt}

---

## 当前任务

用户角色：{user_role}
角色要求：{role_context}

用户问题：{question}

相关文档：
{context_docs}

## 回答要求

请严格按照上述系统提示词中的"回答格式模板"进行回答，确保：
1. 【文档依据】部分列出引用的文档
2. 基于文档内容详细回答
3. 提供操作步骤和注意事项
4. 在回答末尾添加"📌 参考文档"列表
5. **在回答最后必须添加免责声明："⚠️ 免责声明：以上回答由AI基于文档内容分析生成，仅供参考。如有疑问或需要准确信息，请与公司相关部门负责人确认。"**

**重要：不要在回答开头添加"信息来源"标识，因为【文档依据】部分已经说明了来源。**

请开始回答："""
        
        return prompt
    
    def _build_enhanced_prompt_with_history(
        self, 
        question: str, 
        context_docs: str, 
        user_ctx: Dict,
        chat_history: list
    ) -> str:
        """构建包含对话历史的增强提示词（dev-mix新增）"""
        user_role = user_ctx.get('user_role', 'hr_staff')
        department = user_ctx.get('department', 'HR')
        
        # 根据用户角色调整回答风格
        role_context = {
            'hr_staff': '作为人事专员，请提供详细的操作指导',
            'hr_manager': '作为人事经理，请提供管理层面的建议',
            'hr_director': '作为人事总监，请提供战略层面的分析',
            'employee': '作为员工，请提供易懂的政策解释'
        }.get(user_role, '请提供专业的人事指导')
        
        # 格式化对话历史
        history_text = ""
        if chat_history:
            history_lines = []
            for msg in chat_history[-4:]:  # 最近2轮对话
                if hasattr(msg, 'type'):
                    role = "用户" if msg.type == "human" else "AI助手"
                    content = msg.content[:200]  # 限制长度
                    history_lines.append(f"{role}: {content}")
            history_text = "\n".join(history_lines)
        
        # 使用加载的系统提示词作为基础
        prompt = f"""{self.system_prompt}

---

## 对话历史

{history_text if history_text else "（这是新对话的开始）"}

---

## 当前任务

用户角色：{user_role}
角色要求：{role_context}

用户问题：{question}

相关文档：
{context_docs}

## 回答要求

请严格按照上述系统提示词中的"回答格式模板"进行回答，确保：
1. 结合对话历史理解用户意图
2. 【文档依据】部分列出引用的文档
3. 基于文档内容详细回答
4. 提供操作步骤和注意事项
5. 在回答末尾添加"📌 参考文档"列表
6. **在回答最后必须添加免责声明："⚠️ 免责声明：以上回答由AI基于文档内容分析生成，仅供参考。如有疑问或需要准确信息，请与公司相关部门负责人确认。"**

**重要：不要在回答开头添加"信息来源"标识，因为【文档依据】部分已经说明了来源。**

请开始回答："""
        
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
        """生成人事知识库回答 - 基于文档或通用知识，明确标注来源"""
        try:
            if not self.llm:
                logger.error("LLM未初始化，使用降级响应")
                return self._generate_fallback_response(vector_results, question)
            
            # 判断是否有相关文档
            if vector_results:
                # 基于文档的回答
                logger.info("生成基于文档的回答")
                context_docs = self._format_context_documents(vector_results)
                prompt = self._build_enhanced_prompt(question, context_docs, user_ctx)
                
                # 生成回答
                response = self.llm.invoke(prompt)
                
                if response and response.content:
                    answer = response.content.strip()
                    
                    # 不再添加开头的信息来源标识
                    # 因为【文档依据】部分已经说明了来源
                    
                    logger.info("LLM回答生成成功（基于文档）")
                    return answer
                else:
                    logger.warning("LLM返回空响应，使用降级处理")
                    return self._generate_fallback_response(vector_results, question)
            else:
                # 通用知识回答 - 保留开头的信息来源标识
                logger.info("生成通用知识回答（不使用文档）")
                
                # 使用系统提示词的简化版本
                prompt = f"""你是一个友好、专业的AI助手。

用户问题：{question}

当前情况：知识库中没有找到相关的公司文档。

请直接回答用户的问题。如果是简单的常识问题（如数学计算、基础知识等），请简洁明了地回答。
如果是复杂的专业问题，请提供有帮助的建议。

**重要提示：**
1. 请在回答开头添加"💡 **信息来源：AI通用知识（非公司文档）**"
2. 直接回答问题，不要拒绝或说"不在职责范围内"
3. 保持友好、自然的语气

回答格式：
💡 **信息来源：AI通用知识（非公司文档）**

[直接回答用户的问题]

【温馨提示】
此回答基于AI通用知识。如需了解公司具体政策，请联系人事部门或查阅公司文档。"""
                
                response = self.llm.invoke(prompt)
                
                if response and response.content:
                    answer = response.content.strip()
                    
                    # 清理可能重复的来源标识
                    lines = answer.split('\n')
                    cleaned_lines = []
                    source_indicator_count = 0
                    
                    for line in lines:
                        # 检查是否是来源标识行
                        if ('信息来源' in line or '💡' in line) and ('AI通用知识' in line or '非公司文档' in line):
                            source_indicator_count += 1
                            # 只保留第一个来源标识
                            if source_indicator_count == 1:
                                cleaned_lines.append(line)
                        else:
                            cleaned_lines.append(line)
                    
                    answer = '\n'.join(cleaned_lines)
                    
                    # 确保回答包含来源标识（如果LLM完全没有添加）
                    if "信息来源" not in answer and "💡" not in answer:
                        answer = "💡 **信息来源：AI通用知识（非公司文档）**\n\n" + answer
                    
                    logger.info("LLM回答生成成功（通用知识）")
                    return answer
                else:
                    logger.warning("LLM返回空响应")
                    return "💡 **信息来源：AI通用知识（非公司文档）**\n\n抱歉，我暂时无法回答这个问题。请尝试换个方式提问，或联系人事部门获取帮助。"
        
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