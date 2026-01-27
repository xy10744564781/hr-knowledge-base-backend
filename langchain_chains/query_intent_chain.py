"""
查询意图分析链

使用AI分析用户查询的意图，确定问题属于哪个业务领域
"""

from typing import Dict, Any, Optional, List
from .base_chain import BaseKnowledgeChain
from .models import IntentAnalysis, ChainInput, UserContext
from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, LLM_MODEL

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


class QueryIntentChain(BaseKnowledgeChain):
    """查询意图分析链"""
    
    def __init__(self, **kwargs):
        super().__init__(chain_name="query_intent", **kwargs)
        self.llm = self._initialize_llm()
        self.departments = self._load_departments_from_db()
    
    def _initialize_llm(self):
        """初始化LLM"""
        try:
            if ChatOpenAI:
                return ChatOpenAI(
                    model=LLM_MODEL,
                    openai_api_key=DASHSCOPE_API_KEY,
                    openai_api_base=DASHSCOPE_BASE_URL,
                    temperature=0.1,  # 低温度确保一致性
                    max_tokens=500
                )
            else:
                # 降级到现有的 HR Agent
                return get_hr_agent()
        except Exception as e:
            self.logger.error(f"LLM初始化失败: {str(e)}")
            # 降级到现有的 HR Agent
            try:
                return get_hr_agent()
            except:
                return None
    
    def _load_departments_from_db(self) -> List[str]:
        """从数据库加载部门列表"""
        try:
            from database_rbac import get_db_session, Department
            db = get_db_session()
            try:
                departments = db.query(Department).all()
                dept_list = [dept.name for dept in departments if dept.name != "公共"]
                self.logger.info(f"从数据库加载了 {len(dept_list)} 个部门")
                return dept_list
            finally:
                db.close()
        except Exception as e:
            self.logger.error(f"从数据库加载部门失败: {str(e)}")
            return []
    
    def _execute_chain(
        self, 
        inputs: Dict[str, Any], 
        run_manager: Optional[Any] = None
    ) -> Dict[str, Any]:
        """执行查询意图分析"""
        chain_input: ChainInput = inputs["input_data"]
        query = chain_input.query
        user_context = inputs.get("user_context")
        
        self.logger.info(f"分析查询意图: {query[:50]}...")
        
        try:
            # 使用AI进行意图分析
            if self.llm and self.departments:
                intent_analysis = self._analyze_by_ai(query, user_context)
            else:
                # 如果没有LLM或部门列表，返回默认结果
                self.logger.warning("LLM或部门列表不可用，使用默认意图分析")
                intent_analysis = IntentAnalysis(
                    primary_intent="general",
                    confidence=0.0,
                    keywords=[],
                    domain_scores={},
                    detected_department=None
                )
            
            self.logger.info(
                f"意图分析完成: 检测部门={intent_analysis.detected_department} "
                f"(置信度: {intent_analysis.confidence:.2f})"
            )
            
            return {
                "intent_analysis": intent_analysis,
                "analysis_method": "ai" if self.llm else "fallback"
            }
            
        except Exception as e:
            self.logger.error(f"意图分析失败: {str(e)}")
            
            # 返回默认分析结果
            fallback_analysis = IntentAnalysis(
                primary_intent="general",
                confidence=0.0,
                keywords=[],
                domain_scores={},
                detected_department=None
            )
            
            return {
                "intent_analysis": fallback_analysis,
                "analysis_method": "fallback",
                "error": str(e)
            }
    
    def _analyze_by_ai(self, query: str, user_context: Optional[UserContext]) -> IntentAnalysis:
        """基于AI的意图分析"""
        # 构建分析提示词
        prompt = self._build_intent_analysis_prompt(query, user_context)
        
        try:
            # 调用LLM进行分析
            response = self.llm.invoke(prompt)
            
            # 解析AI响应
            return self._parse_ai_response(response.content, query)
            
        except Exception as e:
            self.logger.error(f"AI意图分析失败: {str(e)}")
            raise
    
    def _build_intent_analysis_prompt(self, query: str, user_context: Optional[UserContext]) -> str:
        """构建意图分析提示词"""
        departments_str = ', '.join(self.departments) if self.departments else "无"
        
        user_info = ""
        if user_context:
            user_info = f"""
用户信息：
- 部门：{user_context.department}
- 角色：{user_context.role}
- 可访问部门：{', '.join(user_context.accessible_folders)}
"""
        
        prompt = f"""你是一个专业的企业查询意图分析助手。请分析用户查询的意图，判断这个问题最可能属于哪个部门的业务范畴。

{user_info}

公司现有部门：{departments_str}

用户查询：{query}

请按以下JSON格式返回分析结果：
{{
    "detected_department": "最相关的部门名称（如果无法判断或与公司业务无关则为null）",
    "confidence": 0.85,
    "keywords": ["从查询中提取的关键词"],
    "reasoning": "判断理由"
}}

分析要求：
1. confidence 是 0.0-1.0 之间的数值，表示判断的置信度
2. 如果查询明确提到某个部门或该部门的业务，confidence应该较高（>0.7）
3. 如果查询内容模糊或可能涉及多个部门，confidence应该较低（<0.5）
4. **如果查询与公司业务完全无关（如技术问题、常识问题、天气等），detected_department必须为null，confidence为0.0**
5. detected_department 必须是上述部门列表中的一个，或者为null
6. keywords 应该包含查询中的关键业务词汇

示例：
- "如何请婚假？" → detected_department: "人事", confidence: 0.9
- "质量检测流程是什么？" → detected_department: "质量", confidence: 0.9
- "今天天气怎么样？" → detected_department: null, confidence: 0.0
- "什么是Python？" → detected_department: null, confidence: 0.0
- "如何学习编程？" → detected_department: null, confidence: 0.0
"""
        
        return prompt
    
    def _parse_ai_response(self, response: str, original_query: str) -> IntentAnalysis:
        """解析AI响应"""
        try:
            import json
            
            # 尝试提取JSON部分
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                
                detected_dept = result.get("detected_department")
                confidence = float(result.get("confidence", 0.0))
                
                return IntentAnalysis(
                    primary_intent=detected_dept if detected_dept else "general",
                    confidence=confidence,
                    keywords=result.get("keywords", []),
                    domain_scores={detected_dept: confidence} if detected_dept else {},
                    detected_department=detected_dept
                )
            else:
                raise ValueError("无法找到有效的JSON响应")
                
        except Exception as e:
            self.logger.error(f"解析AI响应失败: {str(e)}, 响应内容: {response[:200]}")
            
            # 返回默认结果
            return IntentAnalysis(
                primary_intent="general",
                confidence=0.0,
                keywords=[],
                domain_scores={},
                detected_department=None
            )
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """验证输入数据"""
        super()._validate_inputs(inputs)
        
        chain_input = inputs["input_data"]
        if not isinstance(chain_input, ChainInput):
            raise ValueError("input_data 必须是 ChainInput 类型")
        
        if not chain_input.query or not chain_input.query.strip():
            raise ValueError("查询内容不能为空")