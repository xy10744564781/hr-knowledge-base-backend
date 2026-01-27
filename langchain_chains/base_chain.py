import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type

# 使用正确的 LangChain 0.2.x 导入路径
try:
    from langchain.chains.base import Chain as BaseChain
    from langchain.callbacks.manager import CallbackManagerForChainRun
except ImportError:
    try:
        # 尝试其他可能的导入路径
        from langchain_core.runnables import Runnable as BaseChain
        CallbackManagerForChainRun = None
    except ImportError:
        # 如果都失败，创建一个简单的基类
        BaseChain = object
        CallbackManagerForChainRun = None
        
from logging_setup import logger
from .models import ChainInput, ChainOutput, UserContext


class BaseKnowledgeChain(ABC):
    """知识库链的基础抽象类 - 兼容 LangChain 0.2.x"""
    
    def __init__(self, chain_name: str, **kwargs):
        self.chain_name = chain_name
        self.logger = logger
        # 如果 BaseChain 可用且不是 object，尝试初始化
        if BaseChain != object and hasattr(BaseChain, '__init__'):
            try:
                super().__init__(**kwargs)
            except:
                pass  # 如果初始化失败，继续使用我们的实现
    
    @property
    def input_keys(self) -> List[str]:
        """定义输入键"""
        return ["input_data"]
    
    @property
    def output_keys(self) -> List[str]:
        """定义输出键"""
        return ["output_data"]
    
    def __call__(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """调用链的主要入口"""
        return self._call(inputs, **kwargs)
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """执行链的主要逻辑"""
        start_time = time.time()
        
        try:
            # 验证输入
            self._validate_inputs(inputs)
            
            # 执行具体的链逻辑
            result = self._execute_chain(inputs, run_manager)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 构建输出
            output = ChainOutput(
                success=True,
                data=result,
                processing_time=processing_time
            )
            
            self.logger.info(f"{self.chain_name} 执行成功，耗时: {processing_time:.3f}s")
            
            return {"output_data": output}
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"{self.chain_name} 执行失败: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            output = ChainOutput(
                success=False,
                data={},
                error_message=error_msg,
                processing_time=processing_time
            )
            
            return {"output_data": output}
    
    @abstractmethod
    def _execute_chain(
        self, 
        inputs: Dict[str, Any], 
        run_manager: Optional[Any] = None
    ) -> Dict[str, Any]:
        """子类需要实现的具体执行逻辑"""
        pass
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """验证输入数据"""
        if "input_data" not in inputs:
            raise ValueError(f"{self.chain_name}: 缺少必需的输入数据 'input_data'")
    
    @property
    def _chain_type(self) -> str:
        """返回链类型"""
        return f"knowledge_{self.chain_name.lower()}"


class ChainManager:
    """链管理器 - 协调多个链的执行"""
    
    def __init__(self):
        self.chains: Dict[str, BaseKnowledgeChain] = {}
        self.execution_order: List[str] = []
        self.logger = logger
    
    def register_chain(self, chain: BaseKnowledgeChain) -> None:
        """注册一个链"""
        self.chains[chain.chain_name] = chain
        self.logger.info(f"注册链: {chain.chain_name}")
    
    def set_execution_order(self, chain_names: List[str]) -> None:
        """设置链的执行顺序"""
        # 验证所有链都已注册
        for name in chain_names:
            if name not in self.chains:
                raise ValueError(f"链 '{name}' 未注册")
        
        self.execution_order = chain_names
        self.logger.info(f"设置执行顺序: {' -> '.join(chain_names)}")
    
    def execute_pipeline(self, initial_input: ChainInput) -> Dict[str, Any]:
        """执行完整的链管道"""
        if not self.execution_order:
            raise ValueError("未设置链执行顺序")
        
        start_time = time.time()
        pipeline_data = {"input_data": initial_input}
        execution_results = {}
        
        self.logger.info(f"开始执行链管道: {' -> '.join(self.execution_order)}")
        
        try:
            for chain_name in self.execution_order:
                chain = self.chains[chain_name]
                
                # 执行当前链
                result = chain(pipeline_data)
                chain_output = result["output_data"]
                
                # 记录执行结果
                execution_results[chain_name] = chain_output
                
                # 检查执行是否成功
                if not chain_output.success:
                    error_msg = f"链 '{chain_name}' 执行失败: {chain_output.error_message}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # 准备下一个链的输入
                pipeline_data = self._prepare_next_input(
                    pipeline_data, 
                    chain_output, 
                    chain_name
                )
            
            total_time = time.time() - start_time
            self.logger.info(f"链管道执行完成，总耗时: {total_time:.3f}s")
            
            return {
                "success": True,
                "results": execution_results,
                "final_data": pipeline_data,
                "total_processing_time": total_time
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"链管道执行失败: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return {
                "success": False,
                "error": error_msg,
                "results": execution_results,
                "total_processing_time": total_time
            }
    
    def _prepare_next_input(
        self, 
        current_data: Dict[str, Any], 
        chain_output: ChainOutput, 
        current_chain_name: str
    ) -> Dict[str, Any]:
        """准备下一个链的输入数据"""
        # 保留原始输入
        next_input = current_data.copy()
        
        # 添加当前链的输出数据
        next_input["input_data"].additional_data[current_chain_name] = chain_output.data
        
        # 根据链的类型，将特定数据提升到顶层
        if current_chain_name == "user_context":
            next_input["user_context"] = chain_output.data.get("user_context")
        elif current_chain_name == "query_intent":
            next_input["intent_analysis"] = chain_output.data.get("intent_analysis")
        elif current_chain_name == "retrieval_strategy":
            next_input["retrieval_strategy"] = chain_output.data.get("retrieval_strategy")
        elif current_chain_name == "document_retrieval":
            next_input["documents"] = chain_output.data.get("documents")
        
        return next_input
    
    def get_chain(self, chain_name: str) -> Optional[BaseKnowledgeChain]:
        """获取指定的链"""
        return self.chains.get(chain_name)
    
    def list_chains(self) -> List[str]:
        """列出所有已注册的链"""
        return list(self.chains.keys())
    
    def clear_chains(self) -> None:
        """清空所有链"""
        self.chains.clear()
        self.execution_order.clear()
        self.logger.info("已清空所有链")


class ChainErrorHandler:
    """链错误处理器"""
    
    @staticmethod
    def handle_chain_error(chain_name: str, error: Exception, inputs: Dict) -> ChainOutput:
        """处理链执行错误"""
        error_msg = f"链 '{chain_name}' 执行失败: {str(error)}"
        logger.error(error_msg, exc_info=True)
        
        return ChainOutput(
            success=False,
            data={},
            error_message=error_msg
        )
    
    @staticmethod
    def create_fallback_response(chain_name: str, error_type: str) -> Dict[str, Any]:
        """创建降级响应"""
        fallback_responses = {
            "user_context": {
                "user_context": UserContext(
                    user_id="anonymous",
                    username="匿名用户",
                    department="公共",
                    department_id="",
                    role="employee",
                    accessible_folders=["公共"] if "公共" in accessible_folders else [],
                    can_upload=False
                )
            },
            "query_intent": {
                "intent_analysis": {
                    "primary_intent": "general",
                    "confidence": 0.0,
                    "keywords": [],
                    "domain_scores": {}
                }
            },
            "retrieval_strategy": {
                "retrieval_strategy": {
                    "primary_folders": ["public"],
                    "secondary_folders": [],
                    "search_filters": {},
                    "max_results": 5,
                    "relevance_threshold": 0.3
                }
            }
        }
        
        return fallback_responses.get(chain_name, {})