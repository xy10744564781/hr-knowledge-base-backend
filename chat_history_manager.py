"""
对话历史管理模块
参考 General_Doc_QA_System/combine_client.py 的 RunnableWithMessageHistory
"""
from typing import Dict, Optional
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from logging_setup import logger
from config import MAX_HISTORY_MESSAGES


class ChatHistoryManager:
    """对话历史管理器"""
    
    def __init__(self, max_messages: int = MAX_HISTORY_MESSAGES):
        """
        初始化对话历史管理器
        
        Args:
            max_messages: 最多保留的消息数量
        """
        self.max_messages = max_messages
        self.histories: Dict[str, ChatMessageHistory] = {}
        logger.info(f"对话历史管理器初始化完成，最大消息数: {max_messages}")
    
    def get_history(self, session_id: str) -> ChatMessageHistory:
        """
        获取会话历史
        
        Args:
            session_id: 会话ID
        
        Returns:
            ChatMessageHistory实例
        """
        if session_id not in self.histories:
            self.histories[session_id] = ChatMessageHistory()
            logger.info(f"创建新的会话历史: {session_id}")
        
        return self.histories[session_id]
    
    def add_user_message(self, session_id: str, message: str):
        """
        添加用户消息
        
        Args:
            session_id: 会话ID
            message: 消息内容
        """
        history = self.get_history(session_id)
        history.add_user_message(message)
        self._trim_history(session_id)
        logger.info(f"添加用户消息到会话 {session_id}: {message[:50]}...")
    
    def add_ai_message(self, session_id: str, message: str):
        """
        添加AI消息
        
        Args:
            session_id: 会话ID
            message: 消息内容
        """
        history = self.get_history(session_id)
        history.add_ai_message(message)
        self._trim_history(session_id)
        logger.info(f"添加AI消息到会话 {session_id}: {message[:50]}...")
    
    def _trim_history(self, session_id: str):
        """
        修剪历史消息，保持在最大数量限制内
        
        Args:
            session_id: 会话ID
        """
        history = self.get_history(session_id)
        
        if len(history.messages) > self.max_messages:
            # 保留最新的消息
            history.messages = history.messages[-self.max_messages:]
            logger.info(f"会话 {session_id} 历史已修剪到 {self.max_messages} 条消息")
    
    def get_messages(self, session_id: str) -> list:
        """
        获取会话的所有消息
        
        Args:
            session_id: 会话ID
        
        Returns:
            消息列表
        """
        history = self.get_history(session_id)
        return history.messages
    
    def clear_history(self, session_id: str):
        """
        清空会话历史
        
        Args:
            session_id: 会话ID
        """
        if session_id in self.histories:
            self.histories[session_id].clear()
            logger.info(f"清空会话历史: {session_id}")
    
    def delete_session(self, session_id: str):
        """
        删除会话
        
        Args:
            session_id: 会话ID
        """
        if session_id in self.histories:
            del self.histories[session_id]
            logger.info(f"删除会话: {session_id}")
    
    def format_history_for_prompt(self, session_id: str) -> str:
        """
        格式化历史消息用于提示词
        
        Args:
            session_id: 会话ID
        
        Returns:
            格式化的历史文本
        """
        messages = self.get_messages(session_id)
        
        if not messages:
            return ""
        
        formatted_lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_lines.append(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_lines.append(f"助手: {msg.content}")
        
        return "\n".join(formatted_lines)
    
    def get_session_count(self) -> int:
        """获取会话数量"""
        return len(self.histories)
    
    def get_all_session_ids(self) -> list:
        """获取所有会话ID"""
        return list(self.histories.keys())


# 全局历史管理器实例
_global_history_manager: Optional[ChatHistoryManager] = None


def get_history_manager() -> ChatHistoryManager:
    """获取全局历史管理器实例"""
    global _global_history_manager
    if _global_history_manager is None:
        _global_history_manager = ChatHistoryManager()
    return _global_history_manager


def load_history_from_db(session_id: str, db_messages: list) -> ChatMessageHistory:
    """
    从数据库消息加载历史
    
    Args:
        session_id: 会话ID
        db_messages: 数据库中的消息列表
    
    Returns:
        ChatMessageHistory实例
    """
    history = ChatMessageHistory()
    
    for msg in db_messages:
        role = msg.get('role', 'user')
        content = msg.get('text', '')
        
        if role == 'user':
            history.add_user_message(content)
        elif role == 'bot' or role == 'assistant':
            history.add_ai_message(content)
    
    logger.info(f"从数据库加载 {len(db_messages)} 条消息到会话 {session_id}")
    return history


def sync_history_to_manager(session_id: str, db_messages: list):
    """
    将数据库消息同步到历史管理器
    
    Args:
        session_id: 会话ID
        db_messages: 数据库中的消息列表
    """
    manager = get_history_manager()
    history = load_history_from_db(session_id, db_messages)
    manager.histories[session_id] = history
    logger.info(f"会话 {session_id} 历史已同步到管理器")
