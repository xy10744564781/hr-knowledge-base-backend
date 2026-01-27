"""
聊天会话管理服务
"""
from sqlalchemy.orm import Session
from database import ChatSession, ChatMessage, get_db
from typing import List, Optional, Dict
from datetime import datetime


def create_chat_session(db: Session, title: str, user_id: Optional[str] = None) -> ChatSession:
    """创建新的聊天会话"""
    session = ChatSession(
        title=title,
        user_id=user_id  # 允许None值用于匿名用户
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_chat_sessions(db: Session, user_id: Optional[str] = None, limit: int = 50) -> List[Dict]:
    """获取聊天会话列表"""
    query = db.query(ChatSession)
    
    # 根据用户ID过滤会话
    if user_id:
        query = query.filter(ChatSession.user_id == user_id)
    else:
        # 匿名用户只能看到没有关联用户的会话
        query = query.filter(ChatSession.user_id.is_(None))
    
    sessions = query.order_by(ChatSession.updated_at.desc()).limit(limit).all()
    
    # 返回会话及其消息
    result = []
    for session in sessions:
        messages = get_chat_messages(db, session.id)
        result.append({
            "id": session.id,
            "user_id": session.user_id,  # 添加user_id字段
            "title": session.title,
            "messages": messages,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "updated_at": session.updated_at.isoformat() if session.updated_at else None
        })
    
    return result


def get_chat_session(db: Session, session_id: str) -> Optional[Dict]:
    """获取单个聊天会话及其消息"""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    
    if not session:
        return None
    
    messages = get_chat_messages(db, session_id)
    
    return {
        "id": session.id,
        "user_id": session.user_id,  # 添加user_id字段
        "title": session.title,
        "messages": messages,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None
    }


def add_chat_message(db: Session, session_id: str, role: str, content: str) -> ChatMessage:
    """添加聊天消息"""
    # 获取当前会话的消息数量，用于设置顺序
    message_count = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).count()
    
    message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        sequence=message_count
    )
    db.add(message)
    
    # 更新会话的更新时间
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session:
        session.updated_at = datetime.now()
    
    db.commit()
    db.refresh(message)
    return message


def get_chat_messages(db: Session, session_id: str) -> List[Dict]:
    """获取聊天会话的所有消息"""
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.sequence).all()
    
    return [msg.to_dict() for msg in messages]


def delete_chat_session(db: Session, session_id: str) -> bool:
    """删除聊天会话及其所有消息"""
    # 删除消息
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    
    # 删除会话
    result = db.query(ChatSession).filter(ChatSession.id == session_id).delete()
    
    db.commit()
    return result > 0


def update_chat_session_title(db: Session, session_id: str, title: str) -> bool:
    """更新聊天会话标题"""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    
    if not session:
        return False
    
    session.title = title
    session.updated_at = datetime.now()
    db.commit()
    return True


def get_first_user_message(db: Session, session_id: str) -> Optional[str]:
    """获取会话的第一条用户消息"""
    message = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id,
        ChatMessage.role == 'user'
    ).order_by(ChatMessage.sequence).first()
    
    return message.content if message else None
