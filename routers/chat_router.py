"""
聊天会话管理 API
"""
from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import Response
from sqlalchemy.orm import Session
from database import get_db
from chat_service import (
    create_chat_session,
    get_chat_sessions,
    get_chat_session,
    add_chat_message,
    delete_chat_session,
    update_chat_session_title,
    get_first_user_message
)
from title_generator import generate_session_title
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

router = APIRouter()


# 请求模型
class CreateSessionRequest(BaseModel):
    title: str
    user_id: Optional[str] = None


class AddMessageRequest(BaseModel):
    role: str  # 'user' or 'bot'
    content: str


class UpdateTitleRequest(BaseModel):
    title: str


# 响应模型
class MessageResponse(BaseModel):
    id: str
    role: str
    text: str
    sequence: int
    created_at: str


class SessionResponse(BaseModel):
    id: str
    title: str
    messages: List[dict]
    created_at: str
    updated_at: str


@router.post("/chat-sessions", response_model=dict)
async def create_session(
    request: CreateSessionRequest,
    db: Session = Depends(get_db)
):
    """创建新的聊天会话"""
    try:
        session = create_chat_session(db, request.title, request.user_id)
        return {
            "status": "success",
            "session": session.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建会话失败: {str(e)}")


@router.get("/chat-sessions", response_model=dict)
async def list_sessions(
    user_id: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """获取聊天会话列表"""
    try:
        sessions = get_chat_sessions(db, user_id, limit)
        return {
            "status": "success",
            "sessions": sessions,
            "count": len(sessions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")


@router.get("/chat-sessions/{session_id}", response_model=dict)
async def get_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """获取单个聊天会话"""
    try:
        session = get_chat_session(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        return {
            "status": "success",
            "session": session
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话失败: {str(e)}")


@router.post("/chat-sessions/{session_id}/messages", response_model=dict)
async def add_message(
    session_id: str,
    request: AddMessageRequest,
    db: Session = Depends(get_db)
):
    """添加聊天消息"""
    try:
        message = add_chat_message(db, session_id, request.role, request.content)
        return {
            "status": "success",
            "message": message.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加消息失败: {str(e)}")


@router.put("/chat-sessions/{session_id}/title", response_model=dict)
async def update_title(
    session_id: str,
    request: UpdateTitleRequest,
    db: Session = Depends(get_db)
):
    """更新聊天会话标题"""
    try:
        success = update_chat_session_title(db, session_id, request.title)
        if not success:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        return {
            "status": "success",
            "message": "标题更新成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新标题失败: {str(e)}")


@router.delete("/chat-sessions/{session_id}", response_model=dict)
async def delete_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """删除聊天会话"""
    try:
        success = delete_chat_session(db, session_id)
        if not success:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        return {
            "status": "success",
            "message": "会话删除成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除会话失败: {str(e)}")


@router.post("/chat-sessions/{session_id}/generate-title", response_model=dict)
async def generate_title(
    session_id: str,
    db: Session = Depends(get_db)
):
    """根据第一条用户消息自动生成会话标题"""
    try:
        # 检查会话是否存在
        session = get_chat_session(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        # 获取第一条用户消息
        first_message = get_first_user_message(db, session_id)
        if not first_message:
            raise HTTPException(status_code=400, detail="会话中没有用户消息")
        
        # 调用 LLM 生成标题
        new_title = generate_session_title(first_message)
        
        # 更新标题
        success = update_chat_session_title(db, session_id, new_title)
        if not success:
            raise HTTPException(status_code=500, detail="更新标题失败")
        
        return {
            "status": "success",
            "title": new_title
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成标题失败: {str(e)}")


@router.get("/chat-sessions/{session_id}/export")
async def export_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """导出聊天会话为 Markdown 格式"""
    from logging_setup import logger
    from urllib.parse import quote
    
    try:
        logger.info(f"开始导出会话: {session_id}")
        
        # 获取会话信息
        session = get_chat_session(db, session_id)
        if not session:
            logger.error(f"会话不存在: {session_id}")
            raise HTTPException(status_code=404, detail="会话不存在")
        
        logger.info(f"会话信息获取成功: {session['title']}, 消息数量: {len(session['messages'])}")
        
        # 生成 Markdown 内容
        markdown_content = f"""# {session['title']}

**导出时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

"""
        
        # 添加消息内容
        for msg in session['messages']:
            role_name = "用户" if msg['role'] == 'user' else "AI 助手"
            markdown_content += f"## {role_name}\n\n{msg['text']}\n\n---\n\n"
        
        logger.info(f"Markdown 内容生成成功，长度: {len(markdown_content)}")
        
        # 生成文件名（URL 编码中文）
        filename = f"chat_{session['title']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        encoded_filename = quote(filename)
        
        logger.info(f"准备返回文件: {filename}")
        
        return Response(
            content=markdown_content.encode('utf-8'),
            media_type='text/markdown; charset=utf-8',
            headers={
                'Content-Disposition': f"attachment; filename*=UTF-8''{encoded_filename}"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导出失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")
