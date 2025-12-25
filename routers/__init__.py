"""
路由模块初始化
"""

from . import query_router
from . import upload_router  
from . import admin_router
from . import chat_router

__all__ = ['query_router', 'upload_router', 'admin_router', 'chat_router']