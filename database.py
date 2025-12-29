"""
数据库配置和连接管理
"""
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 数据库配置（从环境变量读取）
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./hr_knowledge.db")

# 打印数据库配置信息（脱敏处理）
def _mask_db_url(url: str) -> str:
    """脱敏处理数据库URL，隐藏密码"""
    if not url:
        return "未设置"
    # 隐藏密码部分
    if "://" in url and "@" in url:
        parts = url.split("://")
        if len(parts) == 2:
            protocol = parts[0]
            rest = parts[1]
            if "@" in rest:
                credentials, host_db = rest.split("@", 1)
                if ":" in credentials:
                    username = credentials.split(":")[0]
                    return f"{protocol}://{username}:****@{host_db}"
    return url

print(f"[数据库配置] DATABASE_URL: {_mask_db_url(DATABASE_URL)}")

# 创建数据库引擎
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # 自动检测连接是否有效
    pool_recycle=3600,   # 1小时后回收连接
    echo=False           # 不打印SQL语句（生产环境设为False）
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建基类
Base = declarative_base()


# 聊天会话表
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(50), nullable=True, index=True, comment="用户ID（预留字段）")
    title = Column(String(200), nullable=False, comment="聊天标题")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


# 聊天消息表
class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(50), nullable=False, index=True, comment="会话ID")
    role = Column(String(20), nullable=False, comment="角色：user/bot")
    content = Column(Text, nullable=False, comment="消息内容")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    sequence = Column(Integer, nullable=False, comment="消息顺序")
    
    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "text": self.content,  # 前端使用 text 字段
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "sequence": self.sequence
        }


def init_database():
    """初始化数据库表"""
    try:
        Base.metadata.create_all(bind=engine)
        print("数据库表创建成功")
    except Exception as e:
        print(f"数据库表创建失败: {e}")


def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
