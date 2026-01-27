"""
数据库配置和连接管理
"""
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, Boolean, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import uuid
import os
import hashlib
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
    echo=False           # 关闭SQL日志（调试完成）
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建基类
Base = declarative_base()

def generate_uuid():
    """生成不带连字符的UUID"""
    return str(uuid.uuid4()).replace('-', '')


# ==================== 权限管理表 ====================

# 角色-权限关联表（多对多）
role_permissions = Table(
    'role_permissions',
    Base.metadata,
    Column('role_id', String(32), ForeignKey('roles.id'), primary_key=True),
    Column('permission_id', String(32), ForeignKey('permissions.id'), primary_key=True),
    Column('created_at', DateTime, default=datetime.utcnow)
)


class Permission(Base):
    """权限表"""
    __tablename__ = "permissions"
    
    id = Column(String(32), primary_key=True, default=generate_uuid, index=True)
    code = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    category = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    roles = relationship("Role", secondary=role_permissions, back_populates="permissions")
    
    def to_dict(self):
        return {
            "id": self.id,
            "code": self.code,
            "name": self.name,
            "description": self.description,
            "category": self.category
        }


class Role(Base):
    """角色表"""
    __tablename__ = "roles"
    
    id = Column(String(32), primary_key=True, default=generate_uuid, index=True)
    code = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    is_system = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")
    users = relationship("User", back_populates="role_obj")
    
    def to_dict(self, include_permissions=False):
        result = {
            "id": self.id,
            "code": self.code,
            "name": self.name,
            "description": self.description,
            "is_system": self.is_system
        }
        if include_permissions:
            result["permissions"] = [p.to_dict() for p in self.permissions]
        return result
    
    def has_permission(self, permission_code: str) -> bool:
        return any(p.code == permission_code for p in self.permissions)


# ==================== 原有表结构 ====================

# 部门表
class Department(Base):
    __tablename__ = "departments"
    
    id = Column(String(32), primary_key=True, default=generate_uuid, index=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关系
    users = relationship("User", back_populates="department")


# 用户表
class User(Base):
    __tablename__ = "users"
    
    id = Column(String(32), primary_key=True, default=generate_uuid, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    department_id = Column(String(32), ForeignKey("departments.id"), nullable=True)
    role_id = Column(String(32), ForeignKey("roles.id"), nullable=True)  # 新增：角色ID
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    # 关系
    department = relationship("Department", back_populates="users")
    role_obj = relationship("Role", back_populates="users")  # 新增：角色关系
    chat_sessions = relationship("ChatSession", back_populates="user")
    
    def set_password(self, password):
        """设置密码哈希"""
        self.password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    def check_password(self, password):
        """验证密码"""
        return self.password_hash == hashlib.sha256(password.encode()).hexdigest()
    
    def has_permission(self, permission_code: str) -> bool:
        """检查用户是否拥有指定权限"""
        if not self.role_obj:
            return False
        return self.role_obj.has_permission(permission_code)
    
    def get_accessible_folders(self):
        """获取用户可访问的文件夹"""
        if self.has_permission("document.view_all"):
            # 有查看所有文档权限 - 动态从数据库读取所有部门
            from sqlalchemy.orm import Session
            session = Session.object_session(self)
            if session:
                all_departments = session.query(Department).all()
                dept_names = [dept.name for dept in all_departments]
                # 确保"公共"在列表中，且不重复
                if "公共" not in dept_names:
                    return ["公共"] + dept_names
                else:
                    return dept_names
            else:
                # 如果无法获取 session，返回公共文件夹
                return ["公共"]
        else:
            # 普通用户只能访问自己部门和公共文件夹
            try:
                dept_name = self.department.name if self.department else None
                if dept_name and dept_name != "公共":
                    return ["公共", dept_name]
                else:
                    return ["公共"]
            except Exception:
                return ["公共"]
    
    def can_upload(self):
        """检查用户是否可以上传文档"""
        return self.has_permission("document.upload")
    
    def to_dict(self):
        """转换为字典"""
        try:
            department_name = None
            accessible_folders = ["公共"]
            role_code = "employee"
            role_name = "普通员工"
            
            # 获取角色信息（兼容新旧两种方式）
            if self.role_obj:
                # 新方式：使用 role_obj
                role_code = self.role_obj.code
                role_name = self.role_obj.name
                
                # 根据权限判断
                if self.has_permission("document.view_all"):
                    from sqlalchemy.orm import Session
                    session = Session.object_session(self)
                    if session:
                        all_departments = session.query(Department).all()
                        dept_names = [dept.name for dept in all_departments]
                        # 确保"公共"在列表中，且不重复
                        if "公共" not in dept_names:
                            accessible_folders = ["公共"] + dept_names
                        else:
                            accessible_folders = dept_names
                    else:
                        # 如果无法获取 session，只返回公共文件夹
                        accessible_folders = ["公共"]
                    department_name = "全部部门"
                elif self.department:
                    department_name = self.department.name
                    if department_name != "公共":
                        accessible_folders = ["公共", department_name]
                    else:
                        accessible_folders = ["公共"]
                else:
                    department_name = "未分配"
            else:
                # 旧方式：使用 role 字段（兼容性）
                try:
                    from sqlalchemy import inspect as sql_inspect
                    mapper = sql_inspect(self.__class__)
                    if 'role' in [c.key for c in mapper.columns]:
                        from sqlalchemy.orm import Session
                        session = Session.object_session(self)
                        if session:
                            from sqlalchemy import text
                            result = session.execute(
                                text("SELECT role FROM users WHERE id = :id"),
                                {"id": self.id}
                            ).fetchone()
                            if result:
                                role_code = result[0]
                                role_map = {
                                    "employee": "普通员工",
                                    "admin": "部门管理员",
                                    "super_admin": "超级管理员"
                                }
                                role_name = role_map.get(role_code, "普通员工")
                                
                                if role_code == "super_admin":
                                    # 动态获取所有部门
                                    from sqlalchemy.orm import Session
                                    session = Session.object_session(self)
                                    if session:
                                        all_departments = session.query(Department).all()
                                        dept_names = [dept.name for dept in all_departments]
                                        # 确保"公共"在列表中，且不重复
                                        if "公共" not in dept_names:
                                            accessible_folders = ["公共"] + dept_names
                                        else:
                                            accessible_folders = dept_names
                                    else:
                                        accessible_folders = ["公共"]
                                    department_name = "全部部门"
                                elif self.department:
                                    department_name = self.department.name
                                    if department_name != "公共":
                                        accessible_folders = ["公共", department_name]
                                    else:
                                        accessible_folders = ["公共"]
                except Exception:
                    pass
                
                if not department_name and self.department:
                    department_name = self.department.name
                    accessible_folders = ["公共", department_name]
                elif not department_name:
                    department_name = "未分配"
            
            return {
                "id": self.id,
                "username": self.username,
                "email": self.email,
                "department": department_name,
                "department_id": self.department_id,
                "role": role_code,
                "role_name": role_name,
                "is_active": self.is_active,
                "accessible_folders": accessible_folders,
                "can_upload": self.can_upload(),
                "permissions": [p.code for p in self.role_obj.permissions] if self.role_obj else [],
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "last_login": self.last_login.isoformat() if self.last_login else None
            }
        except Exception:
            return {
                "id": self.id,
                "username": self.username,
                "email": self.email,
                "department": "未分配",
                "department_id": self.department_id,
                "role": "employee",
                "role_name": "普通员工",
                "is_active": self.is_active,
                "accessible_folders": ["公共"],
                "can_upload": False,
                "permissions": [],
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "last_login": self.last_login.isoformat() if self.last_login else None
            }


# 聊天会话表
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(String(32), primary_key=True, default=generate_uuid)
    user_id = Column(String(32), ForeignKey("users.id"), nullable=True, index=True)
    title = Column(String(200), nullable=False, comment="聊天标题")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")
    
    # 关系
    user = relationship("User", back_populates="chat_sessions")
    
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
    
    id = Column(String(32), primary_key=True, default=generate_uuid)
    session_id = Column(String(32), nullable=False, index=True, comment="会话ID")
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


def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session():
    """获取数据库会话（同步版本）"""
    return SessionLocal()


def init_database():
    """初始化数据库表和基础数据"""
    try:
        # 创建所有表
        Base.metadata.create_all(bind=engine)
        print("数据库表创建成功")
        
        # 初始化基础数据
        init_base_data()
        
    except Exception as e:
        print(f"数据库初始化失败: {e}")
        raise


def init_base_data():
    """初始化基础数据"""
    db = SessionLocal()
    try:
        # 不再自动创建部门，由超级管理员通过界面创建
        # 注意："公共"不是一个真实的部门，只是用来标识公共文档的标签
        
        # 检查是否已有超级管理员（使用新的 RBAC 系统）
        if db.query(User).count() == 0:
            # 获取或创建超级管理员角色
            super_admin_role = db.query(Role).filter(Role.code == "super_admin").first()
            
            if super_admin_role:
                # 创建默认超级管理员（不绑定部门）
                admin_user = User(
                    username="admin",
                    email="admin@company.com",
                    department_id=None,  # 超级管理员不绑定特定部门
                    role_id=super_admin_role.id  # 使用新的 role_id
                )
                admin_user.set_password("admin123")  # 默认密码，生产环境需要修改
                db.add(admin_user)
                db.commit()
                print("默认超级管理员创建成功 (admin/admin123)")
                print("提示: 请使用超级管理员账号登录后，在设置页面创建部门")
            else:
                print("警告: 未找到超级管理员角色，请先运行 init_rbac_simple.py")
        
    except Exception as e:
        db.rollback()
        print(f"基础数据初始化失败: {e}")
    finally:
        db.close()
