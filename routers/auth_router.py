from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
import jwt as pyjwt  # 使用PyJWT库
import os
from typing import Optional

from database import get_db, User, Department, Role
from logging_setup import logger

router = APIRouter(prefix="/api/auth", tags=["认证"])

# JWT配置
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production-hr-knowledge-base-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 24 * 60  # 24小时

security = HTTPBearer()

# Pydantic模型
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    department_id: str  # 注册时必须指定部门

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict

class UserProfile(BaseModel):
    id: str
    username: str
    email: str
    department: str
    department_id: str
    role: str
    accessible_folders: list
    can_upload: bool

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """创建JWT访问令牌"""
    try:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        
        # 确保数据类型正确
        if "sub" in to_encode:
            to_encode["sub"] = str(to_encode["sub"])  # 转换为字符串
        
        encoded_jwt = pyjwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        
        # PyJWT 2.x 返回字符串，1.x 返回bytes
        if isinstance(encoded_jwt, bytes):
            encoded_jwt = encoded_jwt.decode('utf-8')
            
        return encoded_jwt
        
    except Exception as e:
        logger.error(f"JWT编码失败: {e}, 数据: {data}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"令牌生成失败: {str(e)}"
        )

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """验证JWT令牌"""
    try:
        payload = pyjwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的认证凭据",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user_id
    except pyjwt.PyJWTError as e:
        logger.error(f"JWT验证失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"JWT验证异常: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(user_id: str = Depends(verify_token), db: Session = Depends(get_db)):
    """获取当前用户"""
    from sqlalchemy.orm import joinedload
    
    user = db.query(User).options(
        joinedload(User.department),
        joinedload(User.role_obj).joinedload(Role.permissions)  # 预加载角色和权限信息
    ).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在"
        )
    return user

def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)), db: Session = Depends(get_db)):
    """获取当前用户(可选认证)"""
    logger.info("=" * 60)
    logger.info("[DEBUG] get_current_user_optional 被调用")
    logger.info(f"[DEBUG] credentials: {credentials}")
    
    if credentials is None:
        logger.info("[DEBUG] ❌ credentials is None - 没有提供认证信息")
        logger.info("=" * 60)
        return None
    
    logger.info(f"[DEBUG] ✅ credentials 存在")
    logger.info(f"[DEBUG] credentials.scheme: {credentials.scheme}")
    logger.info(f"[DEBUG] credentials.credentials (token前10字符): {credentials.credentials[:10]}...")
    
    try:
        payload = pyjwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        logger.info(f"[DEBUG] ✅ JWT 解码成功")
        logger.info(f"[DEBUG] payload: {payload}")
        
        user_id: str = payload.get("sub")
        logger.info(f"[DEBUG] user_id from token: {user_id}")
        
        if user_id is None:
            logger.warning("[DEBUG] ❌ user_id is None in payload")
            logger.info("=" * 60)
            return None
        
        from sqlalchemy.orm import joinedload
        user = db.query(User).options(
            joinedload(User.department),
            joinedload(User.role_obj).joinedload(Role.permissions)  # 预加载角色和权限信息
        ).filter(User.id == user_id).first()
        
        if user:
            logger.info(f"[DEBUG] ✅ 找到用户: {user.username} (ID: {user.id})")
        else:
            logger.warning(f"[DEBUG] ❌ 未找到用户 ID: {user_id}")
        
        logger.info("=" * 60)
        return user
    except pyjwt.ExpiredSignatureError as e:
        logger.warning(f"[DEBUG] ❌ JWT 已过期: {e}")
        logger.info("=" * 60)
        return None
    except pyjwt.InvalidTokenError as e:
        logger.warning(f"[DEBUG] ❌ JWT 无效: {e}")
        logger.info("=" * 60)
        return None
    except Exception as e:
        logger.error(f"[DEBUG] ❌ 可选认证失败: {e}", exc_info=True)
        logger.info("=" * 60)
        return None

@router.post("/register", response_model=Token)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """用户注册（只能注册为普通员工）"""
    try:
        # 检查用户名是否已存在
        if db.query(User).filter(User.username == user_data.username).first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户名已存在"
            )
        
        # 检查邮箱是否已存在
        if db.query(User).filter(User.email == user_data.email).first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="邮箱已存在"
            )
        
        # 检查部门是否存在
        department = db.query(Department).filter(Department.id == user_data.department_id).first()
        if not department:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="部门不存在"
            )
        
        # 获取普通员工角色
        employee_role = db.query(Role).filter(Role.code == "employee").first()
        if not employee_role:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="系统角色配置错误"
            )
        
        # 创建新用户（强制为普通员工）
        new_user = User(
            username=user_data.username,
            email=user_data.email,
            department_id=user_data.department_id,
            role_id=employee_role.id  # 使用角色ID
        )
        new_user.set_password(user_data.password)
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # 重新查询用户以确保关联数据正确加载
        user_with_dept = db.query(User).filter(User.id == new_user.id).first()
        
        # 创建访问令牌
        access_token = create_access_token(data={"sub": user_with_dept.id})
        
        logger.info(f"用户注册成功: {user_with_dept.username} (普通员工)")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": user_with_dept.to_dict()
        }
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"用户注册失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"注册失败: {str(e)}"
        )

@router.post("/login", response_model=Token)
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """用户登录"""
    try:
        # 查找用户
        user = db.query(User).filter(User.username == user_data.username).first()
        
        if not user or not user.check_password(user_data.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="账户已被禁用"
            )
        
        # 更新最后登录时间
        user.last_login = datetime.utcnow()
        db.commit()
        
        # 创建访问令牌
        access_token = create_access_token(data={"sub": user.id})
        
        logger.info(f"用户登录成功: {user.username}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": user.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"用户登录失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登录失败"
        )

@router.get("/profile", response_model=UserProfile)
async def get_profile(current_user: User = Depends(get_current_user)):
    """获取用户资料"""
    return UserProfile(**current_user.to_dict())

@router.get("/departments")
async def get_departments(db: Session = Depends(get_db)):
    """获取所有部门列表"""
    departments = db.query(Department).all()
    return [
        {
            "id": dept.id,
            "name": dept.name,
            "description": dept.description
        }
        for dept in departments
    ]

@router.post("/logout")
async def logout():
    """用户登出（前端清除token即可）"""
    return {"message": "登出成功"}


# ==================== 用户管理API（仅超级管理员） ====================

class UserUpdateRole(BaseModel):
    """更新用户角色"""
    role: str

def require_super_admin(current_user: User = Depends(get_current_user)):
    """验证超级管理员权限"""
    if not current_user.has_permission("system.admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要超级管理员权限"
        )
    return current_user

@router.get("/users")
async def list_users(
    current_user: User = Depends(require_super_admin),
    db: Session = Depends(get_db)
):
    """获取所有用户列表（仅超级管理员）"""
    try:
        from sqlalchemy.orm import joinedload
        users = db.query(User).options(
            joinedload(User.department),
            joinedload(User.role_obj).joinedload(Role.permissions)  # 预加载角色和权限信息
        ).all()
        
        # 同时返回所有可用的角色列表，供前端下拉框使用
        roles = db.query(Role).all()
        
        return {
            "status": "success",
            "users": [user.to_dict() for user in users],
            "roles": [{"id": r.id, "code": r.code, "name": r.name} for r in roles]
        }
    except Exception as e:
        logger.error(f"获取用户列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取用户列表失败"
        )

@router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    role_data: UserUpdateRole,
    current_user: User = Depends(require_super_admin),
    db: Session = Depends(get_db)
):
    """更新用户角色（仅超级管理员）"""
    try:
        # 查找目标用户
        target_user = db.query(User).filter(User.id == user_id).first()
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        # 查找角色
        new_role = db.query(Role).filter(Role.code == role_data.role).first()
        if not new_role:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"角色不存在: {role_data.role}"
            )
        
        # 不允许修改自己的角色
        if target_user.id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="不能修改自己的角色"
            )
        
        # 更新角色
        old_role_name = target_user.role_obj.name if target_user.role_obj else "无"
        target_user.role_id = new_role.id
        db.commit()
        db.refresh(target_user)
        
        logger.info(f"超级管理员 {current_user.username} 将用户 {target_user.username} 的角色从 {old_role_name} 更改为 {new_role.name}")
        
        return {
            "status": "success",
            "message": f"用户角色已更新为 {new_role.name}",
            "user": target_user.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新用户角色失败: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新用户角色失败"
        )

@router.put("/users/{user_id}/status")
async def toggle_user_status(
    user_id: str,
    current_user: User = Depends(require_super_admin),
    db: Session = Depends(get_db)
):
    """切换用户激活状态（仅超级管理员）"""
    try:
        # 查找目标用户
        target_user = db.query(User).filter(User.id == user_id).first()
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        # 不允许禁用自己
        if target_user.id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="不能禁用自己的账户"
            )
        
        # 切换状态
        target_user.is_active = not target_user.is_active
        db.commit()
        
        status_text = "激活" if target_user.is_active else "禁用"
        logger.info(f"超级管理员 {current_user.username} {status_text}了用户 {target_user.username}")
        
        return {
            "status": "success",
            "message": f"用户已{status_text}",
            "user": target_user.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"切换用户状态失败: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="切换用户状态失败"
        )
