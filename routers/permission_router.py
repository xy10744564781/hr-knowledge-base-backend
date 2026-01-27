"""
权限管理 API 路由
提供角色和权限的 CRUD 操作
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional

from database import get_db, User, Role, Permission, role_permissions
from routers.auth_router import get_current_user
from logging_setup import logger

router = APIRouter(prefix="/api/permissions", tags=["权限管理"])


# ==================== Pydantic 模型 ====================

class PermissionResponse(BaseModel):
    id: str
    code: str
    name: str
    description: Optional[str]
    category: Optional[str]

class RoleResponse(BaseModel):
    id: str
    code: str
    name: str
    description: Optional[str]
    is_system: bool
    permissions: Optional[List[PermissionResponse]] = None

class RoleCreate(BaseModel):
    code: str
    name: str
    description: Optional[str] = None
    permission_codes: List[str] = []

class RoleUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    permission_codes: Optional[List[str]] = None


# ==================== 权限检查 ====================

def require_permission(permission_code: str):
    """要求特定权限的依赖"""
    def permission_checker(current_user: User = Depends(get_current_user)):
        if not current_user.has_permission(permission_code):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"需要权限: {permission_code}"
            )
        return current_user
    return permission_checker


# ==================== 权限 API ====================

@router.get("/list", response_model=List[PermissionResponse])
async def list_permissions(
    category: Optional[str] = None,
    current_user: User = Depends(require_permission("system.admin")),
    db: Session = Depends(get_db)
):
    """获取所有权限列表（仅系统管理员）"""
    try:
        query = db.query(Permission)
        if category:
            query = query.filter(Permission.category == category)
        
        permissions = query.all()
        return [PermissionResponse(**p.to_dict()) for p in permissions]
        
    except Exception as e:
        logger.error(f"获取权限列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取权限列表失败"
        )


@router.get("/categories")
async def list_permission_categories(
    current_user: User = Depends(require_permission("system.admin")),
    db: Session = Depends(get_db)
):
    """获取权限分类列表"""
    try:
        categories = db.query(Permission.category).distinct().all()
        return [cat[0] for cat in categories if cat[0]]
    except Exception as e:
        logger.error(f"获取权限分类失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取权限分类失败"
        )


# ==================== 角色 API ====================

@router.get("/roles", response_model=List[RoleResponse])
async def list_roles(
    include_permissions: bool = False,
    current_user: User = Depends(require_permission("user.manage")),
    db: Session = Depends(get_db)
):
    """获取所有角色列表"""
    try:
        roles = db.query(Role).all()
        return [RoleResponse(**r.to_dict(include_permissions=include_permissions)) for r in roles]
    except Exception as e:
        logger.error(f"获取角色列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取角色列表失败"
        )


@router.get("/roles/{role_id}", response_model=RoleResponse)
async def get_role(
    role_id: str,
    current_user: User = Depends(require_permission("user.manage")),
    db: Session = Depends(get_db)
):
    """获取角色详情"""
    try:
        role = db.query(Role).filter(Role.id == role_id).first()
        if not role:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="角色不存在"
            )
        return RoleResponse(**role.to_dict(include_permissions=True))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取角色详情失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取角色详情失败"
        )


@router.post("/roles", response_model=RoleResponse)
async def create_role(
    role_data: RoleCreate,
    current_user: User = Depends(require_permission("system.admin")),
    db: Session = Depends(get_db)
):
    """创建新角色（仅系统管理员）"""
    try:
        # 检查角色代码是否已存在
        if db.query(Role).filter(Role.code == role_data.code).first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="角色代码已存在"
            )
        
        # 创建角色
        new_role = Role(
            code=role_data.code,
            name=role_data.name,
            description=role_data.description,
            is_system=False
        )
        
        # 添加权限
        if role_data.permission_codes:
            permissions = db.query(Permission).filter(
                Permission.code.in_(role_data.permission_codes)
            ).all()
            new_role.permissions = permissions
        
        db.add(new_role)
        db.commit()
        db.refresh(new_role)
        
        logger.info(f"创建角色成功: {new_role.name} (by {current_user.username})")
        return RoleResponse(**new_role.to_dict(include_permissions=True))
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"创建角色失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="创建角色失败"
        )


@router.put("/roles/{role_id}", response_model=RoleResponse)
async def update_role(
    role_id: str,
    role_data: RoleUpdate,
    current_user: User = Depends(require_permission("system.admin")),
    db: Session = Depends(get_db)
):
    """更新角色（仅系统管理员）"""
    try:
        role = db.query(Role).filter(Role.id == role_id).first()
        if not role:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="角色不存在"
            )
        
        # 系统内置角色不允许修改权限
        if role.is_system and role_data.permission_codes is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="系统内置角色不允许修改权限"
            )
        
        # 更新基本信息
        if role_data.name:
            role.name = role_data.name
        if role_data.description is not None:
            role.description = role_data.description
        
        # 更新权限
        if role_data.permission_codes is not None:
            permissions = db.query(Permission).filter(
                Permission.code.in_(role_data.permission_codes)
            ).all()
            role.permissions = permissions
        
        db.commit()
        db.refresh(role)
        
        logger.info(f"更新角色成功: {role.name} (by {current_user.username})")
        return RoleResponse(**role.to_dict(include_permissions=True))
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"更新角色失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新角色失败"
        )


@router.delete("/roles/{role_id}")
async def delete_role(
    role_id: str,
    current_user: User = Depends(require_permission("system.admin")),
    db: Session = Depends(get_db)
):
    """删除角色（仅系统管理员）"""
    try:
        role = db.query(Role).filter(Role.id == role_id).first()
        if not role:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="角色不存在"
            )
        
        # 系统内置角色不允许删除
        if role.is_system:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="系统内置角色不允许删除"
            )
        
        # 检查是否有用户使用该角色
        user_count = db.query(User).filter(User.role_id == role_id).count()
        if user_count > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"该角色正在被 {user_count} 个用户使用，无法删除"
            )
        
        db.delete(role)
        db.commit()
        
        logger.info(f"删除角色成功: {role.name} (by {current_user.username})")
        return {"status": "success", "message": "角色已删除"}
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"删除角色失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除角色失败"
        )


# ==================== 用户权限查询 ====================

@router.get("/my-permissions")
async def get_my_permissions(
    current_user: User = Depends(get_current_user)
):
    """获取当前用户的权限列表"""
    try:
        if not current_user.role_obj:
            return {"permissions": []}
        
        permissions = [p.to_dict() for p in current_user.role_obj.permissions]
        return {
            "role": current_user.role_obj.to_dict(),
            "permissions": permissions
        }
    except Exception as e:
        logger.error(f"获取用户权限失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取用户权限失败"
        )
