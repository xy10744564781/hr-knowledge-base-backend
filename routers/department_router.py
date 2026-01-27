"""
部门管理 API 路由
提供部门的 CRUD 操作（仅超级管理员）
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional

from database import get_db, User, Department
from routers.auth_router import get_current_user
from logging_setup import logger

router = APIRouter(prefix="/api/departments", tags=["部门管理"])


# ==================== Pydantic 模型 ====================

class DepartmentResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    created_at: str
    user_count: int  # 该部门的用户数量

class DepartmentCreate(BaseModel):
    name: str
    description: Optional[str] = None

class DepartmentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


# ==================== 权限检查 ====================

def require_super_admin(current_user: User = Depends(get_current_user)):
    """验证超级管理员权限"""
    if not current_user.has_permission("system.admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要超级管理员权限"
        )
    return current_user


# ==================== 部门 API ====================

@router.get("/list", response_model=List[DepartmentResponse])
async def list_departments(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取所有部门列表（所有登录用户可见）"""
    try:
        departments = db.query(Department).all()
        
        result = []
        for dept in departments:
            user_count = db.query(User).filter(User.department_id == dept.id).count()
            result.append(DepartmentResponse(
                id=dept.id,
                name=dept.name,
                description=dept.description,
                created_at=dept.created_at.isoformat() if dept.created_at else "",
                user_count=user_count
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"获取部门列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取部门列表失败"
        )


@router.get("/{department_id}", response_model=DepartmentResponse)
async def get_department(
    department_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取部门详情"""
    try:
        dept = db.query(Department).filter(Department.id == department_id).first()
        if not dept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="部门不存在"
            )
        
        user_count = db.query(User).filter(User.department_id == dept.id).count()
        
        return DepartmentResponse(
            id=dept.id,
            name=dept.name,
            description=dept.description,
            created_at=dept.created_at.isoformat() if dept.created_at else "",
            user_count=user_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取部门详情失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取部门详情失败"
        )


@router.post("/create", response_model=DepartmentResponse)
async def create_department(
    dept_data: DepartmentCreate,
    current_user: User = Depends(require_super_admin),
    db: Session = Depends(get_db)
):
    """创建新部门（仅超级管理员）"""
    try:
        # 检查部门名称是否已存在
        if db.query(Department).filter(Department.name == dept_data.name).first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"部门名称 '{dept_data.name}' 已存在"
            )
        
        # 创建部门
        new_dept = Department(
            name=dept_data.name,
            description=dept_data.description
        )
        
        db.add(new_dept)
        db.commit()
        db.refresh(new_dept)
        
        logger.info(f"创建部门成功: {new_dept.name} (by {current_user.username})")
        
        return DepartmentResponse(
            id=new_dept.id,
            name=new_dept.name,
            description=new_dept.description,
            created_at=new_dept.created_at.isoformat() if new_dept.created_at else "",
            user_count=0
        )
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"创建部门失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="创建部门失败"
        )


@router.put("/{department_id}", response_model=DepartmentResponse)
async def update_department(
    department_id: str,
    dept_data: DepartmentUpdate,
    current_user: User = Depends(require_super_admin),
    db: Session = Depends(get_db)
):
    """更新部门信息（仅超级管理员）"""
    try:
        dept = db.query(Department).filter(Department.id == department_id).first()
        if not dept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="部门不存在"
            )
        
        # 禁止修改"公共"部门的名称
        if dept.name == "公共" and dept_data.name and dept_data.name != "公共":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="'公共'部门是系统保留部门，不能修改名称"
            )
        
        # 如果要修改名称，检查新名称是否已存在
        if dept_data.name and dept_data.name != dept.name:
            if db.query(Department).filter(Department.name == dept_data.name).first():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"部门名称 '{dept_data.name}' 已存在"
                )
            dept.name = dept_data.name
        
        if dept_data.description is not None:
            dept.description = dept_data.description
        
        db.commit()
        db.refresh(dept)
        
        user_count = db.query(User).filter(User.department_id == dept.id).count()
        
        logger.info(f"更新部门成功: {dept.name} (by {current_user.username})")
        
        return DepartmentResponse(
            id=dept.id,
            name=dept.name,
            description=dept.description,
            created_at=dept.created_at.isoformat() if dept.created_at else "",
            user_count=user_count
        )
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"更新部门失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新部门失败"
        )


@router.delete("/{department_id}")
async def delete_department(
    department_id: str,
    current_user: User = Depends(require_super_admin),
    db: Session = Depends(get_db)
):
    """删除部门（仅超级管理员）"""
    try:
        dept = db.query(Department).filter(Department.id == department_id).first()
        if not dept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="部门不存在"
            )
        
        # 禁止删除"公共"部门
        if dept.name == "公共":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="'公共'部门是系统保留部门，不能删除"
            )
        
        # 检查是否有用户属于该部门
        user_count = db.query(User).filter(User.department_id == department_id).count()
        if user_count > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"该部门还有 {user_count} 个用户，无法删除。请先将用户转移到其他部门。"
            )
        
        dept_name = dept.name
        db.delete(dept)
        db.commit()
        
        logger.info(f"删除部门成功: {dept_name} (by {current_user.username})")
        
        return {
            "status": "success",
            "message": f"部门 '{dept_name}' 已删除"
        }
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"删除部门失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除部门失败"
        )


@router.get("/stats/summary")
async def get_department_stats(
    current_user: User = Depends(require_super_admin),
    db: Session = Depends(get_db)
):
    """获取部门统计信息（仅超级管理员）"""
    try:
        departments = db.query(Department).all()
        
        stats = []
        total_users = 0
        
        for dept in departments:
            user_count = db.query(User).filter(User.department_id == dept.id).count()
            total_users += user_count
            
            stats.append({
                "department_id": dept.id,
                "department_name": dept.name,
                "user_count": user_count
            })
        
        # 统计没有部门的用户（超级管理员）
        no_dept_count = db.query(User).filter(User.department_id.is_(None)).count()
        
        return {
            "status": "success",
            "total_departments": len(departments),
            "total_users": total_users,
            "users_without_department": no_dept_count,
            "department_stats": stats
        }
        
    except Exception as e:
        logger.error(f"获取部门统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取部门统计失败"
        )
