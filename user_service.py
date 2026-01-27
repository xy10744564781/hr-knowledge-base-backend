from sqlalchemy.orm import Session
from database import get_db_session, User, Department
from typing import Optional, Dict, List
from logging_setup import logger

class UserService:
    """用户服务类"""
    
    @staticmethod
    def get_user_by_id(user_id: str) -> Optional[User]:
        """根据ID获取用户"""
        db = get_db_session()
        try:
            from sqlalchemy.orm import joinedload
            from database import Role
            user = db.query(User).options(
                joinedload(User.department),
                joinedload(User.role_obj).joinedload(Role.permissions)  # 预加载角色和权限信息
            ).filter(User.id == user_id).first()
            if user:
                # 确保部门信息已加载，避免懒加载问题
                _ = user.department.name if user.department else None
                # 确保角色信息已加载
                _ = user.role_obj.code if user.role_obj else None
                # 确保权限信息已加载
                _ = len(user.role_obj.permissions) if user.role_obj else 0
            return user
        finally:
            db.close()
    
    @staticmethod
    def get_user_context(user_id: str) -> Dict:
        """获取用户上下文信息"""
        db = get_db_session()
        try:
            from sqlalchemy.orm import joinedload
            from database import Role
            user = db.query(User).options(
                joinedload(User.department),
                joinedload(User.role_obj).joinedload(Role.permissions)
            ).filter(User.id == user_id).first()
            
            if not user:
                # 返回默认上下文（兼容旧版本）
                return {
                    "user_id": "anonymous",
                    "username": "匿名用户",
                    "department": "公共",
                    "department_id": None,
                    "role": "employee",
                    "accessible_folders": ["公共"],
                    "can_upload": False
                }
            
            # 在 session 关闭前获取所有需要的数据
            user_id = user.id
            username = user.username
            department_name = user.department.name if user.department else "公共"
            department_id = user.department_id
            user_role_code = user.role_obj.code if user.role_obj else "employee"
            accessible_folders = user.get_accessible_folders()  # 在 session 关闭前调用
            can_upload = user.can_upload()
            
            return {
                "user_id": user_id,
                "username": username,
                "department": department_name,
                "department_id": department_id,
                "role": user_role_code,
                "accessible_folders": accessible_folders,
                "can_upload": can_upload
            }
        finally:
            db.close()
    
    @staticmethod
    def build_search_filters(user_context: Dict) -> Dict:
        """构建搜索过滤条件"""
        accessible_folders = user_context.get("accessible_folders", ["公共"])
        
        # ChromaDB过滤条件
        return {
            "department": {"$in": accessible_folders}
        }
    
    @staticmethod
    def get_search_strategy(question: str, user_context: Dict) -> Dict:
        """获取搜索策略"""
        accessible_folders = user_context.get("accessible_folders", ["公共"])
        user_department = user_context.get("department", "公共")
        
        # 简单的意图分析
        question_lower = question.lower()
        
        # 部门关键词映射
        department_keywords = {
            "人事": ["人事", "hr", "薪资", "工资", "请假", "考勤", "招聘", "离职", "入职"],
            "质量": ["质量", "quality", "检测", "标准", "认证", "审核"],
            "技术": ["技术", "开发", "代码", "系统", "软件", "硬件"],
            "财务": ["财务", "finance", "会计", "报销", "预算", "成本"],
            "销售": ["销售", "sales", "客户", "合同", "订单", "市场"],
            "运营": ["运营", "operations", "流程", "管理", "优化"]
        }
        
        # 分析问题意图
        detected_department = None
        for dept, keywords in department_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_department = dept
                break
        
        # 确定搜索优先级
        if detected_department and detected_department in accessible_folders:
            # 用户有权限访问相关部门
            primary_folders = [detected_department]
            secondary_folders = ["公共"] if "公共" in accessible_folders else []
        elif detected_department and detected_department not in accessible_folders:
            # 用户没有权限，只能在公共文件夹中搜索
            primary_folders = ["公共"] if "公共" in accessible_folders else []
            secondary_folders = []
        else:
            # 通用问题，优先搜索用户部门
            if user_department != "公共" and user_department in accessible_folders:
                primary_folders = [user_department]
                secondary_folders = ["公共"] if "公共" in accessible_folders else []
            else:
                primary_folders = ["公共"] if "公共" in accessible_folders else []
                secondary_folders = []
        
        return {
            "detected_department": detected_department,
            "primary_folders": primary_folders,
            "secondary_folders": secondary_folders,
            "search_folders": primary_folders + secondary_folders
        }

# 全局用户服务实例
user_service = UserService()