"""
用户信息查询链

从数据库获取用户上下文信息，包括部门、角色、权限等
"""

from typing import Dict, Any, Optional, List
from .base_chain import BaseKnowledgeChain, ChainErrorHandler
from .models import UserContext, UserRole, ChainInput
from user_service import user_service


class UserContextChain(BaseKnowledgeChain):
    """用户信息查询链"""
    
    def __init__(self, **kwargs):
        super().__init__(chain_name="user_context", **kwargs)
    
    def _execute_chain(
        self, 
        inputs: Dict[str, Any], 
        run_manager: Optional[Any] = None
    ) -> Dict[str, Any]:
        """执行用户信息查询"""
        chain_input: ChainInput = inputs["input_data"]
        user_id = chain_input.user_id
        
        self.logger.info(f"查询用户信息: user_id={user_id}")
        
        try:
            # 从用户服务获取用户上下文
            user_context_dict = user_service.get_user_context(user_id)
            
            # 转换为UserContext对象
            user_context = self._dict_to_user_context(user_context_dict)
            
            self.logger.info(
                f"用户信息查询成功: {user_context.username} "
                f"({user_context.department}, {user_context.role})"
            )
            
            return {
                "user_context": user_context,
                "raw_user_data": user_context_dict
            }
            
        except Exception as e:
            self.logger.error(f"用户信息查询失败: {str(e)}")
            
            # 返回匿名用户上下文作为降级处理
            fallback_context = self._create_anonymous_context()
            
            return {
                "user_context": fallback_context,
                "raw_user_data": {},
                "fallback": True,
                "error": str(e)
            }
    
    def _dict_to_user_context(self, user_dict: Dict[str, Any]) -> UserContext:
        """将字典转换为UserContext对象"""
        # 处理角色枚举
        role_str = user_dict.get("role", "employee")
        try:
            role = UserRole(role_str)
        except ValueError:
            self.logger.warning(f"未知用户角色: {role_str}，使用默认角色 'employee'")
            role = UserRole.EMPLOYEE
        
        # 获取可访问文件夹列表
        accessible_folders = user_dict.get("accessible_folders", ["公共"])
        
        # 添加调试日志
        self.logger.info(f"[DEBUG] _dict_to_user_context: role={role_str}, accessible_folders={accessible_folders}")
        
        return UserContext(
            user_id=user_dict.get("user_id", ""),
            username=user_dict.get("username", "未知用户"),
            department=user_dict.get("department", "公共"),
            department_id=user_dict.get("department_id", ""),
            role=role,
            accessible_folders=accessible_folders,
            can_upload=user_dict.get("can_upload", False),
            permissions={}
        )
    
    def _create_anonymous_context(self) -> UserContext:
        """创建匿名用户上下文"""
        return UserContext(
            user_id="anonymous",
            username="匿名用户",
            department="公共",
            department_id="",
            role=UserRole.EMPLOYEE,
            accessible_folders=["公共"],
            can_upload=False,
            permissions={}
        )
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """验证输入数据"""
        super()._validate_inputs(inputs)
        
        chain_input = inputs["input_data"]
        if not isinstance(chain_input, ChainInput):
            raise ValueError("input_data 必须是 ChainInput 类型")
        
        if not chain_input.user_id:
            raise ValueError("user_id 不能为空")


class UserPermissionCalculator:
    """用户权限计算器"""
    
    @staticmethod
    def calculate_accessible_folders(user_context: UserContext) -> List[str]:
        """计算用户可访问的文件夹"""
        if user_context.role == UserRole.SUPER_ADMIN:
            # 超级管理员可以访问所有部门 - 从数据库动态读取
            try:
                from database_rbac import get_db_session, Department
                db = get_db_session()
                try:
                    all_departments = db.query(Department).all()
                    dept_names = [dept.name for dept in all_departments]
                    return ["公共"] + dept_names
                finally:
                    db.close()
            except Exception as e:
                # 如果数据库读取失败，返回默认列表
                logger.warning(f"从数据库读取部门列表失败: {e}")
                return ["公共", "人事", "质量", "技术", "财务", "销售", "运营"]
        else:
            # 普通用户和部门管理员可以访问自己部门和公共文件夹
            folders = ["公共"]
            if user_context.department and user_context.department != "公共":
                folders.append(user_context.department)
            return folders
    
    @staticmethod
    def can_upload_to_folder(user_context: UserContext, folder: str) -> bool:
        """检查用户是否可以上传到指定文件夹"""
        # 只有管理员和超级管理员可以上传
        if user_context.role not in [UserRole.ADMIN, UserRole.SUPER_ADMIN]:
            return False
        
        # 超级管理员可以上传到任何文件夹
        if user_context.role == UserRole.SUPER_ADMIN:
            return True
        
        # 部门管理员只能上传到自己部门的文件夹
        return folder == user_context.department
    
    @staticmethod
    def can_access_folder(user_context: UserContext, folder: str) -> bool:
        """检查用户是否可以访问指定文件夹"""
        return folder in user_context.accessible_folders
    
    @staticmethod
    def get_search_priority(user_context: UserContext, query: str) -> List[str]:
        """根据查询内容获取搜索优先级"""
        accessible_folders = user_context.accessible_folders
        
        # 如果用户只能访问公共文件夹，直接返回
        if accessible_folders == ["公共"]:
            return ["公共"]
        
        # 根据查询内容确定优先级
        query_lower = query.lower()
        
        # 部门关键词映射
        department_keywords = {
            "人事": ["人事", "hr", "薪资", "工资", "请假", "考勤", "招聘", "离职", "入职"],
            "质量": ["质量", "quality", "检测", "标准", "认证", "审核"],
            "技术": ["技术", "开发", "代码", "系统", "软件", "硬件"],
            "财务": ["财务", "finance", "会计", "报销", "预算", "成本"],
            "销售": ["销售", "sales", "客户", "合同", "订单", "市场"],
            "运营": ["运营", "operations", "流程", "管理", "优化"]
        }
        
        # 检测查询意图
        detected_department = None
        for dept, keywords in department_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                if dept in accessible_folders:
                    detected_department = dept
                    break
        
        # 构建优先级列表
        if detected_department:
            # 优先搜索检测到的部门
            priority = [detected_department]
            # 添加其他可访问的文件夹
            for folder in accessible_folders:
                if folder != detected_department:
                    priority.append(folder)
        else:
            # 优先搜索用户自己的部门
            if user_context.department in accessible_folders and user_context.department != "公共":
                priority = [user_context.department]
                # 添加其他文件夹
                for folder in accessible_folders:
                    if folder != user_context.department:
                        priority.append(folder)
            else:
                # 使用默认顺序
                priority = accessible_folders.copy()
        
        return priority