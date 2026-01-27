from fastapi import APIRouter, Query, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional, List
import json
import asyncio
from schemas import QueryRequest, QueryResponse, UserContext
from services import service_query_knowledge, service_query_knowledge_stream
from routers.auth_router import get_current_user, get_current_user_optional
from database import User
from user_service import user_service

router = APIRouter()

@router.post("/query-stream")
async def query_knowledge_stream(
    request: QueryRequest,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    企业知识查询 - 流式响应版本
    
    实时流式返回AI生成的回答，提供更好的用户体验
    支持基于用户权限的多部门知识库检索
    
    - **question**: 用户的查询问题
    - **session_id**: 会话ID（支持对话历史）
    - **selected_folders**: 可选，用户选择的检索范围
    
    返回Server-Sent Events (SSE)格式的流式数据
    """
    async def generate_stream():
        try:
            # 发送开始事件
            yield f"data: {json.dumps({'type': 'start', 'message': '开始处理查询...'}, ensure_ascii=False)}\n\n"
            
            # 获取用户上下文
            if current_user:
                user_context = user_service.get_user_context(current_user.id)
                # 安全获取部门名称，避免懒加载错误
                try:
                    department_name = current_user.department.name if current_user.department else "人事"
                except Exception as e:
                    logger.warning(f"获取用户部门信息失败: {e}")
                    department_name = "人事"  # 默认部门
                
                # 更新请求中的用户上下文
                # 获取用户角色代码（兼容RBAC模式）
                user_role_code = current_user.role_obj.code if current_user.role_obj else "employee"
                
                request.user_ctx = UserContext(
                    user_id=str(current_user.id),
                    user_role=user_role_code,
                    department=department_name
                )
            else:
                # 匿名用户，只能访问公共文件夹
                user_context = {
                    "user_id": "anonymous",
                    "department": "公共",
                    "role": "employee",
                    "accessible_folders": ["公共"],
                    "can_upload": False
                }
                request.user_ctx = UserContext(
                    user_id="anonymous",
                    user_role="employee",
                    department="公共"
                )
            
            # 发送用户权限信息
            yield f"data: {json.dumps({'type': 'user_context', 'data': user_context}, ensure_ascii=False)}\n\n"
            
            # 异步生成流式响应
            async for chunk in service_query_knowledge_stream(request):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            
            # 发送结束事件
            yield f"data: {json.dumps({'type': 'end', 'message': '查询完成'}, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            # 发送错误事件
            error_data = {
                'type': 'error',
                'message': f'查询过程中出现错误: {str(e)}'
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@router.post("/query", response_model=QueryResponse)
async def query_knowledge(
    request: QueryRequest,
    current_user: Optional[User] = Depends(get_current_user_optional)
) -> QueryResponse:
    """
    企业知识查询 - 完整版本
    
    基于向量搜索和大模型API，提供智能的企业知识查询服务
    支持多部门权限控制和智能路由
    
    - **question**: 用户的查询问题
    - **selected_folders**: 可选，用户选择的检索范围
    
    返回AI生成的回答、参考来源和置信度
    """
    # 获取用户上下文
    if current_user:
        # 获取用户角色代码（兼容RBAC模式）
        user_role_code = current_user.role_obj.code if current_user.role_obj else "employee"
        
        request.user_ctx = UserContext(
            user_id=str(current_user.id),
            user_role=user_role_code,
            department=current_user.department.name
        )
    else:
        # 匿名用户
        request.user_ctx = UserContext(
            user_id="anonymous",
            user_role="employee",
            department="公共"
        )
    
    return service_query_knowledge(request)

@router.get("/query")
async def query_knowledge_simple(
    question: str = Query(..., description="查询问题"),
    current_user: Optional[User] = Depends(get_current_user_optional)
) -> QueryResponse:
    """
    企业知识查询 - 简化版本
    
    通过URL参数进行简单查询，适用于快速集成
    
    - **question**: 查询问题
    """
    # 构建用户上下文
    if current_user:
        # 获取用户角色代码（兼容RBAC模式）
        user_role_code = current_user.role_obj.code if current_user.role_obj else "employee"
        
        user_ctx = UserContext(
            user_id=str(current_user.id),
            user_role=user_role_code,
            department=current_user.department.name
        )
    else:
        user_ctx = UserContext(
            user_id="anonymous",
            user_role="employee",
            department="公共"
        )
    
    # 构建查询请求
    request = QueryRequest(
        question=question,
        user_ctx=user_ctx
    )
    
    return service_query_knowledge(request)

@router.get("/accessible-folders")
async def get_accessible_folders(current_user: User = Depends(get_current_user)):
    """获取用户可访问的文件夹列表"""
    return {
        "folders": current_user.get_accessible_folders(),
        "can_upload": current_user.can_upload(),
        "user_department": current_user.department.name
    }

@router.post("/batch-query")
async def batch_query_knowledge(
    questions: List[str] = Body(..., description="批量查询问题列表"),
    current_user: Optional[User] = Depends(get_current_user_optional)
) -> dict:
    """
    批量企业知识查询
    
    一次性处理多个查询问题，提高效率
    
    - **questions**: 查询问题列表（最多10个）
    """
    if len(questions) > 10:
        return {
            "status": "error",
            "message": "批量查询最多支持10个问题",
            "results": []
        }
    
    # 构建用户上下文
    if current_user:
        # 获取用户角色代码（兼容RBAC模式）
        user_role_code = current_user.role_obj.code if current_user.role_obj else "employee"
        
        user_ctx = UserContext(
            user_id=str(current_user.id),
            user_role=user_role_code,
            department=current_user.department.name
        )
    else:
        user_ctx = UserContext(
            user_id="anonymous",
            user_role="employee",
            department="公共"
        )
    
    results = []
    for i, question in enumerate(questions):
        try:
            request = QueryRequest(question=question, user_ctx=user_ctx)
            response = service_query_knowledge(request)
            results.append({
                "index": i,
                "question": question,
                "status": "success",
                "response": response.dict()
            })
        except Exception as e:
            results.append({
                "index": i,
                "question": question,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "status": "success",
        "total_questions": len(questions),
        "results": results
    }

@router.get("/query-suggestions")
async def get_query_suggestions(
    category: Optional[str] = Query(None, description="查询分类"),
    current_user: Optional[User] = Depends(get_current_user_optional)
) -> dict:
    """
    获取查询建议
    
    根据用户部门提供相关的查询问题建议
    
    - **category**: 可选的查询分类过滤
    """
    # 部门相关查询建议
    department_suggestions = {
        "人事": [
            "薪资什么时候发放？",
            "年假如何申请？",
            "考勤制度是什么？",
            "新员工入职流程？"
        ],
        "质量": [
            "质量管理体系是什么？",
            "产品检测标准？",
            "质量认证流程？",
            "不合格品处理流程？"
        ],
        "技术": [
            "开发规范是什么？",
            "代码审查流程？",
            "技术文档模板？",
            "系统部署流程？"
        ],
        "财务": [
            "报销流程是什么？",
            "预算申请流程？",
            "发票管理规定？",
            "成本核算方法？"
        ],
        "销售": [
            "销售流程是什么？",
            "客户管理规定？",
            "合同审批流程？",
            "销售目标制定？"
        ],
        "运营": [
            "运营流程优化？",
            "数据分析方法？",
            "项目管理流程？",
            "绩效考核标准？"
        ]
    }
    
    # 公共查询建议
    public_suggestions = [
        "公司组织架构？",
        "员工手册内容？",
        "公司文化价值观？",
        "办公设施使用规定？"
    ]
    
    if current_user:
        user_dept = current_user.department.name
        dept_suggestions = department_suggestions.get(user_dept, [])
        all_suggestions = dept_suggestions + public_suggestions
    else:
        all_suggestions = public_suggestions
    
    if category and category in department_suggestions:
        return {
            "status": "success",
            "category": category,
            "suggestions": department_suggestions[category]
        }
    
    return {
        "status": "success",
        "category": "all",
        "suggestions": all_suggestions[:8],  # 返回前8个建议
        "departments": list(department_suggestions.keys())
    }
