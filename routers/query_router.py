from fastapi import APIRouter, Query, Body
from fastapi.responses import StreamingResponse
from typing import Optional, List
import json
import asyncio
from schemas import QueryRequest, QueryResponse, UserContext
from services import service_query_knowledge, service_query_knowledge_stream

router = APIRouter()

@router.post("/query-stream")
async def query_knowledge_stream(request: QueryRequest):
    """
    人事知识查询 - 流式响应版本
    
    实时流式返回AI生成的回答，提供更好的用户体验
    
    - **question**: 用户的查询问题
    - **user_ctx**: 用户上下文信息（部门、角色等）
    
    返回Server-Sent Events (SSE)格式的流式数据
    """
    async def generate_stream():
        try:
            # 发送开始事件
            yield f"data: {json.dumps({'type': 'start', 'message': '开始处理查询...'}, ensure_ascii=False)}\n\n"
            
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
async def query_knowledge(request: QueryRequest) -> QueryResponse:
    """
    人事知识查询 - 完整版本
    
    基于向量搜索和本地大模型，提供智能的人事政策和流程查询服务
    
    - **question**: 用户的查询问题
    - **user_ctx**: 用户上下文信息（部门、角色等）
    
    返回AI生成的回答、参考来源和置信度
    """
    return service_query_knowledge(request)

@router.get("/query")
async def query_knowledge_simple(
    question: str = Query(..., description="查询问题"),
    user_role: str = Query("employee", description="用户角色"),
    department: str = Query("HR", description="用户部门"),
    user_id: Optional[str] = Query(None, description="用户ID")
) -> QueryResponse:
    """
    人事知识查询 - 简化版本
    
    通过URL参数进行简单查询，适用于快速集成
    
    - **question**: 查询问题
    - **user_role**: 用户角色（employee, hr_staff, hr_manager, hr_director）
    - **department**: 用户部门（默认HR）
    - **user_id**: 可选的用户ID
    """
    # 构建用户上下文
    user_ctx = UserContext(
        user_id=user_id or f"guest_{hash(question) % 10000}",
        user_role=user_role,
        department=department
    )
    
    # 构建查询请求
    request = QueryRequest(
        question=question,
        user_ctx=user_ctx
    )
    
    return service_query_knowledge(request)

@router.post("/batch-query")
async def batch_query_knowledge(
    questions: List[str] = Body(..., description="批量查询问题列表"),
    user_ctx: UserContext = Body(..., description="用户上下文")
) -> dict:
    """
    批量人事知识查询
    
    一次性处理多个查询问题，提高效率
    
    - **questions**: 查询问题列表（最多10个）
    - **user_ctx**: 用户上下文信息
    """
    if len(questions) > 10:
        return {
            "status": "error",
            "message": "批量查询最多支持10个问题",
            "results": []
        }
    
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
    category: Optional[str] = Query(None, description="查询分类")
) -> dict:
    """
    获取查询建议
    
    提供常见的人事查询问题建议
    
    - **category**: 可选的查询分类过滤
    """
    # 常见查询建议
    suggestions = {
        "salary": [
            "薪资什么时候发放？",
            "薪资构成包括哪些部分？",
            "如何申请薪资调整？",
            "年终奖发放标准是什么？"
        ],
        "leave": [
            "年假如何申请？",
            "病假需要什么手续？",
            "事假申请流程是什么？",
            "产假政策是怎样的？"
        ],
        "attendance": [
            "考勤制度是什么？",
            "迟到早退如何处理？",
            "加班费如何计算？",
            "打卡时间要求是什么？"
        ],
        "onboarding": [
            "新员工入职需要准备什么材料？",
            "入职培训安排是怎样的？",
            "试用期政策是什么？",
            "入职手续办理流程？"
        ],
        "benefits": [
            "员工福利有哪些？",
            "社保缴费标准是什么？",
            "公积金政策是怎样的？",
            "员工体检安排？"
        ],
        "training": [
            "培训计划是什么？",
            "如何申请培训？",
            "培训证书如何获得？",
            "职业发展路径？"
        ]
    }
    
    if category and category in suggestions:
        return {
            "status": "success",
            "category": category,
            "suggestions": suggestions[category]
        }
    
    # 返回所有建议
    all_suggestions = []
    for cat, items in suggestions.items():
        all_suggestions.extend(items[:2])  # 每个分类取前2个
    
    return {
        "status": "success",
        "category": "all",
        "suggestions": all_suggestions,
        "categories": list(suggestions.keys())
    }
