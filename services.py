import time
import json
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from fastapi import HTTPException
from logging_setup import logger
from knowledge_base import init_chroma, process_upload_file
from knowledge_base import get_vector_manager
from schemas import (
    QueryRequest, QueryResponse, DocumentUploadResponse, 
    HealthResponse, VectorStoreStatus, UserContext, DocumentInfo,
    DocumentCategory, AccessLevel
)
from llm_agent import integrate_results, analyze_query, get_hr_agent
from typing import List, Optional, AsyncGenerator, Dict
from config import QUERY_TIMEOUT, MAX_QUERY_RESULTS

# dev-mix新增：导入对话历史管理器
from chat_history_manager import get_history_manager, sync_history_to_manager

# 初始化向量库和智能路由器
try:
    # 设置环境变量解决protobuf版本冲突和禁用遥测
    import os
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    os.environ['ANONYMIZED_TELEMETRY'] = 'False'
    os.environ['CHROMA_TELEMETRY'] = 'False'
    
    from knowledge_base import init_chroma, get_vector_manager
    from query_router import create_query_router
    
    vector_store = init_chroma()
    vector_manager = get_vector_manager()
    
    # 保留旧的查询路由器作为降级选项
    legacy_query_router = create_query_router()
    
    # 尝试初始化新的智能路由器
    intelligent_router = None
    try:
        from langchain_chains.intelligent_router import get_intelligent_router
        intelligent_router = get_intelligent_router()
        logger.info('智能多部门路由器初始化成功')
    except Exception as router_error:
        logger.warning(f'智能路由器初始化失败，将使用传统路由器: {router_error}')
        intelligent_router = None
    
    # 初始化响应生成器（保留作为降级选项）
    hr_agent = get_hr_agent()
    
    if vector_store:
        logger.info('ChromaDB向量库初始化成功（使用LangChain + 阿里云API）')
        if intelligent_router:
            logger.info('系统运行在智能多部门模式')
        else:
            logger.info('系统运行在传统路由模式')
    else:
        logger.warning('ChromaDB向量库初始化失败，系统将以简化模式运行')
        
except Exception as e:
    logger.error(f'向量库初始化过程中发生错误: {str(e)}', exc_info=True)
    vector_store = None
    vector_manager = None
    legacy_query_router = None
    intelligent_router = None
    
    # 简化模式下的响应生成器
    hr_agent = get_hr_agent()

def service_query_knowledge(request: QueryRequest) -> QueryResponse:
    """处理企业知识查询请求 - 使用智能多部门路由器"""
    start_time = time.time()
    logger.info(f"收到企业知识查询: {request.question}")

    try:
        # 优先使用新的智能路由器
        if intelligent_router:
            logger.info("使用智能多部门路由器处理查询")
            
            # 使用智能路由器处理查询
            query_result = intelligent_router.route_query(
                query=request.question,
                user_id=request.user_ctx.user_id,
                session_id=request.session_id
            )
            
            # 转换为QueryResponse格式
            response = QueryResponse(
                answer=query_result.answer,
                source_data=[{
                    "tool": "intelligent_multi_department_router",
                    "source_type": query_result.source_type,
                    "user_department": query_result.user_context.department,
                    "intent_analysis": {
                        "primary_intent": query_result.intent_analysis.primary_intent,
                        "confidence": query_result.intent_analysis.confidence,
                        "detected_department": query_result.intent_analysis.detected_department
                    },
                    "retrieval_strategy": {
                        "primary_folders": query_result.retrieval_strategy.primary_folders,
                        "secondary_folders": query_result.retrieval_strategy.secondary_folders
                    },
                    "documents": [
                        {
                            "title": doc.title,
                            "department": doc.department,
                            "score": doc.score,
                            "document_id": doc.document_id
                        }
                        for doc in query_result.documents
                    ]
                }],
                confidence=query_result.confidence,
                processing_time=query_result.processing_time
            )
            
            logger.info("智能多部门路由器处理完成")
            return response
        
        # 降级到旧的查询路由器
        elif legacy_query_router and vector_store:
            logger.warning("智能路由器不可用，降级到传统路由器")
            return _legacy_query_processing(request, start_time)
        
        else:
            raise HTTPException(status_code=500, detail="查询系统未正确初始化")
            
    except Exception as e:
        logger.error(f"查询处理失败: {str(e)}")
        return _handle_query_error(e, start_time)


def _legacy_query_processing(request: QueryRequest, start_time: float) -> QueryResponse:
    """传统查询处理逻辑（降级处理）"""
    logger.info("使用传统查询处理逻辑")
    
    # 查询预处理
    processed_query = _preprocess_query(request.question)
    logger.info(f"查询预处理完成: {processed_query}")

    # 使用传统查询路由器进行智能路由
    user_ctx_dict = _build_user_context(request.user_ctx)
    route_result = legacy_query_router.route(processed_query, vector_store, user_ctx_dict)
    
    strategy = route_result['strategy']
    documents = route_result['documents']
    evaluation = route_result['evaluation']
    
    logger.info(f"传统路由策略: {strategy}, 文档数: {len(documents)}")
    
    # 根据策略生成回答
    if strategy == 'document_based':
        # 基于文档的回答
        answer = hr_agent.generate_response(processed_query, documents, user_ctx_dict)
    else:
        # 通用知识回答
        answer = hr_agent.generate_response(processed_query, [], user_ctx_dict)
    
    logger.info("传统回答生成完成")

    # 构建响应数据
    response_data = _build_response_data(
        answer=answer,
        vector_results=documents,
        query_analysis={'strategy': strategy},
        original_query=request.question,
        processed_query=processed_query,
        start_time=start_time,
        evaluation=evaluation
    )

    logger.info("传统查询处理成功完成")
    return response_data

def _handle_timeout_error(query: str, start_time: float) -> QueryResponse:
    """处理查询超时错误"""
    return QueryResponse(
        answer=f"查询「{query}」处理超时，请尝试：\n\n• 简化查询内容\n• 使用更具体的关键词\n• 稍后再试\n\n如问题持续，请联系技术支持。",
        source_data=[{
            "tool": "timeout_handler",
            "error_type": "QueryTimeout",
            "timeout_seconds": QUERY_TIMEOUT,
            "timestamp": datetime.now().isoformat()
        }],
        confidence=0.0,
        processing_time=time.time() - start_time
    )

def _preprocess_query(query: str) -> str:
    """查询预处理和优化"""
    try:
        # 去除多余空格和特殊字符
        processed = query.strip()
        
        # 标准化常见人事术语
        hr_term_mapping = {
            '工资': '薪资',
            '薪水': '薪资',
            '年假': '年休假',
            '病假': '病事假',
            '迟到': '考勤',
            '早退': '考勤',
            '新人': '新员工',
            '离职手续': '离职流程'
        }
        
        for old_term, new_term in hr_term_mapping.items():
            processed = processed.replace(old_term, new_term)
        
        # 添加查询扩展（同义词）
        if len(processed) < 10:  # 短查询需要扩展
            expansion_map = {
                '薪资': '薪资 工资 薪酬',
                '考勤': '考勤 打卡 出勤',
                '请假': '请假 休假 假期',
                '培训': '培训 学习 发展'
            }
            
            for key, expansion in expansion_map.items():
                if key in processed:
                    processed = expansion
                    break
        
        return processed
        
    except Exception as e:
        logger.warning(f"查询预处理失败: {e}")
        return query

def _execute_vector_search(query: str, query_analysis: dict, user_ctx: UserContext) -> list:
    """执行向量搜索 - 优化版本"""
    try:
        if not vector_manager:
            logger.error("向量数据库未初始化")
            return []

        # 智能搜索参数调整
        confidence = query_analysis.get('confidence', 0.0)
        primary_intent = query_analysis.get('primary_intent', 'general')
        
        # 根据意图和置信度调整搜索参数
        if confidence > 0.7:
            search_k = 8  # 高置信度，多检索
        elif confidence > 0.3:
            search_k = 5  # 中等置信度，标准检索
        else:
            search_k = 3  # 低置信度，少检索

        # 构建智能过滤条件
        filter_metadata = _build_search_filters(primary_intent, user_ctx)

        # 执行向量搜索
        results = vector_manager.search_documents(
            query=query,
            k=search_k,
            filter_metadata=filter_metadata
        )
        
        if results is None:
            results = []
            
        logger.info(f"向量搜索完成: 找到 {len(results)} 个相关文档片段")
        return results
        
    except Exception as e:
        logger.error(f"向量搜索失败: {e}")
        return []

def _build_search_filters(primary_intent: str, user_ctx: UserContext) -> dict:
    """构建搜索过滤条件"""
    filter_metadata = {}
    
    # 暂时禁用分类过滤，因为上传的文档可能使用不同的分类
    # 根据意图设置分类过滤
    # intent_category_map = {
    #     'policy': '政策制度',
    #     'process': '流程指南', 
    #     'benefit': '福利待遇',
    #     'attendance': '考勤管理',
    #     'salary': '薪酬管理',
    #     'training': '培训资料',
    #     'onboarding': '入职指南',
    #     'offboarding': '离职指南'
    # }
    # 
    # if primary_intent in intent_category_map:
    #     filter_metadata['category'] = intent_category_map[primary_intent]
    
    # 根据用户角色设置访问级别过滤
    role_access_map = {
        'employee': ['public', 'internal'],
        'hr_staff': ['public', 'internal', 'confidential'],
        'hr_manager': ['public', 'internal', 'confidential', 'restricted'],
        'hr_director': ['public', 'internal', 'confidential', 'restricted', 'secret']
    }
    
    user_role = user_ctx.user_role
    if user_role in role_access_map:
        # 注意：这里需要向量数据库支持 $in 操作符
        # 如果不支持，可以去掉这个过滤条件
        pass  # 暂时不使用访问级别过滤，避免兼容性问题
    
    return filter_metadata if filter_metadata else None

def _build_user_context(user_ctx: UserContext) -> dict:
    """构建用户上下文字典"""
    return {
        'user_role': user_ctx.user_role,
        'department': user_ctx.department,
        'user_id': user_ctx.user_id
    }

def _filter_relevant_documents_for_stream(documents, intent_analysis, retrieval_strategy, user_ctx):
    """过滤流式查询的文档相关性
    
    检查文档是否与查询意图匹配：
    1. **超级管理员跳过过滤，可以访问所有文档**
    2. 如果检测到的部门为null或"通用"，说明查询与公司业务无关，不使用文档
    3. 如果用户查询特定部门的内容，检查文档是否来自该部门或相关
    4. 使用文档标题和部门信息进行匹配
    """
    # 特殊处理：超级管理员可以访问所有文档，跳过过滤
    if user_ctx and user_ctx.user_role == "super_admin":
        logger.info(
            f"用户是超级管理员，跳过文档相关性过滤，返回所有 {len(documents)} 个文档"
        )
        return documents
    
    if not intent_analysis:
        # 没有意图分析，返回所有文档
        return documents
    
    detected_dept = intent_analysis.detected_department
    keywords = intent_analysis.keywords if hasattr(intent_analysis, 'keywords') else []
    confidence = intent_analysis.confidence if hasattr(intent_analysis, 'confidence') else 0.0
    
    # 特殊情况1：如果检测到的部门为null，说明查询与公司业务无关
    if not detected_dept or detected_dept == "null":
        logger.info(
            f"检测到与公司业务无关的查询（部门: {detected_dept}），不使用公司文档，切换到通用知识回答"
        )
        return []
    
    # 特殊情况2：如果检测到的部门是"通用"、"其他"、"general"等，说明查询与公司业务无关
    if detected_dept.lower() in ['通用', '其他', 'general', 'other', '公共']:
        logger.info(
            f"检测到通用查询（部门: {detected_dept}），不使用公司文档，切换到通用知识回答"
        )
        return []
    
    # 特殊情况3：如果置信度很低（<0.3），说明查询意图不明确，可能与公司业务无关
    if confidence < 0.3:
        logger.info(
            f"意图置信度过低（{confidence:.2f}），可能与公司业务无关，不使用公司文档"
        )
        return []
    
    logger.info(
        f"过滤文档相关性: 检测部门={detected_dept}, "
        f"置信度={confidence:.2f}, 关键词={keywords}, 文档数={len(documents)}"
    )
    
    relevant_docs = []
    for doc in documents:
        # 检查1：文档部门是否匹配
        if doc.department == detected_dept:
            relevant_docs.append(doc)
            logger.debug(f"✓ 文档相关（部门匹配）: {doc.title} ({doc.department})")
            continue
        
        # 检查2：文档标题是否包含检测到的部门名称
        if detected_dept in doc.title:
            relevant_docs.append(doc)
            logger.debug(f"✓ 文档相关（标题包含部门）: {doc.title}")
            continue
        
        # 检查3：文档标题是否包含查询关键词
        if keywords:
            title_lower = doc.title.lower()
            if any(keyword.lower() in title_lower for keyword in keywords):
                relevant_docs.append(doc)
                logger.debug(f"✓ 文档相关（标题包含关键词）: {doc.title}")
                continue
        
        # 文档不相关
        logger.debug(
            f"✗ 文档不相关: {doc.title} (部门: {doc.department}, "
            f"检测部门: {detected_dept})"
        )
    
    logger.info(
        f"文档相关性过滤完成: {len(documents)} -> {len(relevant_docs)}"
    )
    
    return relevant_docs

def _handle_empty_results(query: str, start_time: float) -> QueryResponse:
    """处理空结果的情况"""
    # 提供智能建议
    suggestions = _generate_query_suggestions(query)
    
    answer = f"""抱歉，没有找到与「{query}」直接相关的人事政策信息。

【建议尝试】
{chr(10).join(f'• {suggestion}' for suggestion in suggestions)}

【其他方式】
• 联系人事部门获取最新政策信息
• 查阅公司内部人事管理系统
• 尝试使用更具体的关键词重新查询"""

    return QueryResponse(
        answer=answer,
        source_data=[],
        confidence=0.0,
        processing_time=time.time() - start_time
    )

def _generate_query_suggestions(query: str) -> list:
    """生成查询建议"""
    suggestions = []
    
    # 基于常见人事查询的建议
    common_queries = {
        '薪资': ['薪资发放时间', '薪资构成说明', '薪资调整政策'],
        '请假': ['年休假申请流程', '病假政策', '事假规定'],
        '考勤': ['考勤制度', '打卡要求', '迟到早退处理'],
        '培训': ['培训计划', '培训报名', '培训证书'],
        '入职': ['新员工入职流程', '入职材料清单', '试用期政策'],
        '离职': ['离职手续办理', '离职证明', '工作交接']
    }
    
    for keyword, related_queries in common_queries.items():
        if keyword in query:
            suggestions.extend(related_queries[:2])  # 每个类别最多2个建议
            break
    
    if not suggestions:
        suggestions = ['尝试使用更具体的关键词', '查询具体的政策名称', '描述具体的业务场景']
    
    return suggestions[:3]  # 最多3个建议

def _build_response_data(answer: str, vector_results: list, query_analysis: dict, 
                        original_query: str, processed_query: str, start_time: float,
                        evaluation: dict = None) -> QueryResponse:
    """构建优化的响应数据，包含明确的来源标识"""
    strategy = query_analysis.get('strategy', 'unknown')
    
    # 确定信息来源类型
    if strategy == 'document_based' and vector_results:
        source_type = "company_documents"  # 公司上传的文档
        source_description = "已上传的公司文档"
    else:
        source_type = "general_knowledge"  # AI通用知识
        source_description = "AI通用知识（非公司文档）"
    
    # 构建详细的来源数据
    source_data = [{
        "tool": "hr_knowledge_search",
        "source_type": source_type,  # 新增：明确的来源类型
        "source_description": source_description,  # 新增：来源描述
        "original_query": original_query,
        "processed_query": processed_query,
        "strategy": strategy,
        "evaluation": evaluation or {},
        "search_results": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                "relevance_score": doc.metadata.get('score', 0.0) if hasattr(doc, 'metadata') else 0.0,
                "document_id": doc.metadata.get('document_id', '') if hasattr(doc, 'metadata') else '',
                "title": doc.metadata.get('title', '') if hasattr(doc, 'metadata') else ''
            } for doc in vector_results
        ],
        "result_count": len(vector_results),
        "document_titles": list(set([
            doc.metadata.get('title', '') 
            for doc in vector_results 
            if hasattr(doc, 'metadata') and doc.metadata.get('title')
        ]))  # 新增：参考的文档标题列表
    }]

    # 优化的置信度计算
    if evaluation:
        confidence = evaluation.get('max_score', 0.5)
    else:
        confidence = _calculate_response_confidence(vector_results, query_analysis)

    return QueryResponse(
        answer=answer,
        source_data=source_data,
        confidence=confidence,
        processing_time=time.time() - start_time
    )



def _calculate_response_confidence(vector_results: list, query_analysis: dict) -> float:
    """计算响应置信度"""
    try:
        # 基础置信度：基于检索到的文档数量
        result_count = len(vector_results)
        base_confidence = min(result_count / 5.0, 1.0)
        
        # 意图分析置信度
        intent_confidence = query_analysis.get('confidence', 0.0)
        
        # 文档相关性置信度
        if vector_results:
            scores = [getattr(doc, 'score', 0.0) for doc in vector_results]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            relevance_confidence = min(avg_score, 1.0)
        else:
            relevance_confidence = 0.0
        
        # 综合置信度计算
        confidence = (base_confidence * 0.4 + intent_confidence * 0.3 + relevance_confidence * 0.3)
        
        return round(confidence, 3)
        
    except Exception as e:
        logger.warning(f"置信度计算失败: {e}")
        return 0.5  # 默认中等置信度

def _handle_query_error(error: Exception, start_time: float) -> QueryResponse:
    """处理查询错误"""
    error_msg = str(error)
    
    # 根据错误类型提供不同的用户友好消息
    if "timeout" in error_msg.lower():
        user_message = "查询超时，请稍后再试或简化查询内容。"
    elif "connection" in error_msg.lower():
        user_message = "系统连接异常，请稍后再试。"
    elif "memory" in error_msg.lower():
        user_message = "系统资源不足，请稍后再试。"
    else:
        user_message = "系统处理出现问题，请稍后再试或联系技术支持。"
    
    return QueryResponse(
        answer=user_message,
        source_data=[{
            "tool": "error_handler",
            "error_type": type(error).__name__,
            "error_message": error_msg,
            "timestamp": datetime.now().isoformat()
        }],
        confidence=0.0,
        processing_time=time.time() - start_time
    )

def service_upload_document(file, title: str, category: str, access_level: str, user_ctx: str) -> DocumentUploadResponse:
    """处理文档上传请求 - 优化版本"""
    import shutil
    from pathlib import Path
    from config import UPLOAD_DIR
    
    try:
        logger.info(f"收到文档上传: {file.filename} (标题: {title})")

        if vector_manager is None:
            logger.error('向量存储管理器未初始化')
            raise HTTPException(status_code=500, detail="向量数据库未初始化")

        user_context = json.loads(user_ctx)
        
        # 添加日志：查看前端传递的user_context内容
        logger.info(f"上传文档 - 前端传递的user_context: {user_context}")
        logger.info(f"上传文档 - 目标部门: {user_context.get('department', 'HR')}")

        # 检查标题是否重复
        if vector_manager.check_duplicate_title(title):
            raise HTTPException(status_code=400, detail=f"文档标题 '{title}' 已存在，请使用不同的标题")

        # 1. 保存文件到本地
        upload_dir = Path(UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)  # 如果目录不存在则创建
        
        # 生成唯一文件名（时间戳 + 原文件名）
        timestamp = int(time.time())
        file_extension = Path(file.filename).suffix
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = upload_dir / safe_filename
        
        # 保存文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"文件已保存到: {file_path}")
        
        # 重置文件指针，以便后续处理
        file.file.seek(0)

        # 2. 处理文档并存入向量数据库
        try:
            document_chunks = process_upload_file(file)
        except Exception as e:
            logger.error(f"文档处理失败: {str(e)}", exc_info=True)
            # 如果处理失败，删除已保存的文件
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=400, detail=f"文档处理失败: {str(e)}")

        # 生成文档ID
        doc_id_base = f"hr_doc_{timestamp}"
        document_id = doc_id_base

        # 优化的元数据结构，适配人事场景
        metadatas = [{
            "document_id": document_id,
            "doc_type": "uploaded",
            "access_level": access_level,
            "title": title,
            "category": category,
            "source_file": file.filename,
            "file_path": str(file_path),  # 保存文件路径
            "department": user_context.get("department", "HR"),
            "uploader": user_context.get("user_role", "hr_staff"),
            "upload_time": datetime.now().isoformat(),
            "chunk_index": i,
            "total_chunks": len(document_chunks)
        } for i, _ in enumerate(document_chunks)]

        chunk_ids = [f"{doc_id_base}_chunk_{i}" for i in range(len(document_chunks))]

        # 使用优化的向量存储管理器
        success = vector_manager.add_document(
            texts=document_chunks,
            metadatas=metadatas,
            ids=chunk_ids
        )

        if not success:
            # 如果向量数据库添加失败，删除已保存的文件
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=500, detail="文档添加到向量数据库失败")

        logger.info(f"成功添加 {len(document_chunks)} 个文档块到向量数据库")

        return DocumentUploadResponse(
            status="success",
            document_id=document_id,
            filename=file.filename,
            chunks=len(document_chunks),
            first_chunk=document_chunks[0][:100] + "..." if document_chunks else "",
            message=f"文档 '{title}' 上传成功，已分割为 {len(document_chunks)} 个文档块"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档上传失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")

def service_delete_document(document_id: str) -> dict:
    """删除文档"""
    try:
        if vector_manager is None:
            raise HTTPException(status_code=500, detail="向量数据库未初始化")

        success = vector_manager.delete_document(document_id)
        
        if success:
            return {
                "status": "success",
                "message": f"文档 {document_id} 删除成功"
            }
        else:
            raise HTTPException(status_code=404, detail=f"文档 {document_id} 不存在")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")

def service_update_document(document_id: str, file, title: str, category: str, access_level: str, user_ctx: str) -> DocumentUploadResponse:
    """更新文档"""
    try:
        if vector_manager is None:
            raise HTTPException(status_code=500, detail="向量数据库未初始化")

        user_context = json.loads(user_ctx)

        # 检查标题是否与其他文档重复（排除当前文档）
        if vector_manager.check_duplicate_title(title, exclude_doc_id=document_id):
            raise HTTPException(status_code=400, detail=f"文档标题 '{title}' 已存在，请使用不同的标题")

        try:
            document_chunks = process_upload_file(file)
        except Exception as e:
            logger.error(f"文档处理失败: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"文档处理失败: {str(e)}")

        # 使用原有的文档ID
        metadatas = [{
            "document_id": document_id,
            "doc_type": "uploaded",
            "access_level": access_level,
            "title": title,
            "category": category,
            "source_file": file.filename,
            "department": user_context.get("department", "HR"),
            "uploader": user_context.get("user_role", "hr_staff"),
            "upload_time": datetime.now().isoformat(),
            "chunk_index": i,
            "total_chunks": len(document_chunks)
        } for i, _ in enumerate(document_chunks)]

        chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(document_chunks))]

        success = vector_manager.update_document(
            document_id=document_id,
            texts=document_chunks,
            metadatas=metadatas,
            ids=chunk_ids
        )

        if not success:
            raise HTTPException(status_code=500, detail="文档更新失败")

        return DocumentUploadResponse(
            status="success",
            document_id=document_id,
            filename=file.filename,
            chunks=len(document_chunks),
            first_chunk=document_chunks[0][:100] + "..." if document_chunks else "",
            message=f"文档 '{title}' 更新成功，已分割为 {len(document_chunks)} 个文档块"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新文档失败: {str(e)}")

def service_list_documents(limit: Optional[int] = None) -> List[DocumentInfo]:
    """获取文档列表"""
    try:
        if vector_manager is None:
            raise HTTPException(status_code=500, detail="向量数据库未初始化")

        documents = vector_manager.list_documents(limit=limit)
        
        # 转换为DocumentInfo格式
        doc_infos = []
        for doc in documents:
            # 安全地转换枚举类型
            category_str = doc.get('category', '其他')
            access_level_str = doc.get('access_level', '全员')
            
            # 查找匹配的枚举值
            category = DocumentCategory.OTHER
            for cat in DocumentCategory:
                if cat.value == category_str:
                    category = cat
                    break
            
            access_level = AccessLevel.PUBLIC
            for level in AccessLevel:
                if level.value == access_level_str:
                    access_level = level
                    break
            
            doc_info = DocumentInfo(
                id=doc.get('document_id', ''),
                title=doc.get('title', ''),
                category=category,
                access_level=access_level,
                filename=doc.get('source_file', ''),
                upload_time=datetime.fromisoformat(doc.get('upload_time', datetime.now().isoformat())),
                uploader=doc.get('uploader', ''),
                chunks_count=doc.get('total_chunks', 0),
                department=doc.get('department', '')
            )
            doc_infos.append(doc_info)
        
        return doc_infos
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")

def service_search_documents(query: str, category: Optional[str] = None, access_level: Optional[str] = None, k: int = 5) -> List[dict]:
    """搜索文档"""
    try:
        if vector_manager is None:
            raise HTTPException(status_code=500, detail="向量数据库未初始化")

        # 构建过滤条件
        filter_metadata = {}
        if category:
            filter_metadata['category'] = category
        if access_level:
            filter_metadata['access_level'] = access_level

        results = vector_manager.search_documents(
            query=query,
            k=k,
            filter_metadata=filter_metadata if filter_metadata else None
        )

        # 格式化搜索结果
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                "score": getattr(doc, 'score', 0.0)
            })

        return formatted_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"搜索文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"搜索文档失败: {str(e)}")

def service_health_check() -> HealthResponse:
    """系统健康检查"""
    # 检查各组件状态
    vector_db_status = "active" if vector_store else "inactive"
    
    # 如果关键组件不可用，系统状态为警告
    system_status = "ok" if vector_store else "warning"
    
    detail = {
        "version": "2.0",
        "mode": "api" if vector_store else "simplified",
        "components": {
            "vector_db": vector_db_status,
            "llm_integration": "alibaba_cloud",
            "api_server": "active",
            "query_router": "active" if legacy_query_router else "inactive"
        }
    }
    
    # 如果是简化模式，添加说明
    if not vector_store:
        detail["notice"] = "系统运行在简化模式，向量搜索功能不可用。"
    
    return HealthResponse(
        status=system_status,
        service="人事知识库系统",
        detail=detail
    )

def service_vector_status() -> VectorStoreStatus:
    """获取向量存储状态 - 优化版本"""
    try:
        if vector_manager:
            stats = vector_manager.get_collection_stats()
            
            return VectorStoreStatus(
                status="active",
                documents=stats.get('total_documents', 0),
                collection_name=stats.get('collection_name', 'hr_knowledge'),
                last_updated=datetime.fromisoformat(stats.get('last_updated', datetime.now().isoformat()))
            )
        else:
            return VectorStoreStatus(
                status="not_initialized",
                documents=0,
                collection_name="hr_knowledge"
            )
    except Exception as e:
        logger.error(f"向量数据库状态检查失败: {str(e)}")
        return VectorStoreStatus(
            status="error",
            documents=0,
            collection_name="hr_knowledge"
        )

def service_get_collection_stats() -> dict:
    """获取向量存储详细统计信息"""
    try:
        if vector_manager is None:
            raise HTTPException(status_code=500, detail="向量数据库未初始化")

        stats = vector_manager.get_collection_stats()
        return {
            "status": "success",
            "data": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

async def service_query_knowledge_stream(request: QueryRequest) -> AsyncGenerator[dict, None]:
    """处理人事知识查询请求 - 流式版本（使用智能多部门路由器 + 对话历史）"""
    start_time = time.time()
    logger.info(f"收到流式查询: {request.question}, session_id={request.session_id}")

    try:
        # 优先使用新的智能路由器
        if intelligent_router:
            logger.info("使用智能多部门路由器处理流式查询")
            
            # dev-mix新增：获取对话历史
            history_manager = get_history_manager()
            chat_history = None
            if request.session_id:
                chat_history = history_manager.get_history(request.session_id)
                logger.info(f"会话 {request.session_id} 历史消息数: {len(chat_history.messages)}")
            
            # 1. 发送查询预处理状态
            yield {
                'type': 'status',
                'stage': 'preprocessing',
                'message': '正在预处理查询...',
                'progress': 10
            }
            
            # 2. 发送智能路由分析状态
            yield {
                'type': 'status',
                'stage': 'intelligent_routing',
                'message': '正在进行智能多部门路由分析...',
                'progress': 30
            }

            # 使用智能路由器处理查询
            query_result = intelligent_router.route_query(
                query=request.question,
                user_id=request.user_ctx.user_id,
                session_id=request.session_id
            )
            
            logger.info(f"智能路由完成: 策略={query_result.source_type}, 文档数={len(query_result.documents)}")
            
            # 3. 发送LLM生成开始状态
            yield {
                'type': 'status',
                'stage': 'llm_generation',
                'message': f'正在生成回答（策略: {query_result.source_type}）...',
                'progress': 50
            }

            # 4. 流式生成LLM回答（基于智能路由结果）
            logger.info("开始基于智能路由结果的流式LLM回答生成...")
            
            # 过滤文档相关性
            docs_to_use = []
            if query_result.source_type == "document_based" and query_result.documents:
                # 检查文档是否与查询意图相关
                intent_analysis = query_result.intent_analysis
                retrieval_strategy = query_result.retrieval_strategy
                
                relevant_docs = _filter_relevant_documents_for_stream(
                    query_result.documents,
                    intent_analysis,
                    retrieval_strategy,
                    request.user_ctx
                )
                
                if not relevant_docs:
                    logger.warning(
                        f"检索到{len(query_result.documents)}个文档，但都与查询意图不相关。"
                        f"检测部门: {intent_analysis.detected_department}"
                    )
                    
                    # 判断是否是权限问题
                    if (intent_analysis.detected_department and 
                        retrieval_strategy and 
                        not retrieval_strategy.has_permission):
                        # 用户查询的是无权限部门的内容
                        permission_denied_answer = f"""【权限提示】

您查询的内容涉及「{intent_analysis.detected_department}」部门，但您当前无权访问该部门的文档。

【您的权限】
- 当前部门：{request.user_ctx.department}

【建议操作】
1. 如需查看{intent_analysis.detected_department}部门的文档，请联系{intent_analysis.detected_department}部门负责人申请权限
2. 您可以在公共文件夹中查找相关信息
3. 联系系统管理员了解权限申请流程

【温馨提示】
公司文档按部门进行权限管理，以保护敏感信息。如有业务需要，请通过正规流程申请访问权限。

⚠️ 免责声明：以上回答由AI基于文档内容分析生成，仅供参考。如有疑问或需要准确信息，请与公司相关部门负责人确认。"""
                        
                        yield {
                            'type': 'content',
                            'content': permission_denied_answer,
                            'is_complete': True,
                            'progress': 90
                        }
                        
                        # 发送完成状态
                        processing_time = time.time() - start_time
                        yield {
                            'type': 'complete',
                            'message': '权限检查完成',
                            'progress': 100,
                            'full_answer': permission_denied_answer,
                            'confidence': 1.0,
                            'processing_time': processing_time,
                            'source_count': 0,
                            'strategy': 'permission_denied',
                            'user_department': request.user_ctx.department,
                            'intent_analysis': {
                                'primary_intent': intent_analysis.primary_intent,
                                'confidence': intent_analysis.confidence,
                                'detected_department': intent_analysis.detected_department
                            }
                        }
                        
                        logger.info("权限拒绝回答已发送")
                        return
                    else:
                        # 其他情况：没有找到相关文档，使用通用知识回答
                        logger.info("未找到相关文档，切换到通用知识回答")
                        docs_to_use = []
                else:
                    # 将相关文档转换为兼容格式
                    for doc in relevant_docs:
                        class SimpleDoc:
                            def __init__(self, content, metadata):
                                self.page_content = content
                                self.metadata = metadata
                        
                        docs_to_use.append(SimpleDoc(
                            content=doc.content,
                            metadata={
                                'title': doc.title,
                                'department': doc.department,
                                'score': doc.score,
                                'document_id': doc.document_id
                            }
                        ))
                    
                    logger.info(f"准备使用 {len(docs_to_use)} 个相关文档进行流式生成")
            else:
                logger.info(f"无文档，使用通用知识回答（策略: {query_result.source_type}）")
            
            user_ctx_dict = _build_user_context(request.user_ctx)
            
            # 流式生成答案
            full_answer = ""
            chunk_count = 0
            async for chunk in _generate_streaming_response(docs_to_use, request.question, user_ctx_dict, chat_history):
                full_answer += chunk
                chunk_count += 1
                logger.info(f"[Stream] Chunk #{chunk_count}, length: {len(chunk)}, total: {len(full_answer)}")
                yield {
                    'type': 'content',
                    'content': chunk,
                    'is_complete': False,
                    'progress': min(90, 50 + (len(full_answer) / 10))
                }
            
            logger.info("智能路由流式LLM回答生成完成")
            
            # dev-mix新增：更新对话历史
            if request.session_id and chat_history:
                history_manager.add_user_message(request.session_id, request.question)
                history_manager.add_ai_message(request.session_id, full_answer)
                logger.info(f"已更新会话 {request.session_id} 的对话历史")

            # 5. 发送完成状态
            processing_time = time.time() - start_time
            
            yield {
                'type': 'complete',
                'message': '智能多部门路由查询完成',
                'progress': 100,
                'full_answer': full_answer,
                'confidence': query_result.confidence,
                'processing_time': processing_time,
                'source_count': len(query_result.documents),
                'strategy': query_result.source_type,
                'user_department': query_result.user_context.department,
                'intent_analysis': {
                    'primary_intent': query_result.intent_analysis.primary_intent,
                    'confidence': query_result.intent_analysis.confidence,
                    'detected_department': query_result.intent_analysis.detected_department
                }
            }

            logger.info("智能多部门流式查询成功完成")
            return
        
        # 降级到传统路由器处理流式查询
        elif legacy_query_router and vector_store:
            logger.warning("智能路由器不可用，降级到传统流式路由器")
            
            # dev-mix新增：获取对话历史
            history_manager = get_history_manager()
            chat_history = None
            if request.session_id:
                chat_history = history_manager.get_history(request.session_id)
                logger.info(f"会话 {request.session_id} 历史消息数: {len(chat_history.messages)}")
            
            # 1. 发送查询预处理状态
            yield {
                'type': 'status',
                'stage': 'preprocessing',
                'message': '正在预处理查询...',
                'progress': 10
            }
            
            # 查询预处理
            processed_query = _preprocess_query(request.question)
            logger.info(f"查询预处理完成: {processed_query}")
            
            # 2. 发送路由分析状态
            yield {
                'type': 'status',
                'stage': 'routing',
                'message': '正在分析查询并路由...',
                'progress': 30
            }

            # 使用传统查询路由器
            user_ctx_dict = _build_user_context(request.user_ctx)
            route_result = legacy_query_router.route(processed_query, vector_store, user_ctx_dict)
            
            strategy = route_result['strategy']
            documents = route_result['documents']
            evaluation = route_result['evaluation']
            
            logger.info(f"传统路由策略: {strategy}, 文档数: {len(documents)}")
            
            # 3. 发送LLM生成开始状态
            yield {
                'type': 'status',
                'stage': 'llm_generation',
                'message': f'正在生成回答（策略: {strategy}）...',
                'progress': 50
            }

            # 4. 流式生成LLM回答（dev-mix：传入对话历史）
            logger.info("开始传统流式LLM回答生成...")
            
            # 根据策略决定是否传递文档
            docs_to_use = documents if strategy == 'document_based' else []
            logger.info(f"传递给LLM的文档数: {len(docs_to_use)} (策略: {strategy})")
            
            full_answer = ""
            chunk_count = 0
            async for chunk in _generate_streaming_response(docs_to_use, processed_query, user_ctx_dict, chat_history):
                full_answer += chunk
                chunk_count += 1
                logger.info(f"[Stream] Chunk #{chunk_count}, length: {len(chunk)}, total: {len(full_answer)}")
                yield {
                    'type': 'content',
                    'content': chunk,
                    'is_complete': False,
                    'progress': min(90, 50 + (len(full_answer) / 10))
                }
            
            logger.info("传统流式LLM回答生成完成")
            
            # dev-mix新增：更新对话历史
            if request.session_id and chat_history:
                history_manager.add_user_message(request.session_id, request.question)
                history_manager.add_ai_message(request.session_id, full_answer)
                logger.info(f"已更新会话 {request.session_id} 的对话历史")

            # 5. 发送完成状态
            processing_time = time.time() - start_time
            confidence = evaluation.get('max_score', 0.5)
            
            yield {
                'type': 'complete',
                'message': '传统路由查询完成',
                'progress': 100,
                'full_answer': full_answer,
                'confidence': confidence,
                'processing_time': processing_time,
                'source_count': len(documents),
                'strategy': strategy
            }

            logger.info("传统流式查询成功完成")
        
        else:
            yield {
                'type': 'error',
                'message': '查询系统未正确初始化',
                'error_type': 'SystemError'
            }

    except Exception as e:
        logger.error(f"流式查询处理失败: {str(e)}")
        yield {
            'type': 'error',
            'message': f"查询处理失败: {str(e)}",
            'error_type': type(e).__name__
        }

async def _generate_streaming_response(vector_results: List, query: str, user_ctx: Dict, chat_history=None) -> AsyncGenerator[str, None]:
    """生成流式LLM响应 - 使用阿里云API的流式接口（dev-mix：支持对话历史）"""
    try:
        agent = get_hr_agent()
        if not agent.llm:
            yield "抱歉，AI助手暂时不可用，请稍后再试。"
            return
        
        # 判断是否有相关文档
        if vector_results:
            # 基于文档的回答 - 不在开头添加来源标识
            # 因为回答中的【文档依据】部分已经说明了来源
            context_docs = agent._format_context_documents(vector_results)
            
            # dev-mix新增：构建包含历史的提示词
            if chat_history and len(chat_history.messages) > 0:
                prompt = agent._build_enhanced_prompt_with_history(
                    query, context_docs, user_ctx, chat_history.messages
                )
                logger.info(f"使用对话历史构建提示词，历史消息数: {len(chat_history.messages)}")
            else:
                prompt = agent._build_enhanced_prompt(query, context_docs, user_ctx)
        else:
            # 通用知识回答 - 先输出来源标识
            yield "💡 **信息来源：AI通用知识（非公司文档）**\n\n"
            
            # dev-mix新增：通用知识回答也支持历史
            if chat_history and len(chat_history.messages) > 0:
                history_text = "\n".join([
                    f"{'用户' if i % 2 == 0 else 'AI'}: {msg.content}"
                    for i, msg in enumerate(chat_history.messages[-4:])  # 最近2轮对话
                ])
                
                prompt = f"""你是一个专业的企业知识库助手。

对话历史：
{history_text}

当前问题：{query}

请基于你的通用知识回答这个问题。如果这个问题与企业管理相关，请提供专业的建议。

注意：
1. 不要在回答中重复添加"信息来源"标识，因为已经在前面添加过了
2. 不要添加免责声明，系统会自动添加"""
            else:
                prompt = f"""你是一个专业的企业知识库助手。

用户问题：{query}

请基于你的通用知识回答这个问题。如果这个问题与企业管理相关，请提供专业的建议。

注意：
1. 不要在回答中重复添加"信息来源"标识，因为已经在前面添加过了
2. 不要添加免责声明，系统会自动添加"""
        
        # 使用阿里云API的流式生成
        try:
            logger.info("开始使用流式API生成回答...")
            
            # 使用 astream 进行异步流式生成
            async for chunk in agent.llm.astream(prompt):
                if chunk.content:
                    # 过滤掉LLM可能重复添加的来源标识
                    content = chunk.content
                    if "信息来源" not in content or content.index("信息来源") > 10:
                        # 直接输出每个 chunk，实现真正的流式效果
                        yield content
                    
        except Exception as e:
            error_msg = str(e)
            logger.error(f"LLM流式生成失败: {error_msg}")
            
            # 检查是否是内容审核错误
            if "inappropriate content" in error_msg.lower() or "inappropriate-content" in error_msg.lower():
                yield f"\n\n⚠️ 抱歉，由于内容安全审核限制，无法完整生成回答。这可能是因为某些词汇触发了安全机制。\n\n建议：\n• 尝试换一种方式提问\n• 使用更具体的关键词\n• 如需帮助，请联系系统管理员"
            else:
                yield f"\n\n抱歉，生成回答时出现问题: {error_msg}"
            
    except Exception as e:
        logger.error(f"流式响应生成失败: {str(e)}")
        yield f"系统错误: {str(e)}"