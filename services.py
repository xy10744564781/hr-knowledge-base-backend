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

# 初始化向量库（chroma）
try:
    # 设置环境变量解决protobuf版本冲突和禁用遥测
    import os
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    os.environ['ANONYMIZED_TELEMETRY'] = 'False'
    os.environ['CHROMA_TELEMETRY'] = 'False'
    
    chroma_global = init_chroma()
    vector_manager = get_vector_manager()
    
    # 初始化响应生成器
    hr_agent = get_hr_agent()
    
    if chroma_global:
        logger.info('ChromaDB向量库初始化成功')
    else:
        logger.warning('ChromaDB向量库初始化失败，系统将以简化模式运行')
        
except Exception as e:
    logger.error(f'向量库初始化过程中发生错误: {str(e)}', exc_info=True)
    chroma_global = None
    vector_manager = None
    
    # 简化模式下的响应生成器
    hr_agent = get_hr_agent()

def service_query_knowledge(request: QueryRequest) -> QueryResponse:
    """处理人事知识查询请求 - 简化版本"""
    start_time = time.time()
    logger.info(f"收到人事知识查询: {request.question}")

    try:
        # 查询预处理
        processed_query = _preprocess_query(request.question)
        logger.info(f"查询预处理完成: {processed_query}")

        # 查询意图分析
        query_analysis = analyze_query(processed_query)
        logger.info(f"查询意图分析: {query_analysis}")

        # 向量数据库检索
        vector_results = _execute_vector_search(processed_query, query_analysis, request.user_ctx)
        
        if not vector_results:
            return _handle_empty_results(request.question, start_time)

        # 直接调用LLM生成回答
        user_ctx_dict = _build_user_context(request.user_ctx)
        logger.info("开始LLM回答生成...")
        
        answer = integrate_results(vector_results, [], processed_query, user_ctx_dict)
        
        logger.info("LLM回答生成完成")

        # 构建响应数据
        response_data = _build_response_data(
            answer=answer,
            vector_results=vector_results,
            query_analysis=query_analysis,
            original_query=request.question,
            processed_query=processed_query,
            start_time=start_time
        )

        logger.info("人事知识查询成功完成")
        return response_data

    except Exception as e:
        logger.error(f"查询处理失败: {str(e)}")
        return _handle_query_error(e, start_time)

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
                        original_query: str, processed_query: str, start_time: float) -> QueryResponse:
    """构建优化的响应数据"""
    # 构建详细的来源数据
    source_data = [{
        "tool": "hr_knowledge_search",
        "original_query": original_query,
        "processed_query": processed_query,
        "intent_analysis": query_analysis,
        "search_results": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                "relevance_score": getattr(doc, 'score', 0.0),
                "document_id": doc.metadata.get('document_id', '') if hasattr(doc, 'metadata') else '',
                "title": doc.metadata.get('title', '') if hasattr(doc, 'metadata') else ''
            } for doc in vector_results
        ],
        "result_count": len(vector_results)
    }]

    # 优化的置信度计算
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
    try:
        logger.info(f"收到文档上传: {file.filename} (标题: {title})")

        if vector_manager is None:
            logger.error('向量存储管理器未初始化')
            raise HTTPException(status_code=500, detail="向量数据库未初始化")

        user_context = json.loads(user_ctx)

        # 检查标题是否重复
        if vector_manager.check_duplicate_title(title):
            raise HTTPException(status_code=400, detail=f"文档标题 '{title}' 已存在，请使用不同的标题")

        try:
            document_chunks = process_upload_file(file)
        except Exception as e:
            logger.error(f"文档处理失败: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"文档处理失败: {str(e)}")

        # 生成文档ID
        doc_id_base = f"hr_doc_{int(time.time())}"
        document_id = doc_id_base

        # 优化的元数据结构，适配人事场景
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

        chunk_ids = [f"{doc_id_base}_chunk_{i}" for i in range(len(document_chunks))]

        # 使用优化的向量存储管理器
        success = vector_manager.add_document(
            texts=document_chunks,
            metadatas=metadatas,
            ids=chunk_ids
        )

        if not success:
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
                chunks_count=doc.get('total_chunks', 0)
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
    vector_db_status = "active" if chroma_global else "inactive"
    
    # 如果关键组件不可用，系统状态为警告
    system_status = "ok" if chroma_global else "warning"
    
    detail = {
        "version": "1.0",
        "mode": "full" if chroma_global else "simplified",
        "components": {
            "vector_db": vector_db_status,
            "llm_integration": "configured",
            "api_server": "active"
        }
    }
    
    # 如果是简化模式，添加说明
    if not chroma_global:
        detail["notice"] = "系统运行在简化模式，向量搜索功能不可用。建议升级SQLite版本或使用main_simple.py"
    
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
    """处理人事知识查询请求 - 流式版本"""
    start_time = time.time()
    logger.info(f"收到流式查询: {request.question}")

    try:
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
        
        # 2. 发送意图分析状态
        yield {
            'type': 'status',
            'stage': 'intent_analysis',
            'message': '正在分析查询意图...',
            'progress': 20
        }

        # 查询意图分析
        query_analysis = analyze_query(processed_query)
        logger.info(f"查询意图分析: {query_analysis}")

        # 3. 发送向量检索状态
        yield {
            'type': 'status',
            'stage': 'vector_search',
            'message': '正在搜索相关文档...',
            'progress': 30
        }

        # 向量数据库检索
        vector_results = _execute_vector_search(processed_query, query_analysis, request.user_ctx)
        
        if not vector_results:
            yield {
                'type': 'status',
                'stage': 'no_results',
                'message': '未找到相关文档，生成通用回答...',
                'progress': 40
            }
            
            # 处理空结果
            empty_response = _handle_empty_results(request.question, start_time)
            yield {
                'type': 'content',
                'content': empty_response.answer,
                'is_complete': True,
                'confidence': empty_response.confidence,
                'processing_time': empty_response.processing_time
            }
            return

        # 4. 发送LLM生成开始状态
        yield {
            'type': 'status',
            'stage': 'llm_generation',
            'message': '正在生成回答...',
            'progress': 50
        }

        # 5. 流式生成LLM回答
        user_ctx_dict = _build_user_context(request.user_ctx)
        logger.info("开始流式LLM回答生成...")
        
        # 使用流式生成
        full_answer = ""
        async for chunk in _generate_streaming_response(vector_results, processed_query, user_ctx_dict):
            full_answer += chunk
            yield {
                'type': 'content',
                'content': chunk,
                'is_complete': False,
                'progress': min(90, 50 + (len(full_answer) / 10))  # 动态进度
            }
        
        logger.info("流式LLM回答生成完成")

        # 6. 发送完成状态
        processing_time = time.time() - start_time
        confidence = _calculate_response_confidence(vector_results, query_analysis)
        
        yield {
            'type': 'complete',
            'message': '回答生成完成',
            'progress': 100,
            'full_answer': full_answer,
            'confidence': confidence,
            'processing_time': processing_time,
            'source_count': len(vector_results)
        }

        logger.info("流式人事知识查询成功完成")

    except Exception as e:
        logger.error(f"流式查询处理失败: {str(e)}")
        yield {
            'type': 'error',
            'message': f"查询处理失败: {str(e)}",
            'error_type': type(e).__name__
        }

async def _generate_streaming_response(vector_results: List, query: str, user_ctx: Dict) -> AsyncGenerator[str, None]:
    """生成流式LLM响应"""
    try:
        agent = get_hr_agent()
        if not agent.llm:
            yield "抱歉，AI助手暂时不可用，请稍后再试。"
            return
        
        # 格式化上下文文档
        context_docs = agent._format_context_documents(vector_results)
        
        # 构建提示词
        prompt = agent._build_enhanced_prompt(query, context_docs, user_ctx)
        
        # 使用Ollama的流式API
        try:
            # 这里需要使用支持流式的LLM调用
            # 由于langchain_ollama可能不直接支持流式，我们需要模拟
            response = agent.llm.invoke(prompt)
            
            if response and response.content:
                # 解析思考过程和正式回答
                content = response.content.strip()
                
                # 检测是否包含思考过程（通常以特定模式开始）
                thinking_patterns = [
                    "好的，我现在需要",
                    "首先，用户提供的",
                    "根据用户提供的文档",
                    "我需要仔细理解",
                    "用户的问题可能涉及"
                ]
                
                thinking_content = ""
                formal_answer = ""
                
                # 查找思考过程的结束标志
                thinking_end_patterns = [
                    "【问题理解】",
                    "## 问题理解",
                    "# 问题理解",
                    "问题理解："
                ]
                
                has_thinking = any(pattern in content for pattern in thinking_patterns)
                
                if has_thinking:
                    # 寻找思考过程的结束位置
                    thinking_end_pos = -1
                    for pattern in thinking_end_patterns:
                        pos = content.find(pattern)
                        if pos != -1:
                            thinking_end_pos = pos
                            break
                    
                    if thinking_end_pos != -1:
                        thinking_content = content[:thinking_end_pos].strip()
                        formal_answer = content[thinking_end_pos:].strip()
                    else:
                        # 如果没找到结束标志，尝试按段落分割
                        paragraphs = content.split('\n\n')
                        if len(paragraphs) > 1:
                            # 第一段作为思考过程
                            thinking_content = paragraphs[0]
                            formal_answer = '\n\n'.join(paragraphs[1:])
                        else:
                            formal_answer = content
                else:
                    formal_answer = content
                
                # 先发送思考过程（如果有）
                if thinking_content:
                    yield f"<thinking>{thinking_content}</thinking>"
                    await asyncio.sleep(0.1)
                
                # 流式输出正式回答
                if formal_answer:
                    sentences = []
                    current_sentence = ""
                    
                    for char in formal_answer:
                        current_sentence += char
                        if char in ['。', '！', '？', '\n', '；', '：']:
                            if current_sentence.strip():
                                sentences.append(current_sentence)
                                current_sentence = ""
                    
                    # 添加剩余内容
                    if current_sentence.strip():
                        sentences.append(current_sentence)
                    
                    # 流式输出句子
                    for sentence in sentences:
                        yield sentence
                        # 添加小延迟模拟真实流式效果
                        await asyncio.sleep(0.1)
                else:
                    yield "抱歉，无法生成完整回答，请稍后再试。"
            else:
                yield "抱歉，无法生成回答，请稍后再试。"
                
        except Exception as e:
            logger.error(f"LLM流式生成失败: {str(e)}")
            yield f"生成回答时出现问题: {str(e)}"
            
    except Exception as e:
        logger.error(f"流式响应生成失败: {str(e)}")
        yield f"系统错误: {str(e)}"