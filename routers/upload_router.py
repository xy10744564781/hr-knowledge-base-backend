from fastapi import APIRouter, UploadFile, File, Form, Query, Path
from typing import List, Optional
from services import (
    service_upload_document, service_delete_document, service_update_document,
    service_list_documents, service_search_documents
)
from schemas import DocumentUploadResponse, DocumentInfo

router = APIRouter()

@router.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="要上传的文档文件"),
    title: str = Form(..., description="文档标题"),
    category: str = Form(..., description="文档分类"),
    access_level: str = Form(..., description="访问权限"),
    user_ctx: str = Form(..., description="用户上下文JSON字符串")
) -> DocumentUploadResponse:
    """
    上传人事文档到知识库
    
    - **file**: 支持PDF、Word、文本文件
    - **title**: 文档标题，用于标识文档
    - **category**: 文档分类（政策制度、流程指南、培训资料等）
    - **access_level**: 访问权限（public、internal、confidential、restricted）
    - **user_ctx**: 用户上下文信息
    """
    return service_upload_document(file, title, category, access_level, user_ctx)

@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents(
    limit: Optional[int] = Query(None, description="返回文档数量限制", ge=1, le=100)
) -> List[DocumentInfo]:
    """
    获取文档列表
    
    - **limit**: 可选，限制返回的文档数量（1-100）
    """
    return service_list_documents(limit)

@router.get("/documents/{document_id}")
async def get_document_info(
    document_id: str = Path(..., description="文档ID")
) -> dict:
    """
    获取指定文档的详细信息
    
    - **document_id**: 文档唯一标识符
    """
    documents = service_list_documents()
    for doc in documents:
        if doc.id == document_id:
            return {
                "status": "success",
                "document": doc.dict()
            }
    
    return {
        "status": "not_found",
        "message": f"文档 {document_id} 不存在"
    }

@router.put("/documents/{document_id}", response_model=DocumentUploadResponse)
async def update_document(
    document_id: str = Path(..., description="文档ID"),
    file: UploadFile = File(..., description="要更新的文档文件"),
    title: str = Form(..., description="文档标题"),
    category: str = Form(..., description="文档分类"),
    access_level: str = Form(..., description="访问权限"),
    user_ctx: str = Form(..., description="用户上下文JSON字符串")
) -> DocumentUploadResponse:
    """
    更新指定文档
    
    - **document_id**: 要更新的文档ID
    - **file**: 新的文档文件
    - **title**: 新的文档标题
    - **category**: 新的文档分类
    - **access_level**: 新的访问权限
    - **user_ctx**: 用户上下文信息
    """
    return service_update_document(document_id, file, title, category, access_level, user_ctx)

@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str = Path(..., description="文档ID")
) -> dict:
    """
    删除指定文档
    
    - **document_id**: 要删除的文档ID
    """
    return service_delete_document(document_id)

@router.get("/search-documents")
async def search_documents(
    query: str = Query(..., description="搜索查询"),
    category: Optional[str] = Query(None, description="文档分类过滤"),
    access_level: Optional[str] = Query(None, description="访问权限过滤"),
    k: int = Query(5, description="返回结果数量", ge=1, le=20)
) -> dict:
    """
    搜索文档
    
    - **query**: 搜索关键词
    - **category**: 可选，按分类过滤
    - **access_level**: 可选，按访问权限过滤
    - **k**: 返回结果数量（1-20）
    """
    results = service_search_documents(query, category, access_level, k)
    
    return {
        "status": "success",
        "query": query,
        "results": results,
        "count": len(results)
    }
