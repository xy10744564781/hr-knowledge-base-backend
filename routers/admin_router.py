from fastapi import APIRouter
from datetime import datetime
import psutil
import os
from services import service_health_check, service_vector_status, service_get_collection_stats
from schemas import HealthResponse, VectorStoreStatus

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    系统健康检查
    
    返回系统运行状态和组件信息
    """
    return service_health_check()

@router.get("/health/detailed")
async def detailed_health_check() -> dict:
    """
    详细系统健康检查
    
    返回详细的系统运行状态、资源使用情况和组件信息
    """
    try:
        # 基础健康检查
        basic_health = service_health_check()
        
        # 系统资源信息
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        # 向量数据库状态
        vector_status = service_vector_status()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": basic_health.service,
            "version": basic_health.detail.get("version", "1.0"),
            "system_info": {
                "memory": {
                    "total": f"{memory.total / (1024**3):.2f} GB",
                    "available": f"{memory.available / (1024**3):.2f} GB",
                    "used_percent": f"{memory.percent}%"
                },
                "disk": {
                    "total": f"{disk.total / (1024**3):.2f} GB",
                    "free": f"{disk.free / (1024**3):.2f} GB",
                    "used_percent": f"{(disk.used / disk.total) * 100:.1f}%"
                },
                "process_id": os.getpid()
            },
            "components": {
                "vector_db": {
                    "status": vector_status.status,
                    "documents": vector_status.documents,
                    "collection": vector_status.collection_name
                },
                "llm_integration": basic_health.detail.get("components", {}).get("llm_integration", "unknown")
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/vector-status", response_model=VectorStoreStatus)
async def vector_status() -> VectorStoreStatus:
    """
    向量数据库状态检查
    
    返回向量数据库的状态和文档统计信息
    """
    return service_vector_status()

@router.get("/vector-stats")
async def vector_statistics() -> dict:
    """
    向量数据库详细统计
    
    返回向量数据库的详细统计信息，包括文档分类、类型分布等
    """
    return service_get_collection_stats()

@router.get("/system-info")
async def system_information() -> dict:
    """
    系统信息查询
    
    返回系统配置信息和运行环境
    """
    try:
        from config import (
            APP_TITLE, APP_VERSION, CHROMA_DB_PATH, OLLAMA_MODEL,
            MAX_FILE_SIZE, SUPPORTED_FORMATS, QUERY_TIMEOUT
        )
        
        return {
            "status": "success",
            "system_config": {
                "app_title": APP_TITLE,
                "app_version": APP_VERSION,
                "database_path": CHROMA_DB_PATH,
                "llm_model": OLLAMA_MODEL,
                "max_file_size": f"{MAX_FILE_SIZE / (1024*1024):.0f} MB",
                "supported_formats": SUPPORTED_FORMATS,
                "query_timeout": f"{QUERY_TIMEOUT} seconds"
            },
            "runtime_info": {
                "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
                "platform": psutil.sys.platform,
                "cpu_count": psutil.cpu_count(),
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@router.post("/system/restart")
async def restart_system() -> dict:
    """
    系统重启请求
    
    注意：这个接口只是返回重启指令，实际重启需要外部进程管理器支持
    """
    return {
        "status": "restart_requested",
        "message": "系统重启请求已接收，请通过进程管理器执行重启操作",
        "timestamp": datetime.now().isoformat()
    }

@router.get("/logs/recent")
async def get_recent_logs(lines: int = 50) -> dict:
    """
    获取最近的系统日志
    
    - **lines**: 返回的日志行数（默认50行，最多200行）
    """
    try:
        from config import LOG_FILE_NAME
        
        lines = min(lines, 200)  # 限制最大行数
        
        if os.path.exists(LOG_FILE_NAME):
            with open(LOG_FILE_NAME, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                
            return {
                "status": "success",
                "log_file": LOG_FILE_NAME,
                "total_lines": len(all_lines),
                "returned_lines": len(recent_lines),
                "logs": [line.strip() for line in recent_lines]
            }
        else:
            return {
                "status": "not_found",
                "message": f"日志文件 {LOG_FILE_NAME} 不存在"
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
