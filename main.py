from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from config import APP_TITLE, APP_DESCRIPTION, APP_VERSION
from logging_setup import logger
from routers import query_router, upload_router, admin_router, chat_router, auth_router, permission_router, department_router
from database import init_database

# 加载 services 模块，完成全局初始化
import services

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION
)

# 添加CORS中间件，支持前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000", 
        "http://127.0.0.1:8000", 
        "http://localhost:8004", 
        "http://127.0.0.1:8004",
        "http://192.168.100.20:8080",  # Vue 开发服务器
        "http://192.168.100.20:8004",  # 生产环境
        "*"  # 允许所有来源（开发环境）
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],  # 暴露 Content-Disposition header 给前端
)

# 创建必要的目录
os.makedirs("log", exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)
os.makedirs("static", exist_ok=True)

# 初始化数据库
try:
    init_database()
    logger.info("数据库初始化成功")
except Exception as e:
    logger.error(f"数据库初始化失败: {e}")

# 注册路由
app.include_router(auth_router.router)  # 认证路由
app.include_router(permission_router.router)  # 权限管理路由
app.include_router(department_router.router)  # 部门管理路由
app.include_router(query_router.router, prefix="/api")
app.include_router(upload_router.router, prefix="/api")
app.include_router(admin_router.router, prefix="/api")
app.include_router(chat_router.router, prefix="/api")

# 静态文件服务（用于前端）
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
async def root():
    return {"message": "企业知识库系统运行中", "status": "ok"}

if __name__ == "__main__":
    import multiprocessing
    
    # 根据 CPU 核心数计算最佳 worker 数量
    # 公式: CPU核心数 / 2，最少1个worker
    cpu_count = multiprocessing.cpu_count()
    workers = max(1, cpu_count // 2)
    
    logger.info(f"启动{APP_TITLE}")
    logger.info(f"检测到 {cpu_count} 个 CPU 核心")
    logger.info(f"启动 {workers} 个 worker 进程以支持并发请求")
    
    # 启动服务器（监听所有网络接口，允许局域网访问）
    # 使用字符串导入方式，避免 uvicorn 警告
    uvicorn.run(
        "main:app",  # 使用字符串导入，而不是直接传递 app 对象
        host="0.0.0.0", 
        port=8004,
        workers=workers,
        timeout_keep_alive=75,
        limit_concurrency=1000,
        limit_max_requests=10000
    )
