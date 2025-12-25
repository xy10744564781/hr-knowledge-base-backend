from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from config import APP_TITLE, APP_DESCRIPTION, APP_VERSION
from logging_setup import logger
from routers import query_router, upload_router, admin_router, chat_router
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
app.include_router(query_router.router, prefix="/api")
app.include_router(upload_router.router, prefix="/api")
app.include_router(admin_router.router, prefix="/api")
app.include_router(chat_router.router, prefix="/api")

# 静态文件服务（用于前端）
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
async def root():
    return {"message": "人事知识库系统运行中", "status": "ok"}

if __name__ == "__main__":
    logger.info(f"启动{APP_TITLE}")
    
    # 启动服务器（监听所有网络接口，允许局域网访问）
    uvicorn.run(app, host="0.0.0.0", port=8004)
