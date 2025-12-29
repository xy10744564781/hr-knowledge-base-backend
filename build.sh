#!/bin/bash
# HR知识库系统打包脚本

set -e

echo "=== HR知识库系统打包 ==="

# 检查是否在conda环境中
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "警告: 未检测到conda环境，请先激活: conda activate hr-knowledge-base-backend"
    exit 1
fi

echo "当前conda环境: $CONDA_DEFAULT_ENV"

# 安装PyInstaller
echo "安装PyInstaller..."
pip install pyinstaller -q

# 清理旧的构建文件
echo "清理旧构建..."
rm -rf build dist *.spec

# 打包 - 使用 --paths 指定本地模块路径
echo "开始打包..."
pyinstaller \
    --name hr-knowledge-base \
    --onedir \
    --paths . \
    --add-data "prompt:prompt" \
    --add-data ".env.example:." \
    --add-data "config.py:." \
    --add-data "database.py:." \
    --add-data "schemas.py:." \
    --add-data "services.py:." \
    --add-data "logging_setup.py:." \
    --add-data "llm_agent.py:." \
    --add-data "knowledge_base.py:." \
    --add-data "query_router.py:." \
    --add-data "relevance_evaluator.py:." \
    --add-data "document_splitter.py:." \
    --add-data "dashscope_embeddings.py:." \
    --add-data "title_generator.py:." \
    --add-data "chat_service.py:." \
    --add-data "__init__.py:." \
    --add-data "routers:routers" \
    --hidden-import=tiktoken_ext.openai_public \
    --hidden-import=tiktoken_ext \
    --hidden-import=chromadb \
    --hidden-import=chromadb.config \
    --hidden-import=pydantic \
    --hidden-import=sqlalchemy.dialects.mysql \
    --hidden-import=pymysql \
    --hidden-import=langchain_openai \
    --hidden-import=httpx \
    --hidden-import=uvicorn \
    --hidden-import=uvicorn.logging \
    --hidden-import=uvicorn.loops \
    --hidden-import=uvicorn.loops.auto \
    --hidden-import=uvicorn.protocols \
    --hidden-import=uvicorn.protocols.http \
    --hidden-import=uvicorn.protocols.http.auto \
    --hidden-import=uvicorn.protocols.websockets \
    --hidden-import=uvicorn.protocols.websockets.auto \
    --hidden-import=uvicorn.lifespan \
    --hidden-import=uvicorn.lifespan.on \
    --hidden-import=fastapi \
    --collect-all chromadb \
    --collect-all tiktoken \
    --noconfirm \
    main.py

# 复制必要的运行时文件
echo "复制运行时文件..."
cp -r static dist/hr-knowledge-base/ 2>/dev/null || mkdir -p dist/hr-knowledge-base/static
mkdir -p dist/hr-knowledge-base/log
mkdir -p dist/hr-knowledge-base/chroma_db

# 创建启动脚本
cat > dist/hr-knowledge-base/start.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"

# 设置 PYTHONPATH 让程序能找到 _internal 目录下的模块
export PYTHONPATH="$(pwd)/_internal:$PYTHONPATH"

# 复制.env配置文件（如果不存在）
if [ ! -f .env ]; then
    if [ -f _internal/.env.example ]; then
        cp _internal/.env.example .env
        echo "已创建 .env 配置文件，请修改后重新运行"
        exit 0
    else
        echo "请创建.env配置文件"
        exit 1
    fi
fi

# 复制 .env 到 _internal 目录（程序从那里读取）
cp .env _internal/.env 2>/dev/null || true

./hr-knowledge-base
EOF
chmod +x dist/hr-knowledge-base/start.sh

echo "=== 打包完成 ==="
echo "输出目录: dist/hr-knowledge-base/"
echo "运行方式: cd dist/hr-knowledge-base && ./start.sh"
