# 人事知识库系统

基于本地大模型的人事文档管理和智能查询系统

## 功能特点

- 📁 文档上传与管理（支持PDF、Word、文本文件）
- 🔍 智能文档检索（基于ChromaDB向量数据库）
- 🤖 AI智能问答（基于Ollama本地大模型）
- 🌐 Web界面操作（简洁易用的前端界面）
- 🔒 安全访问控制（本地访问限制）

## 系统要求

- Python 3.11+
- Ollama（本地大模型服务）
- 8GB+ 内存
- Windows/Linux/MacOS

## 快速开始

### 1. 安装依赖

```bash
cd hr_knowledge_base
pip install -r requirements.txt
```

### 2. 启动Ollama服务

```bash
ollama serve
```

### 3. 下载模型

```bash
# 下载deepseek-r1:7b模型
ollama pull deepseek-r1:7b

# 下载embedding模型
ollama pull mxbai-embed-large
```

### 4. 启动系统

Windows:
```bash
start.bat
```

或直接运行:
```bash
python main.py
```

### 5. 访问系统

打开浏览器访问: http://127.0.0.1:8004

## 配置说明

主要配置在 `config.py` 文件中：

```python
# Ollama配置
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "deepseek-r1:7b"  # 可更换其他模型
OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large:latest"

# 数据库配置
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "hr_knowledge"

# 文档处理配置
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
SUPPORTED_FORMATS = ['.pdf', '.docx', '.doc', '.txt', '.md']
```

## 使用说明

### 上传文档

1. 点击"选择文件"按钮或拖拽文件到上传区域
2. 支持批量上传多个文件
3. 系统自动处理并存储到向量数据库

### 智能查询

1. 在查询框输入问题
2. 点击"查询"按钮
3. 系统会检索相关文档并生成智能回答

### 文档管理

- 查看已上传的文档列表
- 删除不需要的文档
- 在特定文档中搜索

## 项目结构

```
hr_knowledge_base/
├── main.py                 # 主程序入口
├── config.py              # 配置文件
├── services.py            # 业务逻辑层
├── schemas.py             # 数据模型
├── knowledge_base.py      # 向量数据库管理
├── llm_agent.py          # LLM集成
├── routers/              # API路由
│   ├── query_router.py   # 查询接口
│   ├── upload_router.py  # 上传接口
│   └── admin_router.py   # 管理接口
├── static/               # 前端文件
│   └── index.html        # Web界面
├── prompt/               # 提示词模板
├── chroma_db/            # 向量数据库存储
└── log/                  # 日志文件
```

## 常见问题

### 1. 查询超时

如果使用较大的模型（如qwen3:8b），查询可能需要较长时间。可以：
- 切换到更快的模型（如deepseek-r1:7b）
- 增加系统内存
- 简化查询问题

### 2. 文档上传失败

检查：
- 文件格式是否支持
- 文件大小是否超过50MB
- 磁盘空间是否充足

### 3. 无法连接Ollama

确保：
- Ollama服务已启动（`ollama serve`）
- 端口11434未被占用
- 模型已下载

## 技术栈

- **后端**: FastAPI, Python 3.11
- **向量数据库**: ChromaDB
- **LLM**: Ollama (deepseek-r1:7b)
- **文档处理**: LangChain
- **前端**: HTML, CSS, JavaScript

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue。