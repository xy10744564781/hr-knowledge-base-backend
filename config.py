from datetime import datetime
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 应用配置
APP_TITLE = '人事知识库系统'
APP_DESCRIPTION = '基于本地大模型的人事文档管理和智能查询系统'
APP_VERSION = '1.0'

# 日志配置
LOG_FILE_NAME = f'./log/hr_kb_log_{datetime.now().strftime("%Y%m%d")}.log'

# ChromaDB配置
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "hr_knowledge"

# Ollama配置
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:8b"  # 切换到更快的模型（原: deepseek-r1:7b）
OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large:latest"

# LLM生成配置 - 优化配置
LLM_TEMPERATURE = 0.1
LLM_TOP_P = 0.9
LLM_MAX_TOKENS = 2048  # 增加生成长度，确保回答完整
LLM_CONTEXT_LENGTH = 2048  # 增加上下文长度

# 响应一致性配置
RESPONSE_MODE = "hybrid"  # "llm", "template", "hybrid", "optimized"
SIMPLE_QUERY_THRESHOLD = 10  # 简单查询字符数阈值
OPTIMIZATION_ATTEMPTS = 3  # LLM优化尝试次数
MIN_CONFIDENCE_FOR_TEMPLATE = 0.7  # 使用模板的最低置信度

# 查询配置 - 性能优化
MAX_SEARCH_RESULTS = 3  # 减少检索文档数量
MIN_SIMILARITY_SCORE = 0.0  # 降低阈值，允许所有搜索结果
QUERY_TIMEOUT = 30  # 秒 - 大幅减少超时时间
MAX_QUERY_RESULTS = 10  # 最大查询结果数量

# 文档处理配置
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
SUPPORTED_FORMATS = ['.pdf', '.docx', '.doc', '.txt', '.md']
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
