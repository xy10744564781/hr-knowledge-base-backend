from datetime import datetime
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 应用配置
APP_TITLE = '人事知识库系统'
APP_DESCRIPTION = '基于大模型API的人事文档管理和智能查询系统'
APP_VERSION = '2.0'

# 日志配置
LOG_FILE_NAME = f'./log/hr_kb_log_{datetime.now().strftime("%Y%m%d")}.log'

# ChromaDB配置
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "hr_knowledge"

# 阿里云百炼API配置（从环境变量读取）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 模型配置（从环境变量读取，带默认值）
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")

# LLM生成配置
LLM_TEMPERATURE = 0.1
LLM_TOP_P = 0.9
LLM_MAX_TOKENS = 2048
LLM_CONTEXT_LENGTH = 32000  # qwen-plus支持32K上下文

# 相关性阈值配置（从环境变量读取，带默认值）
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.5"))

# 查询配置
MAX_SEARCH_RESULTS = 5  # 检索文档数量
QUERY_TIMEOUT = 30  # 秒
MAX_QUERY_RESULTS = 10  # 最大查询结果数量

# 文档处理配置
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
SUPPORTED_FORMATS = ['.pdf', '.docx', '.doc', '.txt', '.md']

# 文档切割配置
CHUNK_SIZE = 1200  # 增加chunk大小
CHUNK_OVERLAP = 300  # 增加overlap
MIN_CHUNK_SIZE = 300  # 最小chunk大小

# 验证必需的环境变量
if not DASHSCOPE_API_KEY:
    raise ValueError("DASHSCOPE_API_KEY环境变量未设置！请在.env文件中配置。")
