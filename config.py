from datetime import datetime
import os
from dotenv import load_dotenv

# 清除 SOCKS 代理设置，避免 httpx 不兼容问题
# （httpx 默认不支持 socks:// 协议）
for _proxy_var in ['ALL_PROXY', 'all_proxy']:
    os.environ.pop(_proxy_var, None)

# 加载环境变量
# 检查 .env 文件是否存在
_env_file_path = os.path.join(os.path.dirname(__file__), '.env')
_env_file_exists = os.path.exists(_env_file_path)

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

# 打印API Key信息（脱敏处理）
def _mask_api_key(key: str) -> str:
    """脱敏处理API Key，只显示前8位和后4位"""
    if not key:
        return "未设置"
    if len(key) <= 12:
        return key[:4] + "****" + key[-2:]
    return key[:8] + "****" + key[-4:]

# 使用简单的日志记录（避免循环导入）
import logging
_config_logger = logging.getLogger("config")
_config_logger.setLevel(logging.INFO)

# 同时输出到控制台和日志文件
if not _config_logger.handlers:
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    _config_logger.addHandler(console_handler)
    
    # 文件处理器
    try:
        os.makedirs('./log', exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE_NAME, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(console_formatter)
        _config_logger.addHandler(file_handler)
    except Exception as e:
        print(f"无法创建日志文件处理器: {e}")

_config_logger.info(f"配置文件 .env 存在: {_env_file_exists}")
_config_logger.info(f"API Key: {_mask_api_key(DASHSCOPE_API_KEY)} (来源: {'系统环境变量' if not _env_file_exists else '.env文件或系统环境变量'})")
_config_logger.info(f"Embedding模型: {os.getenv('EMBEDDING_MODEL', 'text-embedding-v3')}")
_config_logger.info(f"LLM模型: {os.getenv('LLM_MODEL', 'qwen-plus')}")
_config_logger.info(f"相关性阈值: {os.getenv('RELEVANCE_THRESHOLD', '0.5')}")

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
