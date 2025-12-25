"""
聊天会话标题生成服务
使用本地 LLM 根据用户第一条消息生成简短标题
"""
from langchain_ollama import ChatOllama
from config import OLLAMA_MODEL, OLLAMA_BASE_URL
from logging_setup import logger


def generate_session_title(first_message: str) -> str:
    """
    根据用户第一条消息生成会话标题
    
    Args:
        first_message: 用户的第一条消息内容
        
    Returns:
        生成的标题（5-10字）
    """
    try:
        # 构建 prompt
        prompt = f"""请根据用户的问题，生成一个5-10字的简短标题，要求：
1. 简洁明了，概括核心主题
2. 不要加引号、书名号或其他符号
3. 直接返回标题文本，不要有任何解释
4. 使用中文

用户问题：{first_message}

标题："""

        # 创建 LLM 实例（使用简单配置）
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
            num_predict=50,  # 限制生成长度
        )
        
        # 生成标题
        response = llm.invoke(prompt)
        title = response.content.strip()
        
        # 如果标题太长，截取前10个字
        if len(title) > 15:
            title = title[:15] + "..."
        
        # 如果标题为空或生成失败，使用消息前10字作为标题
        if not title or len(title) < 2:
            title = first_message[:10] + "..." if len(first_message) > 10 else first_message
        
        logger.info(f"生成会话标题成功: {title}")
        return title
        
    except Exception as e:
        logger.error(f"生成会话标题失败: {e}")
        # 降级方案：使用消息前10字
        fallback_title = first_message[:10] + "..." if len(first_message) > 10 else first_message
        return fallback_title
