"""
智能文档切割器 - 基于语义的人事文档切割
使用LangChain框架，针对人事文档特点进行优化
"""
from typing import List, Dict, Any
from langchain.text_splitter import TextSplitter
from langchain.schema import Document
import re
from logging_setup import logger
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


class SemanticHRSplitter(TextSplitter):
    """人事文档语义切割器"""
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # 人事领域关键词
        self.hr_keywords = {
            '薪资': ['薪资', '工资', '薪酬', '奖金', '津贴', '补贴'],
            '考勤': ['考勤', '打卡', '请假', '休假', '迟到', '早退', '出勤'],
            '培训': ['培训', '学习', '发展', '课程', '培训计划'],
            '入职': ['入职', '新员工', '报到', '入职手续', '试用期'],
            '离职': ['离职', '辞职', '退休', '离职手续', '离职流程'],
            '福利': ['福利', '待遇', '补贴', '津贴', '福利待遇'],
            '绩效': ['绩效', '考核', '评估', '考评', 'KPI'],
            '招聘': ['招聘', '面试', '录用', '招聘流程']
        }
    
    def split_text(self, text: str) -> List[str]:
        """切割文本为语义完整的块"""
        from logging_setup import logger
        
        # 识别章节结构
        sections = self._identify_sections(text)
        
        logger.info(f"识别到 {len(sections)} 个章节")
        
        if not sections:
            # 如果没有明显的章节结构，使用递归切割
            logger.info(f"未识别到章节结构，使用递归切割（文本长度: {len(text)}）")
            chunks = self._recursive_split(text)
            logger.info(f"递归切割完成，生成 {len(chunks)} 个块")
            return chunks
        
        # 基于章节进行切割
        chunks = []
        for section in sections:
            section_text = section['content']
            
            if len(section_text) <= self.chunk_size:
                # 章节足够小，直接作为一个chunk
                chunks.append(section_text)
            else:
                # 章节太大，需要进一步切割
                sub_chunks = self._split_large_section(section_text, section['title'])
                chunks.extend(sub_chunks)
        
        logger.info(f"章节切割完成，生成 {len(chunks)} 个块")
        return chunks
    
    def _identify_sections(self, text: str) -> List[Dict[str, Any]]:
        """识别文档的章节结构"""
        sections = []
        
        # 匹配常见的章节标题模式
        patterns = [
            r'^(第?[一二三四五六七八九十百]+[章节条][\s、：:].+)$',  # 第一章、第二节
            r'^([一二三四五六七八九十]+[、\s].+)$',  # 一、标题
            r'^(\d+[\.\s、].+)$',  # 1. 标题 或 1、标题
            r'^([A-Z][\.\s、].+)$',  # A. 标题
        ]
        
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_content:
                    current_content.append('')
                continue
            
            # 检查是否是章节标题
            is_title = False
            for pattern in patterns:
                if re.match(pattern, line, re.MULTILINE):
                    # 保存上一个章节
                    if current_section:
                        sections.append({
                            'title': current_section,
                            'content': '\n'.join(current_content).strip()
                        })
                    
                    # 开始新章节
                    current_section = line
                    current_content = []
                    is_title = True
                    break
            
            if not is_title:
                current_content.append(line)
        
        # 保存最后一个章节
        if current_section and current_content:
            sections.append({
                'title': current_section,
                'content': '\n'.join(current_content).strip()
            })
        
        return sections
    
    def _split_large_section(self, text: str, section_title: str) -> List[str]:
        """切割大章节，保持语义完整性"""
        chunks = []
        
        # 按段落切割
        paragraphs = text.split('\n\n')
        current_chunk = f"【{section_title}】\n\n"
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 检查添加这个段落后是否超过大小限制
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk += para + '\n\n'
            else:
                # 保存当前chunk
                if len(current_chunk.strip()) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                
                # 开始新chunk，包含章节标题和overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = f"【{section_title}】\n\n{overlap_text}{para}\n\n"
        
        # 保存最后一个chunk
        if len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """获取overlap文本"""
        if len(text) <= self.chunk_overlap:
            return text
        
        # 从末尾取overlap大小的文本
        overlap = text[-self.chunk_overlap:]
        
        # 尝试从完整句子开始
        sentence_start = max(
            overlap.find('。') + 1,
            overlap.find('！') + 1,
            overlap.find('？') + 1,
            overlap.find('\n') + 1
        )
        
        if sentence_start > 0:
            return overlap[sentence_start:]
        
        return overlap
    
    def _recursive_split(self, text: str) -> List[str]:
        """递归切割文本（当没有明显章节结构时）"""
        if len(text) <= self.chunk_size:
            return [text] if len(text) >= self.min_chunk_size else []
        
        chunks = []
        separators = ['\n\n', '\n', '。', '；', '，', ' ']
        
        for separator in separators:
            if separator in text:
                parts = text.split(separator)
                current_chunk = ''
                
                for part in parts:
                    if len(current_chunk) + len(part) + len(separator) <= self.chunk_size:
                        current_chunk += part + separator
                    else:
                        if len(current_chunk.strip()) >= self.min_chunk_size:
                            chunks.append(current_chunk.strip())
                        
                        # 添加overlap
                        overlap = self._get_overlap_text(current_chunk)
                        current_chunk = overlap + part + separator
                
                if len(current_chunk.strip()) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                
                if chunks:
                    return chunks
        
        # 如果所有分隔符都失败，强制按大小切割
        return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取文本中的人事关键词"""
        keywords = []
        for category, words in self.hr_keywords.items():
            for word in words:
                if word in text:
                    keywords.append(word)
        return list(set(keywords))
    
    def create_documents(self, texts: List[str], metadatas: List[Dict] = None) -> List[Document]:
        """创建带元数据的文档对象"""
        documents = []
        
        for i, text in enumerate(texts):
            # 提取关键词
            keywords = self._extract_keywords(text)
            
            # 构建元数据
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            metadata.update({
                'keywords': keywords,
                'chunk_length': len(text),
                'has_section_title': '【' in text and '】' in text
            })
            
            documents.append(Document(page_content=text, metadata=metadata))
        
        return documents


def create_hr_splitter(chunk_size: int = None, chunk_overlap: int = None) -> SemanticHRSplitter:
    """创建人事文档切割器的工厂函数"""
    return SemanticHRSplitter(
        chunk_size=chunk_size or CHUNK_SIZE,
        chunk_overlap=chunk_overlap or CHUNK_OVERLAP
    )
