import tempfile
import os
import mimetypes
from typing import List, Tuple, Optional
from fastapi import UploadFile, HTTPException
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logging_setup import logger
from config import MAX_FILE_SIZE, SUPPORTED_FORMATS

chroma_collection = None

def init_chroma():
    """初始化ChromaDB向量数据库"""
    global chroma_collection
    if chroma_collection is None:
        try:
            import chromadb
            from chromadb.config import Settings
            
            # 创建设置，明确禁用遥测
            settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
            
            # 尝试持久化模式
            client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=settings
            )
            logger.info("使用ChromaDB持久化模式")
            
            # 获取或创建集合
            chroma_collection = client.get_or_create_collection(
                name="hr_knowledge_base",
                metadata={"description": "人事知识库向量存储"}
            )
            
            logger.info("ChromaDB初始化成功")
            
        except Exception as e:
            logger.error(f"ChromaDB初始化失败: {e}", exc_info=True)
            chroma_collection = None
            
    return chroma_collection

class VectorManager:
    """向量管理器包装类"""
    
    def __init__(self, collection):
        self.collection = collection
    
    def check_duplicate_title(self, title, exclude_doc_id=None):
        """检查标题是否重复"""
        if not self.collection:
            return False
            
        try:
            # 查询所有文档的元数据
            results = self.collection.get(include=['metadatas'])
            
            if not results or not results.get('metadatas'):
                return False
                
            # 检查是否有相同标题的文档
            for i, metadata in enumerate(results['metadatas']):
                if metadata and metadata.get('title') == title:
                    # 如果指定了排除的文档ID，跳过该文档
                    if exclude_doc_id and results.get('ids') and results['ids'][i] == exclude_doc_id:
                        continue
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"检查重复标题时出错: {e}")
            return False
    
    def add_document(self, texts, metadatas, document_id=None, ids=None):
        """添加文档"""
        if not self.collection:
            raise Exception("ChromaDB未初始化")
        
        # 生成文档ID
        if ids is None:
            if isinstance(texts, list):
                if document_id:
                    ids = [f"{document_id}_chunk_{i}" for i in range(len(texts))]
                else:
                    import uuid
                    base_id = str(uuid.uuid4())
                    ids = [f"{base_id}_chunk_{i}" for i in range(len(texts))]
            else:
                ids = [document_id or str(uuid.uuid4())]
                texts = [texts]
                metadatas = [metadatas]
        else:
            # 如果提供了ids，确保texts和metadatas是列表
            if not isinstance(texts, list):
                texts = [texts]
            if not isinstance(metadatas, list):
                metadatas = [metadatas]
            
        try:
            self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
            return True
        except Exception as e:
            logger.error(f"添加文档时出错: {e}")
            return False
    
    def delete_document(self, document_id):
        """删除文档"""
        if not self.collection:
            raise Exception("ChromaDB未初始化")
        
        try:
            # 查找所有相关的chunk
            results = self.collection.get(include=['metadatas'])
            if results and results.get('ids'):
                # 找到所有属于该文档的chunk
                chunk_ids = []
                for i, metadata in enumerate(results.get('metadatas', [])):
                    if metadata and metadata.get('document_id') == document_id:
                        chunk_ids.append(results['ids'][i])
                
                if chunk_ids:
                    self.collection.delete(ids=chunk_ids)
                    return True
            return False
        except Exception as e:
            logger.error(f"删除文档时出错: {e}")
            return False
    
    def update_document(self, document_id, texts, metadatas, ids=None):
        """更新文档"""
        if not self.collection:
            raise Exception("ChromaDB未初始化")
        
        try:
            # 先删除旧的chunks
            self.delete_document(document_id)
            # 再添加新的chunks
            return self.add_document(texts, metadatas, document_id, ids)
        except Exception as e:
            logger.error(f"更新文档时出错: {e}")
            return False
    
    def search_documents(self, query, k=10, filter_metadata=None):
        """搜索文档"""
        if not self.collection:
            raise Exception("ChromaDB未初始化")
        
        try:
            where_clause = filter_metadata if filter_metadata else None
            results = self.collection.query(
                query_texts=[query], 
                n_results=k,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            # 转换ChromaDB结果为文档对象列表
            documents = []
            if results and 'documents' in results and results['documents']:
                docs = results['documents'][0]  # 第一个查询的结果
                metadatas = results.get('metadatas', [[]])[0]
                distances = results.get('distances', [[]])[0]
                
                for i, doc_content in enumerate(docs):
                    # 创建文档对象
                    doc_obj = type('Document', (), {
                        'page_content': doc_content,
                        'metadata': metadatas[i] if i < len(metadatas) else {},
                        'score': max(0.0, 1.0 - distances[i]) if i < len(distances) else 0.0  # 转换距离为相似度分数，确保非负
                    })()
                    documents.append(doc_obj)
            
            return documents
        except Exception as e:
            logger.error(f"搜索文档时出错: {e}")
            return []
    
    def list_documents(self, limit=None):
        """列出文档"""
        if not self.collection:
            return []
        
        try:
            results = self.collection.get(include=['metadatas'])
            if not results or not results.get('metadatas'):
                return []
            
            # 按document_id分组，避免重复
            documents = {}
            for i, metadata in enumerate(results['metadatas']):
                if metadata and 'document_id' in metadata:
                    doc_id = metadata['document_id']
                    if doc_id not in documents:
                        documents[doc_id] = metadata
            
            doc_list = list(documents.values())
            if limit:
                doc_list = doc_list[:limit]
                
            return doc_list
        except Exception as e:
            logger.error(f"列出文档时出错: {e}")
            return []
    
    def get_collection_stats(self):
        """获取集合统计信息"""
        if not self.collection:
            return {"total_documents": 0}
        try:
            count = self.collection.count()
            return {"total_documents": count}
        except Exception as e:
            logger.error(f"获取统计信息时出错: {e}")
            return {"total_documents": 0}

def get_vector_manager():
    """获取向量管理器"""
    collection = init_chroma()
    return VectorManager(collection)

def validate_file(file: UploadFile) -> Tuple[bool, str]:
    """
    验证上传文件的有效性
    返回: (是否有效, 错误信息)
    """
    try:
        # 检查文件名
        if not file.filename:
            return False, "文件名不能为空"
        
        # 检查文件扩展名
        file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        if not file_ext:
            return False, "文件必须有扩展名"
        
        if f".{file_ext}" not in SUPPORTED_FORMATS:
            return False, f"不支持的文件格式: .{file_ext}。支持的格式: {', '.join(SUPPORTED_FORMATS)}"
        
        # 检查文件大小
        file.file.seek(0, 2)  # 移动到文件末尾
        file_size = file.file.tell()
        file.file.seek(0)  # 重置到文件开头
        
        if file_size == 0:
            return False, "文件不能为空"
        
        if file_size > MAX_FILE_SIZE:
            return False, f"文件大小超过限制 ({file_size / (1024*1024):.1f}MB > {MAX_FILE_SIZE / (1024*1024)}MB)"
        
        # 检查MIME类型（如果可能）
        mime_type, _ = mimetypes.guess_type(file.filename)
        if mime_type:
            allowed_mimes = {
                'application/pdf': ['.pdf'],
                'text/plain': ['.txt', '.md', '.log'],
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
                'application/msword': ['.doc']
            }
            
            valid_mime = False
            for allowed_mime, extensions in allowed_mimes.items():
                if mime_type == allowed_mime and f".{file_ext}" in extensions:
                    valid_mime = True
                    break
            
            if not valid_mime and mime_type not in ['text/x-python', 'application/javascript', 'application/json']:
                logger.warning(f"MIME类型可能不匹配: {mime_type} for {file.filename}")
        
        return True, ""
        
    except Exception as e:
        logger.error(f"文件验证失败: {str(e)}")
        return False, f"文件验证失败: {str(e)}"

def get_optimal_chunk_settings(file_ext: str, file_size: int) -> Tuple[int, int]:
    """
    根据文件类型和大小获取最优的分块设置
    返回: (chunk_size, chunk_overlap)
    """
    # 基础设置
    base_chunk_size = 1000
    base_overlap = 200
    
    # 根据文件类型调整
    if file_ext == 'pdf':
        # PDF通常内容密度较高
        chunk_size = 1200
        overlap = 250
    elif file_ext in ['docx', 'doc']:
        # Word文档通常格式化较好
        chunk_size = 1000
        overlap = 200
    elif file_ext in ['md', 'txt']:
        # 纯文本文档
        chunk_size = 800
        overlap = 150
    else:
        # 代码文件等
        chunk_size = 600
        overlap = 100
    
    # 根据文件大小调整
    if file_size > 5 * 1024 * 1024:  # 大于5MB
        chunk_size = int(chunk_size * 1.2)
        overlap = int(overlap * 1.1)
    elif file_size < 100 * 1024:  # 小于100KB
        chunk_size = int(chunk_size * 0.8)
        overlap = int(overlap * 0.8)
    
    return chunk_size, overlap

def create_document_loader(temp_path: str, file_ext: str, filename: str):
    """
    根据文件类型创建合适的文档加载器
    """
    try:
        if file_ext == 'pdf':
            # 优先使用PyPDFLoader
            return PyPDFLoader(temp_path)
        elif file_ext in ["txt", "md", "c", "h", "cpp", "hpp", "java", "js", "ts", "jsx", "tsx",
                          "vue", "py", "go", "rs", "sql", "json", "yml", "yaml", "cfg", "ini", "log"]:
            # 尝试多种编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16', 'latin1']
            for encoding in encodings:
                try:
                    return TextLoader(temp_path, encoding=encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"无法使用任何编码读取文件: {filename}")
        elif file_ext in ['docx', 'doc']:
            return Docx2txtLoader(temp_path)
        else:
            raise ValueError(f'不支持的文档格式: {file_ext}')
    except Exception as e:
        logger.error(f"创建文档加载器失败: {str(e)}")
        raise

def process_upload_file(file: UploadFile) -> List[str]:
    """
    处理上传的文件并提取文本内容
    优化版本：更好的错误处理、文件验证和内存管理
    """
    logger.info(f"开始处理文件: {file.filename}")
    
    # 1. 文件验证
    is_valid, error_msg = validate_file(file)
    if not is_valid:
        logger.error(f"文件验证失败: {error_msg}")
        raise ValueError(error_msg)
    
    # 2. 获取文件信息
    file_ext = file.filename.split('.')[-1].lower()
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    logger.info(f"文件信息: 类型={file_ext}, 大小={file_size / 1024:.1f}KB")
    
    # 3. 创建临时文件
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
            # 分块读取文件内容，避免内存问题
            chunk_size = 8192
            while True:
                chunk = file.file.read(chunk_size)
                if not chunk:
                    break
                temp_file.write(chunk)
            temp_path = temp_file.name
        
        logger.info(f"临时文件创建成功: {temp_path}")
        
        # 4. 获取最优分块设置
        chunk_size, chunk_overlap = get_optimal_chunk_settings(file_ext, file_size)
        logger.info(f"分块设置: chunk_size={chunk_size}, overlap={chunk_overlap}")
        
        # 5. 创建文档加载器
        loader = create_document_loader(temp_path, file_ext, file.filename)
        
        # 6. 创建文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        
        # 7. 加载和分割文档
        try:
            documents = loader.load()
            logger.info(f"文档加载成功，原始页数: {len(documents)}")
            
            if not documents:
                raise ValueError("文档内容为空或无法读取")
            
            # 检查文档内容
            total_content = sum(len(doc.page_content) for doc in documents)
            if total_content < 10:
                raise ValueError("文档内容过少，可能是空文档或格式不正确")
            
            documents = text_splitter.split_documents(documents)
            logger.info(f"文档分割完成，分块数: {len(documents)}")
            
            # 过滤空白或过短的块
            valid_chunks = []
            for doc in documents:
                content = doc.page_content.strip()
                if len(content) >= 20:  # 至少20个字符
                    valid_chunks.append(content)
            
            if not valid_chunks:
                raise ValueError("文档分割后没有有效内容块")
            
            logger.info(f"有效内容块数量: {len(valid_chunks)}")
            return valid_chunks
            
        except Exception as e:
            logger.error(f'主要加载器失败: {str(e)}')
            
            # 8. 备用处理方法
            if file_ext == 'pdf':
                try:
                    logger.warning("尝试使用备用PDF处理方法")
                    loader = UnstructuredPDFLoader(temp_path)
                    documents = loader.load()
                    
                    if documents:
                        documents = text_splitter.split_documents(documents)
                        valid_chunks = [doc.page_content.strip() for doc in documents 
                                      if len(doc.page_content.strip()) >= 20]
                        if valid_chunks:
                            logger.info(f"备用方法成功，有效块数: {len(valid_chunks)}")
                            return valid_chunks
                    
                except Exception as backup_e:
                    logger.error(f'备用PDF处理方法也失败: {str(backup_e)}')
            
            # 如果所有方法都失败，抛出详细错误
            raise ValueError(f'文件处理失败: {str(e)}。请检查文件格式是否正确，或尝试转换为其他支持的格式。')
    
    except ValueError:
        # 重新抛出验证错误
        raise
    except Exception as e:
        logger.error(f'文件处理过程中发生未预期错误: {str(e)}', exc_info=True)
        raise ValueError(f'文件处理失败: {str(e)}')
    
    finally:
        # 9. 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.info("临时文件清理完成")
            except Exception as e:
                logger.warning(f"临时文件清理失败: {str(e)}")

def get_document_info(file: UploadFile) -> dict:
    """
    获取文档基本信息，不进行实际处理
    """
    try:
        file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        mime_type, _ = mimetypes.guess_type(file.filename)
        
        return {
            "filename": file.filename,
            "extension": file_ext,
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "mime_type": mime_type,
            "is_supported": f".{file_ext}" in SUPPORTED_FORMATS,
            "estimated_chunks": max(1, file_size // 1000)  # 粗略估计
        }
    except Exception as e:
        logger.error(f"获取文档信息失败: {str(e)}")
        return {"error": str(e)}
