# dev-api 分支说明

## 概述

本分支实现了基于阿里云百炼API的智能RAG系统，解决了之前本地部署方案的以下问题：

1. **问答机制不完善** - 实现了智能路由机制，根据相关性决定回答策略
2. **文档切割不完善** - 实现了基于语义的智能文档切割
3. **性能限制** - 切换到云端API，无性能顾虑
4. **检索不准确** - 使用更强大的embedding模型和相关性评估

## 技术栈

### API服务
- **阿里云百炼API** - 提供Embedding和LLM服务
- **Embedding模型**: `text-embedding-v3` (1536维)
- **LLM模型**: `qwen-plus` (性价比最高)

### 框架和库
- **LangChain** - RAG框架（性能优化好）
- **ChromaDB** - 向量存储
- **FastAPI** - Web服务框架

## 核心组件

### 1. 智能文档切割器 (`document_splitter.py`)

**功能**：
- 识别文档章节结构（一、二、三等标题）
- 保持语义完整性
- 提取人事领域关键词
- 元数据增强

**特点**：
- 继承LangChain的`TextSplitter`
- 支持多种章节标题格式
- 智能overlap处理
- 自动关键词提取

### 2. 相关性评估器 (`relevance_evaluator.py`)

**功能**：
- 评估检索结果与查询的相关性
- 支持动态阈值调整
- 提供详细的评估指标

**评估指标**：
- `is_relevant`: 是否相关（bool）
- `max_score`: 最高相似度分数
- `avg_score`: 平均相似度分数
- `relevant_count`: 相关文档数量
- `relevant_ratio`: 相关文档比例

### 3. 查询路由器 (`query_router.py`)

**功能**：
- 智能路由查询到合适的回答策略
- 相关→文档回答，不相关→通用回答
- 集成相关性评估

**路由策略**：
- `document_based`: 基于文档的回答（相关性满足阈值）
- `general_knowledge`: 通用知识回答（相关性不满足阈值）

### 4. 向量存储管理 (`knowledge_base.py`)

**重构内容**：
- 使用`langchain_openai.OpenAIEmbeddings`配置阿里云API
- 使用`langchain_chroma.Chroma`作为向量存储
- 集成智能文档切割器
- 优化文档管理接口

### 5. LLM代理 (`llm_agent.py`)

**重构内容**：
- 使用`langchain_openai.ChatOpenAI`配置阿里云API
- 支持基于文档和通用知识两种回答模式
- 优化提示词构建
- 支持流式响应

### 6. 业务服务 (`services.py`)

**重构内容**：
- 集成`QueryRouter`进行智能路由
- 优化查询处理流程
- 更新流式响应生成
- 改进健康检查信息

## 配置说明

### 环境变量 (`.env`)

```env
# 阿里云百炼API配置（必需）
DASHSCOPE_API_KEY=your_api_key_here

# 模型配置（可选，有默认值）
EMBEDDING_MODEL=text-embedding-v3
LLM_MODEL=qwen-plus

# 相关性阈值（可选，默认0.5）
RELEVANCE_THRESHOLD=0.5
```

### 配置文件 (`config.py`)

主要配置项：
- `DASHSCOPE_API_KEY`: 从环境变量读取（不写入配置文件）
- `DASHSCOPE_BASE_URL`: 阿里云百炼API地址
- `EMBEDDING_MODEL`: Embedding模型名称
- `LLM_MODEL`: LLM模型名称
- `RELEVANCE_THRESHOLD`: 相关性阈值
- `CHUNK_SIZE`: 文档切割大小（1200）
- `CHUNK_OVERLAP`: 切割重叠大小（300）

## 安装和运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要新增依赖：
- `langchain-openai==0.1.0` - LangChain的OpenAI集成（兼容阿里云API）

### 2. 配置环境变量

复制`.env.example`为`.env`并填入API Key：

```bash
cp .env.example .env
# 编辑.env文件，填入DASHSCOPE_API_KEY
```

### 3. 测试配置

运行测试脚本验证配置：

```bash
python test_api_setup.py
```

测试内容：
- 环境变量配置
- Embedding功能
- LLM功能
- 文档切割器
- 相关性评估器

### 4. 启动服务

```bash
python main.py
```

服务将在 `http://localhost:8000` 启动。

## 工作流程

### 文档上传流程

1. 接收文档文件
2. 使用智能文档切割器切割文档
   - 识别章节结构
   - 保持语义完整性
   - 提取关键词
3. 使用阿里云Embedding API生成向量
4. 存储到ChromaDB向量数据库

### 查询处理流程

1. 接收用户查询
2. 查询预处理（标准化术语）
3. 使用`QueryRouter`进行智能路由：
   - 向量检索（使用阿里云Embedding API）
   - 相关性评估（使用`RelevanceEvaluator`）
   - 决策路由策略
4. 根据策略生成回答：
   - **document_based**: 使用检索到的文档 + LLM生成回答
   - **general_knowledge**: 使用LLM的通用知识回答
5. 返回结果（支持流式响应）

## 优势对比

### vs 本地Ollama方案

| 特性 | 本地Ollama | 阿里云API |
|------|-----------|----------|
| 性能 | 受限于本地硬件 | 云端高性能 |
| 稳定性 | 容易出现500错误 | 高可用性 |
| 模型质量 | 受限于本地模型 | 商业级模型 |
| 维护成本 | 需要管理本地服务 | 无需维护 |
| 扩展性 | 受限于硬件 | 弹性扩展 |

### 智能路由机制

**问题场景**：用户问"今天天气怎么样？"

- **旧方案**：强制使用检索到的文档回答，导致答非所问
- **新方案**：相关性评估不通过 → 使用通用知识回答 → 礼貌说明主要负责人事问题

## 下一步计划

- [ ] 添加更多人事领域的关键词
- [ ] 优化章节识别算法
- [ ] 实现文档摘要功能
- [ ] 添加查询历史分析
- [ ] 实现多轮对话支持
- [ ] 添加用户反馈机制

## 注意事项

1. **API Key安全**：
   - 不要将API Key提交到Git仓库
   - 使用环境变量管理敏感信息
   - `.env`文件已添加到`.gitignore`

2. **成本控制**：
   - 阿里云API按调用次数计费
   - 建议设置合理的阈值避免过度调用
   - 监控API使用量

3. **向量数据库**：
   - ChromaDB数据存储在`./chroma_db`目录
   - 切换模型后需要删除旧数据重新生成
   - 定期备份向量数据库

## 故障排查

### 问题1：API调用失败

**症状**：`DASHSCOPE_API_KEY环境变量未设置`

**解决**：
1. 检查`.env`文件是否存在
2. 确认`DASHSCOPE_API_KEY`已正确配置
3. 重启服务

### 问题2：Embedding维度不匹配

**症状**：`Embedding dimension不匹配`

**解决**：
1. 删除`chroma_db`目录
2. 重启服务，让系统创建新collection

### 问题3：相关性评估总是不通过

**症状**：所有查询都返回通用回答

**解决**：
1. 检查`RELEVANCE_THRESHOLD`配置（建议0.5）
2. 确认文档已正确上传
3. 查看日志中的相似度分数

## 联系方式

如有问题，请查看日志文件：`./log/hr_kb_log_YYYYMMDD.log`
