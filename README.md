# 企业知识库系统 v3.0

基于阿里云通义千问的企业级文档管理和智能查询系统

## 🚀 v3.0 特性

- 🏢 **多部门权限管理** - 基于RBAC的细粒度权限控制
- 👥 **用户角色系统** - 支持普通员工、部门管理员、超级管理员
- 💬 **智能聊天会话** - 支持多轮对话，会话历史管理和导出
- 🔐 **JWT身份认证** - 安全的用户登录和token自动过期处理
- 📊 **部门文档隔离** - 不同部门文档权限隔离访问
- 🌐 **现代化前端** - Vue3 + Ant Design Vue响应式设计
- 🔍 **混合检索** - 向量检索 + BM25 + 重排序的智能检索
- 🤖 **智能路由** - 自动识别问题意图，路由到相关部门文档

## 功能特点

- �  **文档管理** - 支持PDF、Word、Excel、文本文件的上传和管理
- � ***智能检索** - 基于ChromaDB向量数据库 + BM25的混合检索
- 🤖 **AI问答** - 基于阿里云通义千问的智能对话系统
- � **安企业级权限** - 多部门、多角色的访问控制
- 📱 **响应式界面** - 适配桌面和移动设备
- 🔒 **安全认证** - JWT token + 自动过期处理
- 📈 **查询优化** - 查询重述、结果重排序、相关性过滤

## 系统要求

- Python 3.11+
- MySQL 5.7+ 或 MariaDB 10.3+
- Node.js 16+
- 8GB+ 内存
- 阿里云通义千问API密钥
- Windows/Linux/MacOS

## 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd hr-knowledge-base-backend
```

### 2. 后端环境配置

```bash
# 安装Python依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
```

编辑 `.env` 文件：

```env
# 阿里云百炼API配置
DASHSCOPE_API_KEY=your_api_key_here

# 模型配置
EMBEDDING_MODEL=text-embedding-v3
LLM_MODEL=qwen-plus

# 相关性阈值
RELEVANCE_THRESHOLD=0.5

# 数据库配置
DATABASE_URL=mysql+pymysql://username:password@localhost:3306/hr_knowledge_base?charset=utf8mb4
```

### 3. 前端环境配置

```bash
cd aiDemo
npm install
# 或使用 yarn
yarn install
```

### 4. 初始化数据库

```bash
# 返回后端目录
cd ..
python database.py
```

### 5. 启动服务

**启动后端服务**：
```bash
python main.py
```

**启动前端服务**（新终端）：
```bash
cd aiDemo
npm run serve
# 或使用 yarn
yarn serve
```

### 6. 访问系统

- 前端界面: http://localhost:8080
- 后端API: http://localhost:8004
- API文档: http://localhost:8004/docs
- 默认管理员账号: admin / admin123

## 配置说明

### 环境变量配置

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `DASHSCOPE_API_KEY` | 阿里云通义千问API密钥 | 必填 |
| `EMBEDDING_MODEL` | 向量化模型 | text-embedding-v3 |
| `LLM_MODEL` | 对话模型 | qwen-plus |
| `RELEVANCE_THRESHOLD` | 相关性阈值 | 0.5 |
| `DATABASE_URL` | 数据库连接字符串 | 必填 |

### 模型配置

- **对话模型**: qwen-plus（支持32K上下文）
- **向量模型**: text-embedding-v3（高精度向量化）
- **重排序模型**: gte-rerank-v2（结果重排序）

### 检索配置

- **混合检索**: 向量检索 + BM25算法
- **查询重述**: 自动优化查询语句
- **结果重排序**: 提升检索精度
- **相关性过滤**: 过滤低相关性结果

## 用户角色说明

### � 角色权限

| 角色 | 权限说明 |
|------|----------|
| **超级管理员** | 查看所有部门文档、管理用户、上传删除文档、系统管理 |
| **部门管理员** | 查看本部门+公共文档、上传删除文档、编辑文档信息 |
| **普通员工** | 查看本部门+公共文档、使用聊天功能、导出聊天记录 |

### 📁 部门文档隔离

- **公共文档** - 所有用户都可以访问
- **部门文档** - 只有本部门用户和超级管理员可以访问
- **智能路由** - 系统自动根据问题内容路由到相关部门文档

## 使用说明

### 👤 用户注册登录

1. 访问系统首页，选择"注册"
2. 填写用户名、邮箱、密码和所属部门
3. 注册成功后自动登录（默认为普通员工角色）
4. 管理员可在用户管理页面调整用户角色

### 📁 文档管理

**上传文档**（需要上传权限）：
1. 点击"文档管理"进入上传页面
2. 选择文件或拖拽到上传区域
3. 选择目标部门文件夹
4. 系统自动处理并存储到向量数据库

**支持格式**：
- 文档：PDF、Word（.docx/.doc）、文本（.txt/.md）
- 表格：Excel（.xlsx/.xls）

**文档权限**：
- 上传到"公共"文件夹的文档所有人可见
- 上传到部门文件夹的文档只有本部门和管理员可见

### 💬 智能对话

1. 在聊天界面输入问题
2. 系统智能分析问题意图，路由到相关部门文档
3. 使用混合检索（向量+BM25）找到相关文档
4. 基于检索到的文档内容生成智能回答
5. 支持多轮对话，保持上下文连贯性
6. 可导出聊天记录为Markdown文件

### 👥 用户管理（仅超级管理员）

1. 查看所有用户列表
2. 调整用户角色（普通员工/部门管理员/超级管理员）
3. 启用/禁用用户账户

## 项目结构

```
hr-knowledge-base-backend/
├── main.py                           # 主程序入口
├── config.py                         # 配置管理
├── database.py                       # 数据库模型和RBAC权限
├── services.py                       # 业务逻辑层
├── knowledge_base.py                 # 向量数据库管理
├── llm_agent.py                     # LLM集成
├── chat_service.py                  # 聊天会话管理
├── user_service.py                  # 用户服务
├── hybrid_retriever.py              # 混合检索器
├── query_rephrase.py                # 查询重述
├── reranker.py                      # 结果重排序
├── routers/                         # API路由模块
│   ├── auth_router.py              # 认证相关接口
│   ├── chat_router.py              # 聊天会话接口
│   ├── query_router.py             # 查询接口
│   ├── upload_router.py            # 上传接口
│   ├── admin_router.py             # 管理接口
│   ├── permission_router.py        # 权限管理接口
│   └── department_router.py        # 部门管理接口
├── langchain_chains/               # LangChain处理链
│   └── intelligent_router.py      # 智能路由器
├── aiDemo/                         # Vue3前端项目
│   ├── src/
│   │   ├── components/            # Vue组件
│   │   ├── api/                   # API接口
│   │   └── utils/                 # 工具函数
│   ├── package.json
│   └── vue.config.js
├── prompt/                         # 提示词模板
├── chroma_db/                      # 向量数据库存储
├── files/                          # 上传文件存储
├── log/                            # 日志文件
├── .env                            # 环境变量配置
└── requirements.txt                # Python依赖
```

## 技术栈

### 后端技术
- **框架**: FastAPI 0.115+
- **数据库**: MySQL 5.7+ / MariaDB 10.3+
- **ORM**: SQLAlchemy 2.0+
- **认证**: JWT (PyJWT)
- **向量数据库**: ChromaDB 0.5+
- **LLM**: 阿里云通义千问 (DashScope API)
- **文档处理**: LangChain + Unstructured
- **检索算法**: 向量检索 + BM25 + 重排序
- **异步**: Uvicorn + 多进程

### 前端技术
- **框架**: Vue 3 + Composition API
- **构建工具**: Vue CLI / Webpack
- **UI组件**: Ant Design Vue
- **状态管理**: Vue 3 Reactivity
- **HTTP客户端**: Fetch API + 拦截器

### 开发工具
- **Python**: 3.11+
- **Node.js**: 16+
- **包管理**: pip + npm/yarn
- **版本控制**: Git

## 常见问题

### 1. 登录问题

**Token过期自动登出**：
- 系统会在token过期时自动跳转到登录页面
- 默认token有效期为24小时
- 可在后端代码中调整过期时间

### 2. 权限问题

**无法访问某些文档**：
- 检查用户角色和部门设置
- 普通员工只能访问本部门和公共文档
- 联系管理员调整权限

### 3. 文档上传失败

检查：
- 文件格式是否支持（PDF、Word、Excel、TXT、MD）
- 文件大小是否超过50MB
- 是否有上传权限（部门管理员或超级管理员）
- 磁盘空间是否充足

### 4. 查询无结果

可能原因：
- 相关文档未上传到系统
- 问题描述不够准确
- 相关性阈值设置过高（可调整RELEVANCE_THRESHOLD）
- 文档所在部门无访问权限

### 5. 数据库连接失败

确保：
- MySQL服务已启动
- 数据库连接信息正确
- 数据库用户有足够权限
- 网络连接正常

### 6. API调用失败

检查：
- 阿里云通义千问API密钥是否正确
- API额度是否充足
- 网络连接是否正常
- 是否正确配置了DASHSCOPE_API_KEY

## 部署说明

### 生产环境部署

1. **环境准备**
```bash
# 安装Python 3.11+
# 安装MySQL 5.7+
# 安装Node.js 16+
```

2. **后端部署**
```bash
# 克隆代码
git clone <repository-url>
cd hr-knowledge-base-backend

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件

# 初始化数据库
python database.py

# 启动服务（生产模式）
python main.py
```

3. **前端部署**
```bash
cd aiDemo
npm install
npm run build

# 将 dist 目录部署到 Web 服务器
# 或配置反向代理
```

4. **Nginx配置示例**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    # 前端静态文件
    location / {
        root /path/to/aiDemo/dist;
        try_files $uri $uri/ /index.html;
    }
    
    # 后端API代理
    location /api/ {
        proxy_pass http://127.0.0.1:8004;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 更新日志

### v3.0.0 (2025-01-27)

🎉 **重大版本更新**

**新增功能**：
- ✨ 混合检索系统（向量检索 + BM25 + 重排序）
- 🔍 查询重述和智能路由优化
- 📊 支持Excel文件处理
- 💬 聊天会话导出功能
- 🎨 优化前端界面和用户体验

**技术改进**：
- 🚀 升级到阿里云通义千问API
- ⚡ 优化检索精度和响应速度
- 🛡️ 增强安全性和错误处理
- 📈 改进文档处理和分块策略

**代码优化**：
- 🧹 清理临时调试脚本和迁移工具
- 📝 完善API文档和类型注解
- 🔧 统一配置管理和环境变量
- 📦 优化项目结构和依赖管理

### v2.0.0 (2024)
- 🎯 多部门权限管理系统
- 👥 用户角色和身份认证
- 💬 智能聊天会话管理
- 🎨 Vue3现代化前端界面

### v1.0.0 (2024)
- 🎯 基础文档管理和查询功能
- 🤖 本地模型集成
- 📁 简单文件上传和检索

## 许可证

MIT License

## 贡献指南

欢迎提交Issue和Pull Request！

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 联系方式

如有问题或建议，请提交Issue或联系开发团队。

---

**企业知识库系统 v3.0** - 让企业知识管理更智能、更安全、更高效！