# 🏘️ 李营村便民政策AI问答助手 — Streamlit 网页版

> 大学生暑期社会实践项目 · 河南省南阳市镇平县石佛寺镇李营村

一个面向李营村村民的 **政策AI问答助手**，基于 RAG（检索增强生成）技术，让村民用大白话提问，AI 根据本村政策文件给出准确回答。配套 **村地图导览** 功能，方便村民导航到村内各服务地点。

---

## ✨ 功能特点

- 📄 **基于本地政策文件**：上传本村的政策文档（txt/md），AI 仅基于文档回答
- 💬 **大白话问答**：村民用日常语言提问，AI 用通俗易懂的方式回答
- 🔒 **安全可靠**：不编造政策，无关问题自动拒绝
- 🗺️ **村地图导览**：列出村委、卫生所等地点，一键导航至高德地图
- 🌐 **网页链接**：给个链接就能用，无需安装任何软件
- 💰 **完全免费**：使用硅基流动免费 API，零成本运行
- 📂 **在线管理**：可在页面中上传新的政策文件

---

## 📋 项目结构

```
policy-qa/
├── streamlit_app.py       # 主程序（全部逻辑 + UI）
├── config.py              # 配置文件（村名、地点坐标、API Key 等）
├── requirements.txt       # Python 依赖
├── .streamlit/
│   └── secrets.toml       # API Key 配置模板（部署用）
├── README.md              # 本文件
├── policies/              # 政策文件目录
│   ├── 医保政策.txt
│   ├── 养老保险.txt
│   └── 种地补贴.txt
└── chroma_db/             # 向量数据库（自动生成）
```

---

## 🚀 本地快速启动

### 1. 安装依赖

```bash
cd policy-qa
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 启动应用

```bash
streamlit run streamlit_app.py
```

### 3. 打开浏览器

终端会显示一个本地地址，通常是 **http://localhost:8501**，在浏览器中打开即可使用。

---

## ☁️ 部署到 Hugging Face Spaces（推荐，免费）

### 第一步：将代码推送到 GitHub

1. 在 GitHub 上创建新仓库
2. 将本项目代码推送到该仓库
3. 确保仓库为 **公开（public）**（Hugging Face 免费计划需要公开仓库）

### 第二步：在 Hugging Face 创建 Space

1. 访问 https://huggingface.co/spaces 并登录/注册
2. 点击 **"Create new Space"**
3. 填写：
   - **Space Name**：`liying-policy-qa`
   - **License**：MIT
   - **SDK**：选择 **Streamlit**
   - **Space type**：Free（CPU 免费计划）
4. 在 **"Connect to GitHub"** 部分，选择你的 GitHub 仓库
5. 点击 **"Create Space"**

### 第三步：设置 API Key（Secrets）

1. 在 Space 页面，点击顶部 **"Settings"**
2. 在左侧找到 **"Repository Secrets"**（或直接搜索 secrets）
3. 点击 **"Add secret"**
4. Key：`SILICONFLOW_API_KEY`
5. Value：你的硅基流动 API Key（在 https://cloud.siliconflow.cn/account/ak 获取）
6. 点击 **"Save"**

### 第四步：自动部署

Space 会自动开始构建和部署，约 2-5 分钟后即可访问。

**你的链接就是：** `https://huggingface.co/spaces/你的用户名/liying-policy-qa`

把链接发给书记和村民，打开就能用！

---

## 🔑 获取免费 API Key

1. 访问硅基流动官网：https://cloud.siliconflow.cn
2. 注册账号（手机号即可）
3. 进入 **"账户管理" → "API密钥"**
4. 点击 **"新建API密钥"**，复制密钥
5. 本地开发：粘贴到 `config.py` 的 `API_KEY` 位置
6. 云部署：设置为 Secrets（见上一步）

---

## 📝 管理政策文件

### 方式一：在页面上传（最简单）

1. 打开网页
2. 在政策问答页面，点击 **"政策文件管理"**
3. 选择 `.txt` 或 `.md` 文件上传
4. 知识库会自动更新

### 方式二：直接放到文件夹

1. 将文件放到 `policies/` 目录下
2. 重启应用或重新上传即可

---

## ⚙️ 自定义配置

在 `config.py` 中可以修改：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `API_KEY` | 硅基流动 API 密钥 | 需填写 |
| `VILLAGE_NAME` | 村名 | 李营村 |
| `VILLAGE_FULL` | 完整地址 | 河南省... |
| `VILLAGE_LOCATIONS` | 村内地点坐标 | 5 个地点 |
| `QUICK_QUESTIONS` | 预设快捷问题 | 5 个 |

---

## 🛠️ 技术栈

- **框架**：Streamlit（前后端合一）
- **向量库**：Chroma（嵌入式，无需安装数据库）
- **大模型**：Qwen2.5-7B-Instruct（通过硅基流动免费 API 调用）
- **Embedding**：BAAI/bge-m3（中文向量模型）
- **导航**：高德地图 URI API

---

## 📜 免责声明

本工具基于 AI 大模型技术，回答仅供参考，**具体政策执行以村委会解释为准**。

本项目为大学生暑期社会实践项目，使用开源免费技术构建。

---

_大学生社会实践助力乡村振兴 🌾_
