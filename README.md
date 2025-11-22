# 📚 AI 小说创作助手 - 基于 LangGraph 的智能写作平台
一个基于 LangGraph、LangChain 和 FastAPI 构建的智能小说创作辅助系统，通过 RAG 工作流帮助作家轻松创作高质量小说内容。

# ✨ 核心特性
🤖 AI 辅助创作 - 基于大语言模型的智能内容生成

📖 项目化管理 - 完整的小说项目管理体系

🔄 循环反馈工作流 - 带自我优化的章节生成流程

🏷️ 智能实体抽取 - 自动分析角色关系和故事要素

🌐 现代化 Web 界面 - 直观易用的创作环境

# 🛠️ 技术架构
核心技术栈
LangGraph - 工作流编排与状态管理

LangChain - LLM 应用开发框架

FastAPI - 高性能 Web 后端

Ollama - 本地 LLM 服务

现代前端 - 响应式 Web 界面

模型支持
ollama 模型
qwen3:8b - 主要生成模型

nomic-embed-text - 嵌入模型

# 核心流程
<img width="784" height="497" alt="image" src="https://github.com/user-attachments/assets/89383d76-4500-4b2e-b03b-8e87b14c8845" />

# 🚀 快速开始
## 环境配置
**创建 Conda 环境**

**conda create -n novel_ai python=3.10**

**conda activate novel_ai**

**pip install -r requirements.txt**

## 前往ollama.com下载ollama

**ollama run qwen3:8b**

**ollama run nomic-embed-text:latest**

**然后直接启动 main.py**

**python main.py**

**进入localhost:8000/index.html**

# 📖 使用指南

## 1. 项目创建
   
<img width="1867" height="982" alt="image" src="https://github.com/user-attachments/assets/ebd2e778-653a-4b0b-8564-d19de236f97f" />

## 2. 智能项目配置

   
<img width="1864" height="998" alt="image" src="https://github.com/user-attachments/assets/c0c4f778-fbf4-4086-bdee-01b2d3e9faff" />

**在左侧文本框输入自己的需求，AI帮你完善你的项目配置**

<img width="1862" height="992" alt="image" src="https://github.com/user-attachments/assets/514b3b91-43f7-47d7-aada-edbe36f1fac6" />

## 3. 进入创作空间
   
**项目创建完成之后，点击完成创建回到主页面，找到自己创建的项目，并点击进入**

<img width="1850" height="990" alt="image" src="https://github.com/user-attachments/assets/e120f179-57ce-48a4-84d1-1b622348d320" />

<img width="1854" height="981" alt="image" src="https://github.com/user-attachments/assets/695edab9-d9f8-489b-a198-9b638d219055" />

## 4. 章节生成
   
**在创作需求出输入你该章节的创作需求，点击生成章节指令，AI会帮你生成本章章节生成指令。    
（你也可以自定义章节生成指令）**

<img width="1855" height="980" alt="image" src="https://github.com/user-attachments/assets/e92cac49-73b1-4dc3-9d0f-6c9eddec5fda" />

## 5. 工作流执行
   
**章节指令生成之后，点击生成章节内容，AI会帮你生成本章节内容，由于生成章节的过程是一个带循环反馈的workflow，因此这个过程会比较漫长，
你可以在控制台查看章节生成过程：**

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/14b62554-877a-48de-a5ab-9094a01e1ce1" />

**生成结束后，你可以自由修改生成的内容，如果不满意生成内容的话，可以修改创作指令重新生成，可以在章节大纲中加入更详细的需求，例如：字数不能少于1000字。**

<img width="1851" height="954" alt="image" src="https://github.com/user-attachments/assets/eb70dbdc-70f2-48e7-bc17-cb926a8b629e" />

## 6. 内容优化与发布
   
**若满意生成内容点击发布，发布过程中，AI会分析本章节内容，抽取实体及其关系，并进行保存**

<img width="1080" height="430" alt="image" src="https://github.com/user-attachments/assets/58702874-1429-48e5-b6bf-a88b889d6f27" />

<img width="1309" height="740" alt="image" src="https://github.com/user-attachments/assets/6d1462d4-c981-4ffc-9c00-1977314a6712" />

<img width="1486" height="818" alt="image" src="https://github.com/user-attachments/assets/3cb6e2de-4124-4de5-b850-ce41468bfb6a" />

## 7. 章节管理
    
**有章节发布之后可以在页面左侧的目录查看已经生成的章节信息，点击具体的章节可以查看章节内容**

<img width="1638" height="844" alt="image" src="https://github.com/user-attachments/assets/f955e3aa-87df-4bf0-ae7f-ff243dcb520c" />

<img width="1632" height="845" alt="image" src="https://github.com/user-attachments/assets/bdcfa338-2ba7-4c34-8216-b9a53510dd5f" />



