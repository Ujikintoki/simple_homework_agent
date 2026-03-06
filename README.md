# Simple Homework Agent

A multi-turn homework tutoring agent designed for the CSIT5900 AI project at HKUST.

## Prerequisites

### 1. Setup Environment

Please create and activate a virtual environment, then install the required dependencies:

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. setup api
# 根目录新建.env文件：

AZURE_OPENAI_API_KEY=YOUR_API_KEY
AZURE_OPENAI_ENDPOINT="[https://hkust.azure-api.net/](https://hkust.azure-api.net/)"
AZURE_OPENAI_API_VERSION="2025-02-01-preview"
AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-5-mini"

### 3. test agent
# 虚拟环境中运行
```bash
python3 main.py
```
