# Simple Homework Agent

A multi-turn homework tutoring agent designed for the CSIT5900 AI project at HKUST.

## 1. Setup Environment

It is recommended to use a virtual environment to manage dependencies.

```bash
# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate the environment
# For macOS/Linux:
source venv/bin/activate
# For Windows:
venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt
```


### 2. Setup Api

Create a .env file in the project root directory with your Azure OpenAI credentials. Ensure the endpoint and deployment name match your allocated resources.

```bash
AZURE_OPENAI_API_KEY=YOUR_API_KEY
AZURE_OPENAI_ENDPOINT="https://hkust.azure-api.net/"
AZURE_OPENAI_API_VERSION="2025-02-01-preview"
AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-5-mini"
```

### 3. Execution Instructions

# 1.Option A: Web Graphical Interface (Streamlit)

To launch the SmartTutor graphical interface in your browser:

```bash
streamlit run app.py
```

# 2.Option B: Terminal Mode
To run the agent directly in the command line:

```bash
python3 main.py
```