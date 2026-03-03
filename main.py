"""
企业知识库智能助手 - 主入口
"""
import asyncio
import sys

# Windows 环境下启用 ProactorEventLoop 以支持子进程 - 必须在任何异步导入之前设置
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from config.settings import get_settings
from api.routes import chat, knowledge
from core.mcp_client import init_mcp, close_mcp
from rag.vectorstore import get_vectorstore_manager

# 初始化设置
settings = get_settings()


def check_and_ingest_knowledge_base():
    """启动时检查并嵌入知识库"""
    try:
        vectorstore_manager = get_vectorstore_manager()
        info = vectorstore_manager.get_collection_info()
        doc_count = info.get("count", 0)
        
        if doc_count == 0:
            print("=" * 50)
            print("⚠️  知识库为空，正在自动嵌入文档...")
            print("=" * 50)
            
            # 导入并运行嵌入脚本
            from scripts.ingest import ingest_knowledge_base
            ingest_knowledge_base(reset=False)
            
            print("✅ 知识库嵌入完成！")
        else:
            print(f"✅ 知识库已就绪，当前包含 {doc_count} 个文档块")
    except Exception as e:
        print(f"⚠️  检查知识库时出错: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时检查并嵌入知识库
    print("=" * 50)
    print("正在检查知识库状态...")
    check_and_ingest_knowledge_base()
    print("=" * 50)
    
    # 启动时初始化 MCP 服务器
    print("=" * 50)
    print("正在初始化 MCP 服务器...")
    
    try:
        await init_mcp()
    except asyncio.CancelledError:
        print("MCP 初始化被取消")
    except Exception as e:
        print(f"MCP 初始化出错（不影响运行）: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 50)
    
    yield
    
    # 关闭时断开 MCP 连接
    print("正在关闭 MCP 服务器连接...")
    try:
        await close_mcp()
    except Exception as e:
        print(f"关闭 MCP 出错: {e}")


# 创建FastAPI应用
app = FastAPI(
    title="企业知识库智能助手",
    description="基于 LangChain、LangGraph、ReAct 和 MCP 的企业级 RAG Agent 系统",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router)
app.include_router(knowledge.router)


# 根路由 - 返回简单界面
@app.get("/", response_class=HTMLResponse)
async def root():
    """根路径 - 返回简单的Web界面"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>企业知识库智能助手</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                width: 100%;
                max-width: 800px;
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 {
                font-size: 28px;
                margin-bottom: 10px;
            }
            .header p {
                opacity: 0.9;
                font-size: 14px;
            }
            .content {
                padding: 30px;
            }
            .chat-box {
                height: 400px;
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                overflow-y: auto;
                padding: 20px;
                margin-bottom: 20px;
                background: #f9f9f9;
            }
            .message {
                margin-bottom: 15px;
                padding: 12px 16px;
                border-radius: 10px;
                max-width: 80%;
            }
            .message.user {
                background: #667eea;
                color: white;
                margin-left: auto;
            }
            .message.assistant {
                background: #e8e8e8;
                color: #333;
            }
            .input-area {
                display: flex;
                gap: 10px;
            }
            input[type="text"] {
                flex: 1;
                padding: 15px;
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                font-size: 14px;
                outline: none;
            }
            input[type="text"]:focus {
                border-color: #667eea;
            }
            button {
                padding: 15px 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                font-size: 14px;
                transition: transform 0.2s;
            }
            button:hover {
                transform: scale(1.05);
            }
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            .loading {
                text-align: center;
                color: #999;
                padding: 10px;
            }
            .options {
                margin-top: 15px;
                display: flex;
                gap: 20px;
                justify-content: center;
            }
            .options label {
                display: flex;
                align-items: center;
                gap: 5px;
                font-size: 14px;
                color: #666;
            }
            .docs-link {
                display: block;
                text-align: center;
                margin-top: 20px;
                color: #667eea;
                text-decoration: none;
            }
            .docs-link:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 企业知识库智能助手</h1>
                <p>基于 LangChain + LangGraph + ReAct + MCP 构建</p>
            </div>
            <div class="content">
                <div class="chat-box" id="chatBox">
                    <div class="message assistant">
                        你好！我是企业知识库智能助手，可以帮助你回答关于公司规章制度、技术文档、FAQ等问题。请输入你的问题。
                    </div>
                </div>
                <div class="input-area">
                    <input type="text" id="messageInput" placeholder="请输入你的问题..." />
                    <button id="sendBtn" onclick="sendMessage()">发送</button>
                </div>
                <div class="options">
                    <label>
                        <input type="checkbox" id="useRag" checked />
                        使用RAG
                    </label>
                    <label>
                        <input type="checkbox" id="useReact" />
                        使用ReAct
                    </label>
                </div>
                <a href="/docs" class="docs-link">查看 API 文档 →</a>
            </div>
        </div>
        
        <script>
            const chatBox = document.getElementById('chatBox');
            const messageInput = document.getElementById('messageInput');
            const sendBtn = document.getElementById('sendBtn');
            const useRag = document.getElementById('useRag');
            const useReact = document.getElementById('useReact');
            
            messageInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            async function sendMessage() {
                const message = messageInput.value.trim();
                if (!message) return;
                
                // 添加用户消息
                addMessage(message, 'user');
                messageInput.value = '';
                
                // 显示加载状态
                const loadingDiv = document.createElement('div');
                loadingDiv.className = 'message assistant';
                loadingDiv.innerHTML = '<div class="loading">正在思考中...</div>';
                chatBox.appendChild(loadingDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
                
                sendBtn.disabled = true;
                
                try {
                    const response = await fetch('/api/v1/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            message: message,
                            use_rag: useRag.checked,
                            use_react: useReact.checked
                        })
                    });
                    
                    const data = await response.json();
                    
                    // 移除加载消息
                    loadingDiv.remove();
                    
                    // 添加助手回复
                    addMessage(data.answer, 'assistant');
                    
                } catch (error) {
                    loadingDiv.remove();
                    addMessage('抱歉，发生了错误: ' + error.message, 'assistant');
                }
                
                sendBtn.disabled = false;
            }
            
            function addMessage(content, role) {
                const div = document.createElement('div');
                div.className = 'message ' + role;
                div.textContent = content;
                chatBox.appendChild(div);
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    return html_content


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "service": "enterprise-knowledge-assistant",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    # Windows 下禁用 reload 模式，避免子进程事件循环问题
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False
    )


