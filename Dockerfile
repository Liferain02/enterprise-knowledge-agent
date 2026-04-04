FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv（高性能 Python 包管理器）
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖（使用 uv）
RUN uv pip install --system --no-cache -r requirements.txt

# 复制应用代码
COPY . .

# 创建数据目录
RUN mkdir -p /app/data /app/chroma_db /app/logs

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${API_PORT:-8080}/health || exit 1

# 环境变量（默认值）
ENV PYTHONUNBUFFERED=1
ENV API_PORT=8080
ENV API_HOST=0.0.0.0
ENV ENVIRONMENT=production

# 运行
CMD ["python", "main.py"]
