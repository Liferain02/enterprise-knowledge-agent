#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${PROJECT_DIR}/.run"
BACKEND_PID_FILE="${RUN_DIR}/后端.pid"
FRONTEND_PID_FILE="${RUN_DIR}/前端.pid"
BACKEND_LOG="${RUN_DIR}/后端.log"
FRONTEND_LOG="${RUN_DIR}/前端.log"

mkdir -p "${RUN_DIR}"

process_alive() {
    local pid="${1:-}"
    [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

read_live_pid() {
    local pid_file="$1"
    local pid=""
    [[ -f "${pid_file}" ]] && pid="$(<"${pid_file}")"
    if process_alive "${pid}"; then
        printf '%s' "${pid}"
        return 0
    fi
    rm -f "${pid_file}"
    return 1
}

port_in_use() {
    local port="$1"
    (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null
}

wait_for_url() {
    local url="$1"
    local max_seconds="$2"
    local attempts=$((max_seconds * 2))
    local i
    for ((i = 0; i < attempts; i++)); do
        if curl --noproxy '*' --fail --silent --max-time 2 "${url}" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

cleanup_started_processes() {
    local pid_file pid
    for pid_file in "${FRONTEND_PID_FILE}" "${BACKEND_PID_FILE}"; do
        pid=""
        [[ -f "${pid_file}" ]] && pid="$(<"${pid_file}")"
        if process_alive "${pid}"; then
            kill "${pid}" 2>/dev/null || true
        fi
        rm -f "${pid_file}"
    done
}

backend_pid="$(read_live_pid "${BACKEND_PID_FILE}" || true)"
frontend_pid="$(read_live_pid "${FRONTEND_PID_FILE}" || true)"
if [[ -n "${backend_pid}" || -n "${frontend_pid}" ]]; then
    echo "项目已有脚本管理的进程正在运行："
    [[ -n "${backend_pid}" ]] && echo "  后端 PID: ${backend_pid}"
    [[ -n "${frontend_pid}" ]] && echo "  前端 PID: ${frontend_pid}"
    echo "如需重启，请先执行：./scripts/stop.sh"
    exit 0
fi

if port_in_use 8010; then
    echo "启动失败：端口 8010 已被其他进程占用。"
    exit 1
fi
if port_in_use 3000; then
    echo "启动失败：端口 3000 已被其他进程占用。"
    exit 1
fi

if [[ ! -f "${PROJECT_DIR}/config/.env" ]]; then
    echo "启动失败：缺少 config/.env。请从 config/env.template 复制并填写安全配置。"
    exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "agent-demo" ]] && command -v python >/dev/null 2>&1; then
    BACKEND_PYTHON="$(command -v python)"
elif command -v conda >/dev/null 2>&1; then
    BACKEND_PYTHON="$(conda run -n agent-demo python -c 'import sys; print(sys.executable)' | tail -n 1)"
else
    echo "启动失败：未找到 Conda 的 agent-demo 环境。"
    exit 1
fi

FRONTEND_BIN="${PROJECT_DIR}/frontend/node_modules/.bin/vite"
if [[ ! -x "${FRONTEND_BIN}" ]]; then
    echo "启动失败：前端依赖尚未安装。请先进入 frontend 执行 npm install。"
    exit 1
fi

trap 'cleanup_started_processes; exit 1' INT TERM

echo "正在启动后端（端口 8010）..."
nohup "${BACKEND_PYTHON}" "${PROJECT_DIR}/main.py" >"${BACKEND_LOG}" 2>&1 </dev/null &
backend_pid=$!
printf '%s\n' "${backend_pid}" >"${BACKEND_PID_FILE}"

if ! wait_for_url "http://127.0.0.1:8010/health/ready" 45; then
    echo "后端启动失败，最近日志："
    tail -n 30 "${BACKEND_LOG}" || true
    cleanup_started_processes
    exit 1
fi

echo "正在启动前端（端口 3000）..."
nohup "${FRONTEND_BIN}" "${PROJECT_DIR}/frontend" --host 0.0.0.0 --port 3000 >"${FRONTEND_LOG}" 2>&1 </dev/null &
frontend_pid=$!
printf '%s\n' "${frontend_pid}" >"${FRONTEND_PID_FILE}"

if ! wait_for_url "http://127.0.0.1:3000/" 20; then
    echo "前端启动失败，最近日志："
    tail -n 30 "${FRONTEND_LOG}" || true
    cleanup_started_processes
    exit 1
fi

trap - INT TERM

ACCESS_HOST="127.0.0.1"
if command -v hostname >/dev/null 2>&1; then
    detected_host="$(hostname -I 2>/dev/null | awk '{print $1}')"
    [[ -n "${detected_host}" ]] && ACCESS_HOST="${detected_host}"
fi
SSH_HOST="$(hostname -s 2>/dev/null || hostname)"
SSH_USER="${USER:-你的用户名}"

echo
echo "项目启动成功："
echo "  服务器本机前端：http://127.0.0.1:3000"
echo "  服务器本机后端：http://127.0.0.1:8010/docs"
echo "  同一内网可尝试：http://${ACCESS_HOST}:3000"
echo
echo "从远程电脑访问时，请在远程电脑执行本地转发（-L）："
echo "  ssh -N -T -o ExitOnForwardFailure=yes -L 3000:127.0.0.1:3000 -L 8010:127.0.0.1:8010 ${SSH_USER}@${SSH_HOST}"
echo "然后在远程电脑打开：http://127.0.0.1:3000"
echo "  后端日志：${BACKEND_LOG}"
echo "  前端日志：${FRONTEND_LOG}"
echo "停止项目：./scripts/stop.sh"
