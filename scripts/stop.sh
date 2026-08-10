#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${PROJECT_DIR}/.run"

process_alive() {
    local pid="${1:-}"
    [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

stop_process() {
    local name="$1"
    local pid_file="$2"
    local pid=""

    if [[ ! -f "${pid_file}" ]]; then
        echo "${name}未由启动脚本运行。"
        return 0
    fi

    pid="$(<"${pid_file}")"
    if ! process_alive "${pid}"; then
        echo "${name}进程已停止，清理过期 PID 文件。"
        rm -f "${pid_file}"
        return 0
    fi

    local command_line
    command_line="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
    if [[ "${command_line}" != *"${PROJECT_DIR}"* ]]; then
        echo "拒绝停止 ${name}：PID ${pid} 不属于当前项目。"
        return 1
    fi

    echo "正在停止${name}（PID ${pid}）..."
    kill "${pid}"

    local i
    for ((i = 0; i < 20; i++)); do
        if ! process_alive "${pid}"; then
            rm -f "${pid_file}"
            echo "${name}已停止。"
            return 0
        fi
        sleep 0.5
    done

    echo "${name}未在 10 秒内退出，执行强制停止。"
    kill -KILL "${pid}" 2>/dev/null || true
    rm -f "${pid_file}"
}

stop_process "前端" "${RUN_DIR}/前端.pid"
stop_process "后端" "${RUN_DIR}/后端.pid"

echo "项目已停止。日志保留在 ${RUN_DIR}。"
