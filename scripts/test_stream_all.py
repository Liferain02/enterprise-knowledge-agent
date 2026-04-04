#!/usr/bin/env python
"""
继续测试
"""
import sys
import os
sys.path.insert(0, '/share/home/lifr/workspace/code/enterprise-knowledge-agent')
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'

# 继续测试 operation_agent 和 general_agent 的流式输出

import requests

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJMaWZlcmFpbiIsImlhdCI6MTc3MjY5MTY5MywiZXhwIjoxNzcyNzM0ODkzfQ.MWdiftxHPN2XxFeGqjmug_XHbgo3fphZPDCJDQsqYxo"

print("=== 测试 general_agent ===")
response = requests.post(
    "http://localhost:8000/api/v1/chat/stream",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOKEN}"
    },
    json={"message": "今天天气怎么样？", "session_id": "test-general"},
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))

print("\n=== 测试 operation_agent ===")
response = requests.post(
    "http://localhost:8000/api/v1/chat/stream",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOKEN}"
    },
    json={"message": "现在几点？", "session_id": "test-operation"},
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))

print("\n=== 测试带有历史session的对话 ===")
response = requests.post(
    "http://localhost:8000/api/v1/chat/stream",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOKEN}"
    },
    json={"message": "继续刚才的话题", "session_id": "test-general"},
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))

print("\n=== 测试新session（无历史） ===")
response = requests.post(
    "http://localhost:8000/api/v1/chat/stream",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOKEN}"
    },
    json={"message": "你好，请介绍一下自己", "session_id": "new-session"},
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
