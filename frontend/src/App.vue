<template>
  <div class="app-container">
    <!-- 侧边栏 -->
    <aside class="sidebar">
      <div class="logo">
        <div class="logo-icon">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M2 17L12 22L22 17" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M2 12L12 17L22 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
        <span>企业知识助手</span>
      </div>
      
      <div class="session-controls">
        <div class="session-selector">
          <label>会话ID</label>
          <input v-model="sessionId" type="text" placeholder="default" />
        </div>
        <button class="btn-clear" @click="clearHistory" :disabled="loading">
          🗑️ 清空对话
        </button>
      </div>
      
      <div class="agent-status">
        <div class="status-title">当前 Agent</div>
        <div class="status-item" :class="{ active: currentAgent === 'supervisor' }">
          <span class="status-dot supervisor"></span>
          路由调度
        </div>
        <div class="status-item" :class="{ active: currentAgent === 'knowledge_agent' }">
          <span class="status-dot knowledge"></span>
          知识检索
        </div>
        <div class="status-item" :class="{ active: currentAgent === 'operation_agent' }">
          <span class="status-dot operation"></span>
          任务执行
        </div>
        <div class="status-item" :class="{ active: currentAgent === 'general_agent' }">
          <span class="status-dot general"></span>
          闲聊问答
        </div>
      </div>
      
      <div class="quick-prompts">
        <div class="quick-title">💡 快捷问题</div>
        <button @click="sendQuick('请介绍一下公司的发展历程')">公司历史</button>
        <button @click="sendQuick('现在几点了？')">当前时间</button>
        <button @click="sendQuick('帮我计算 123 * 456 = ?')">数学计算</button>
        <button @click="sendQuick('查看 data/knowledge 目录')">文件列表</button>
      </div>
    </aside>

    <!-- 主聊天区 -->
    <main class="chat-main">
      <header class="chat-header">
        <h1>{{ chatTitle }}</h1>
        <div class="header-actions">
          <span class="connection-status" :class="{ connected: apiConnected }">
            {{ apiConnected ? '🟢 已连接' : '🔴 未连接' }}
          </span>
        </div>
      </header>
      
      <div class="messages-container" ref="messagesContainer">
        <div v-if="messages.length === 0" class="empty-state">
          <div class="empty-icon">💬</div>
          <h2>欢迎使用企业知识助手</h2>
          <p>我可以帮您：</p>
          <ul>
            <li>📚 检索企业知识库</li>
            <li>🕐 查询当前时间</li>
            <li>🔢 执行数学计算</li>
            <li>📁 查看文件系统</li>
          </ul>
        </div>
        
        <div
          v-for="(msg, index) in messages"
          :key="index"
          class="message"
          :class="msg.role"
        >
          <div class="message-avatar">
            <template v-if="msg.role === 'user'">👤</template>
            <template v-else>
              <div class="bot-avatar" :class="msg.agent">
                🤖
              </div>
            </template>
          </div>
          <div class="message-content">
            <div class="message-header">
              <span class="sender-name">
                {{ msg.role === 'user' ? '你' : getAgentName(msg.agent) }}
              </span>
              <span v-if="msg.agent" class="agent-badge" :class="msg.agent">
                {{ getAgentBadge(msg.agent) }}
              </span>
            </div>
            <div class="message-body" v-html="renderMarkdown(msg.content)"></div>
          </div>
        </div>
        
        <div v-if="loading" class="message assistant">
          <div class="message-avatar">
            <div class="bot-avatar typing">🤖</div>
          </div>
          <div class="message-content">
            <div class="message-header">
              <span class="sender-name">AI 助手</span>
              <span class="agent-badge" :class="currentAgent">{{ getAgentBadge(currentAgent) }}</span>
            </div>
            <div class="message-body">
              <div class="typing-indicator">
                <span></span><span></span><span></span>
              </div>
              正在思考中...
            </div>
          </div>
        </div>
      </div>
      
      <footer class="input-area">
        <div class="input-container">
          <textarea
            v-model="inputMessage"
            @keydown.enter.exact.prevent="sendMessage"
            placeholder="输入您的问题，按 Enter 发送..."
            rows="1"
            :disabled="loading"
          ></textarea>
          <button
            class="send-btn"
            @click="sendMessage"
            :disabled="!inputMessage.trim() && !loading"
            :class="{ 'is-loading': loading }"
          >
            <!-- 发送图标 -->
            <svg v-if="!loading" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M22 2L11 13" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <!-- 停止图标 -->
            <svg v-else viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/>
            </svg>
          </button>
        </div>
        <div class="input-hint">
          按 <kbd>Enter</kbd> 发送，<kbd>Shift + Enter</kbd> 换行
        </div>
      </footer>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick, onMounted, watch } from 'vue'
import axios from 'axios'
import { marked } from 'marked'
import hljs from 'highlight.js'

interface Message {
  role: 'user' | 'assistant'
  content: string
  agent?: string
}

const API_BASE = '/api/v1'

const sessionId = ref('default')
const inputMessage = ref('')
const messages = ref<Message[]>([])
const loading = ref(false)
const currentAgent = ref('supervisor')
const apiConnected = ref(false)
const messagesContainer = ref<HTMLElement | null>(null)

// 用于取消请求
let abortController: AbortController | null = null

const chatTitle = computed(() => {
  const session = sessionId.value || 'default'
  return session === 'default' ? '默认会话' : `会话: ${session}`
})

const getAgentName = (agent?: string) => {
  const map: Record<string, string> = {
    supervisor: '路由调度',
    knowledge_agent: '知识专家',
    operation_agent: '任务执行',
    general_agent: '闲聊助手'
  }
  return map[agent || ''] || 'AI 助手'
}

const getAgentBadge = (agent: string) => {
  const map: Record<string, string> = {
    supervisor: '调度',
    knowledge_agent: '知识',
    operation_agent: '执行',
    general_agent: '闲聊'
  }
  return map[agent] || agent
}

const sendQuick = (text: string) => {
  inputMessage.value = text
  sendMessage()
}

const renderMarkdown = (content: string) => {
  // 配置 marked
  marked.setOptions({
    highlight: function(code, lang) {
      if (lang && hljs.getLanguage(lang)) {
        return hljs.highlight(code, { language: lang }).value
      }
      return hljs.highlightAuto(code).value
    }
  })
  return marked.parse(content)
}

const scrollToBottom = () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

const sendMessage = async () => {
  // 如果正在加载，点击按钮则取消请求
  if (loading.value) {
    if (abortController) {
      abortController.abort()
      abortController = null
    }
    loading.value = false
    messages.value.push({
      role: 'assistant',
      content: '⏹️ 已手动取消回答',
      agent: currentAgent.value
    })
    return
  }

  const text = inputMessage.value.trim()
  if (!text) return

  // 创建新的 AbortController
  abortController = new AbortController()

  // 添加用户消息
  messages.value.push({ role: 'user', content: text })
  inputMessage.value = ''
  loading.value = true
  currentAgent.value = 'supervisor'
  scrollToBottom()

  try {
    const response = await axios.post(`${API_BASE}/chat`, {
      session_id: sessionId.value,
      message: text
    }, {
      signal: abortController.signal
    })

    const data = response.data
    
    // 添加 AI 回复
    messages.value.push({
      role: 'assistant',
      content: data.answer || data.final_answer || '无回复',
      agent: data.used_agent || 'unknown'
    })
    
    currentAgent.value = data.used_agent || 'supervisor'
  } catch (error: any) {
    // 判断是否是主动取消
    if (axios.isCancel(error)) {
      console.log('请求已取消')
      return
    }
    console.error('请求失败:', error)
    messages.value.push({
      role: 'assistant',
      content: `❌ 请求失败: ${error.response?.data?.detail || error.message}`,
      agent: 'error'
    })
  } finally {
    loading.value = false
    abortController = null
    scrollToBottom()
  }
}

const clearHistory = async () => {
  if (!confirm('确定要清空对话历史吗？')) return
  
  try {
    await axios.delete(`${API_BASE}/history`, {
      data: { session_id: sessionId.value }
    })
    messages.value = []
  } catch (error: any) {
    console.error('清空历史失败:', error)
  }
}

const checkConnection = async () => {
  try {
    await axios.get(`${API_BASE}/health`, { timeout: 3000 })
    apiConnected.value = true
  } catch {
    apiConnected.value = false
  }
}

onMounted(() => {
  checkConnection()
  // 定期检查连接状态
  setInterval(checkConnection, 30000)
})

// 监听 session 变化，重新加载历史
watch(sessionId, async () => {
  messages.value = []
  try {
    const response = await axios.get(`${API_BASE}/history/${sessionId.value}`)
    if (response.data.messages) {
      messages.value = response.data.messages.map((m: any) => ({
        role: m.role,
        content: m.content,
        agent: m.agent
      }))
    }
  } catch (error) {
    console.log('获取历史失败或无历史记录')
  }
})
</script>


