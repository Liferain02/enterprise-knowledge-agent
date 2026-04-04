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

      <!-- 会话列表 -->
      <div class="session-list">
        <div class="session-list-header">
          <span>会话历史</span>
          <button class="btn-new-session" :disabled="!isAuthed" @click="createNewSession" title="新建会话">
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 5V19M5 12H19" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            </svg>
          </button>
        </div>
        <div class="sessions">
          <div
            v-for="session in sessions"
            :key="session.session_id"
            class="session-item"
            :class="{ active: session.session_id === sessionId }"
            @click="switchSession(session.session_id)"
          >
            <div class="session-title">{{ session.title }}</div>
            <div class="session-time">{{ formatTime(session.updated_at) }}</div>
            <div class="session-actions">
              <button class="btn-rename-session" @click.stop="renameSession(session.session_id, session.title)" title="重命名">
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                  <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
              </button>
              <button class="btn-delete-session" @click.stop="deleteSession(session.session_id)" title="删除会话">
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                </svg>
              </button>
            </div>
          </div>
          <div v-if="sessions.length === 0" class="no-sessions">
            暂无会话
          </div>
        </div>
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
          <button v-if="isAuthed" class="btn-logout" @click="logout">退出</button>
        </div>
      </header>
      
      <div v-if="!isAuthed" class="login-view">
        <div class="login-card">
          <div class="login-tabs">
            <button :class="{ active: loginMode === 'login' }" @click="loginMode = 'login'">登录</button>
            <button :class="{ active: loginMode === 'register' }" @click="loginMode = 'register'">注册</button>
          </div>

          <!-- 登录表单 -->
          <div v-if="loginMode === 'login'">
            <p class="login-desc">请输入账号密码后开始对话</p>
            <div class="login-form">
              <input v-model="loginUsername" class="login-input" placeholder="用户名" autocomplete="username" @keydown.enter="login" />
              <input v-model="loginPassword" class="login-input" placeholder="密码" type="password" autocomplete="current-password" @keydown.enter="login" />
              <button class="login-btn" @click="login" :disabled="loginLoading">登录</button>
              <div v-if="loginError" class="login-error">{{ loginError }}</div>
            </div>
          </div>

          <!-- 注册表单 -->
          <div v-else>
            <p class="login-desc">注册新账号，开始使用企业知识助手</p>
            <div class="login-form">
              <input v-model="regUsername" class="login-input" placeholder="用户名（3-32字符）" autocomplete="username" />
              <input v-model="regPassword" class="login-input" placeholder="密码（6-128字符）" type="password" autocomplete="new-password" />
              <input v-model="regPassword2" class="login-input" placeholder="确认密码" type="password" autocomplete="new-password" @keydown.enter="register" />
              <button class="login-btn" @click="register" :disabled="registerLoading">注册</button>
              <div v-if="registerError" class="login-error">{{ registerError }}</div>
              <div v-if="registerSuccess" class="login-success">{{ registerSuccess }}</div>
            </div>
          </div>
        </div>
      </div>

      <div v-else class="messages-container" ref="messagesContainer">
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
            <div class="message-body" v-html="renderMarkdown(msg.content)">            </div>
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
      
      <footer v-if="isAuthed" class="input-area">
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

interface Message {
  role: 'user' | 'assistant'
  content: string
  agent?: string
}

interface Session {
  session_id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

// 后端 API 基础路径（通过 Vite 代理到后端）
const API_BASE = '/api/v1'

// 登录态
const token = ref<string>(localStorage.getItem('eka_token') || '')
const loginUsername = ref('')
const loginPassword = ref('')
const loginError = ref('')
const loginLoading = ref(false)
const loginMode = ref<'login' | 'register'>('login')
const regUsername = ref('')
const regPassword = ref('')
const regPassword2 = ref('')
const registerError = ref('')
const registerSuccess = ref('')
const registerLoading = ref(false)

const isAuthed = computed(() => token.value.length > 0)

const applyAuthHeader = () => {
  if (token.value) {
    axios.defaults.headers.common['Authorization'] = `Bearer ${token.value}`
  } else {
    delete axios.defaults.headers.common['Authorization']
  }
}
applyAuthHeader()

const login = async () => {
  loginError.value = ''
  loginLoading.value = true
  try {
    const resp = await axios.post<{ access_token: string; token_type: string }>('/api/v1/auth/login', {
      username: loginUsername.value,
      password: loginPassword.value
    })
    token.value = resp.data.access_token
    localStorage.setItem('eka_token', token.value)
    applyAuthHeader()
    loginPassword.value = ''
    await checkConnection()
    await loadSessions()
    if (sessionId.value) {
      await loadHistory(sessionId.value)
    }
  } catch (e: any) {
    loginError.value = e?.response?.data?.detail || '登录失败'
  } finally {
    loginLoading.value = false
  }
}

const register = async () => {
  registerError.value = ''
  registerSuccess.value = ''
  if (regPassword.value !== regPassword2.value) {
    registerError.value = '两次密码不一致'
    return
  }
  registerLoading.value = true
  try {
    await axios.post('/api/v1/auth/register', {
      username: regUsername.value,
      password: regPassword.value
    })
    registerSuccess.value = '注册成功！请登录'
    // 清空表单并切换到登录
    regUsername.value = ''
    regPassword.value = ''
    regPassword2.value = ''
    loginMode.value = 'login'
    loginUsername.value = ''
    loginPassword.value = ''
  } catch (e: any) {
    registerError.value = e?.response?.data?.detail || '注册失败'
  } finally {
    registerLoading.value = false
  }
}

const logout = () => {
  token.value = ''
  localStorage.removeItem('eka_token')
  applyAuthHeader()
  apiConnected.value = false
  messages.value = []
  sessions.value = []
  loginMode.value = 'login'
  regUsername.value = ''
  regPassword.value = ''
  regPassword2.value = ''
  registerError.value = ''
  registerSuccess.value = ''
}

const sessionId = ref('default')
const inputMessage = ref('')
const messages = ref<Message[]>([])
const loading = ref(false)
const currentAgent = ref('supervisor')
const apiConnected = ref(false)
const messagesContainer = ref<HTMLElement | null>(null)
const sessions = ref<Session[]>([])
const partialAnswer = ref('')      // 流式累积中的部分回答
const partialAnswerKey = ref<number | null>(null)  // 流式消息在 messages 中的索引

// 用于取消请求
let abortController: AbortController | null = null

const chatTitle = computed(() => {
  const session = sessions.value.find(s => s.session_id === sessionId.value)
  if (session) {
    return session.title
  }
  return sessionId.value === 'default' ? '默认会话' : `会话: ${sessionId.value}`
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

const formatTime = (isoString: string) => {
  const date = new Date(isoString)
  const now = new Date()
  const diff = now.getTime() - date.getTime()
  const dayMs = 24 * 60 * 60 * 1000

  if (diff < dayMs) {
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  } else if (diff < 2 * dayMs) {
    return '昨天'
  } else {
    return date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })
  }
}

const loadSessions = async () => {
  try {
    const response = await axios.get<{ sessions: Session[] }>(`${API_BASE}/sessions`)
    sessions.value = response.data.sessions || []
  } catch (error) {
    console.error('加载会话列表失败:', error)
  }
}

const loadHistory = async (sid: string) => {
  try {
    const response = await axios.get(`${API_BASE}/history/${sid}`)
    if (response.data.messages) {
      messages.value = response.data.messages.map((m: any) => ({
        role: m.role,
        content: m.content,
        agent: m.metadata?.agent || m.agent
      }))
    } else {
      messages.value = []
    }
  } catch (error) {
    console.log('获取历史失败或无历史记录')
    messages.value = []
  }
}

const createNewSession = async () => {
  try {
    const response = await axios.post(`${API_BASE}/sessions`, {})
    const newSession = response.data
    sessionId.value = newSession.session_id
    messages.value = []
    await loadSessions()
  } catch (error) {
    console.error('创建会话失败:', error)
  }
}

const switchSession = async (sid: string) => {
  if (sid === sessionId.value) return
  sessionId.value = sid
}

const deleteSession = async (sid: string) => {
  if (!confirm('确定要删除这个会话吗？')) return

  try {
    await axios.delete(`${API_BASE}/sessions/${sid}`)
    await loadSessions()

    // 如果删除的是当前会话，切换到第一个会话或创建新会话
    if (sid === sessionId.value) {
      if (sessions.value.length > 0) {
        sessionId.value = sessions.value[0].session_id
      } else {
        await createNewSession()
      }
    }
  } catch (error) {
    console.error('删除会话失败:', error)
  }
}

const renameSession = async (sid: string, currentTitle: string) => {
  const newTitle = prompt('请输入新标题:', currentTitle)
  if (!newTitle || newTitle.trim() === '' || newTitle === currentTitle) return

  try {
    await axios.put(`${API_BASE}/sessions/${sid}/title`, { title: newTitle.trim() })
    await loadSessions()
  } catch (error) {
    console.error('重命名会话失败:', error)
    alert('重命名失败，请重试')
  }
}

const sendQuick = (text: string) => {
  inputMessage.value = text
  sendMessage()
}

const renderMarkdown = (content: string) => {
  // 使用 marked 解析 markdown
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
    if (partialAnswerKey.value !== null) {
      messages.value.push({
        role: 'assistant',
        content: '⏹️ 已手动取消回答',
        agent: currentAgent.value
      })
    }
    partialAnswerKey.value = null
    partialAnswer.value = ''
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

  // 创建占位消息用于流式更新
  const placeholderKey = messages.value.length
  const placeholderMsg = {
    role: 'assistant' as const,
    content: '',
    agent: currentAgent.value
  }
  messages.value.push(placeholderMsg)
  partialAnswerKey.value = placeholderKey
  partialAnswer.value = ''

  try {
    // 使用 SSE 流式输出
    const response = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token.value}`,
      },
      body: JSON.stringify({
        session_id: sessionId.value,
        message: text,
      }),
      signal: abortController.signal,
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }

    const reader = response.body!.getReader()
    const decoder = new TextDecoder()

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value, { stream: true })
      // 解析 SSE 行：data: {"type": "...", "data": "..."}
      for (const line of chunk.split('\n')) {
        if (!line.startsWith('data: ')) continue
        try {
          const event = JSON.parse(line.slice(6))
          if (event.type === 'llm_token' && event.data) {
            partialAnswer.value += event.data
            // 实时更新占位消息
            if (partialAnswerKey.value !== null) {
              messages.value[partialAnswerKey.value] = {
                role: 'assistant',
                content: partialAnswer.value,
                agent: currentAgent.value
              }
              scrollToBottom()
            }
          } else if (event.type === 'thinking' && event.data) {
            // 可选：显示思考状态（暂时静默处理）
          } else if (event.type === 'used_agent' && event.data) {
            currentAgent.value = event.data
          } else if (event.type === 'done') {
            // 流式完成，最终答案已通过 llm_token 累积
          }
        } catch {
          // 忽略解析错误（SSE 行可能不完整）
        }
      }
    }

    // 完成：正式提交消息
    partialAnswerKey.value = null
    partialAnswer.value = ''

    if (messages.value.length > 0 && messages.value[messages.value.length - 1].role === 'assistant') {
      messages.value[messages.value.length - 1].agent = currentAgent.value
    }

    // 刷新会话列表（更新标题和消息数）
    await loadSessions()
  } catch (error: any) {
    // 判断是否是主动取消
    if (error.name === 'AbortError' || abortController?.signal?.aborted) {
      console.log('请求已取消')
      if (partialAnswerKey.value !== null) {
        messages.value[partialAnswerKey.value].content = '⏹️ 已手动取消回答'
      }
    } else {
      console.error('请求失败:', error)
      messages.value.push({
        role: 'assistant',
        content: `❌ 请求失败: ${error.message || '未知错误'}`,
        agent: 'error'
      })
    }
  } finally {
    loading.value = false
    partialAnswerKey.value = null
    partialAnswer.value = ''
    abortController = null
    scrollToBottom()
  }
}

const checkConnection = async () => {
  if (!isAuthed.value) {
    apiConnected.value = false
    return
  }
  try {
    await axios.get(`${API_BASE}/health`, { timeout: 3000 })
    apiConnected.value = true
  } catch {
    apiConnected.value = false
  }
}

onMounted(async () => {
  // 页面加载时验证 token 是否有效
  if (isAuthed.value) {
    try {
      // 调用验证接口检查 token 是否有效
      await axios.get(`${API_BASE}/sessions`, { timeout: 3000 })
      await checkConnection()
      await loadSessions()
      await loadHistory(sessionId.value)
      setInterval(checkConnection, 30000)
    } catch (error) {
      // token 无效，清除登录状态
      console.log('Token 已过期，请重新登录')
      logout()
    }
  }
})

// 监听 session 变化，重新加载历史
watch(sessionId, async (newSid) => {
  await loadHistory(newSid)
})
</script>


