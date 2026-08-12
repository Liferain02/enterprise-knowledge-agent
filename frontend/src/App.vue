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
        <span>实验室智能助手</span>
      </div>

      <nav v-if="isAuthed" class="workspace-nav">
        <button :class="{ active: activeWorkspace === 'chat' }" @click="activeWorkspace = 'chat'">
          <span>对话助手</span>
          <small>检索、问答与任务执行</small>
        </button>
        <button :class="{ active: activeWorkspace === 'knowledge' }" @click="openKnowledgeWorkspace">
          <span>资料中心</span>
          <small>浏览、筛选与维护资料</small>
        </button>
        <button :class="{ active: activeWorkspace === 'research' }" @click="openResearchWorkspace">
          <span>科研空间</span>
          <small>项目、成员与实验记录</small>
        </button>
      </nav>

      <!-- 会话列表 -->
      <div v-if="activeWorkspace === 'chat'" class="session-list">
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

      <div v-if="activeWorkspace === 'chat'" class="agent-status">
        <div class="status-title">当前 Agent</div>
        <div class="status-item" :class="{ active: currentAgent === 'planner' }">
          <span class="status-dot planner"></span>
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

      <div v-if="activeWorkspace === 'chat'" class="quick-prompts">
        <div class="quick-title">快捷问题</div>
        <button @click="sendQuick('我刚加入实验室应该先看什么？')">新人导览</button>
        <button @click="sendQuick('环境配置相关资料有哪些？')">环境配置</button>
        <button @click="sendQuick('最近的组会记录里提到了哪些任务？')">组会记录</button>
        <button @click="sendQuick('帮我计算 123 * 456 = ?')">数学计算</button>
      </div>

      <div v-if="isAuthed && activeWorkspace === 'knowledge'" class="search-panel">
        <div class="quick-title">资料检索</div>
        <input
          v-model="searchForm.query"
          class="upload-input"
          placeholder="检索论文、组会、FAQ..."
          @keydown.enter="searchKnowledge"
        />
        <select v-model="searchForm.docType" class="upload-input">
          <option value="">全部类型</option>
          <option value="lab_policy">规章制度</option>
          <option value="project_doc">项目资料</option>
          <option value="paper_note">论文笔记</option>
          <option value="env_setup">环境配置</option>
          <option value="meeting_minutes">组会记录</option>
          <option value="faq">FAQ</option>
          <option value="experiment_log">实验记录</option>
          <option value="onboarding">新人导览</option>
        </select>
        <button class="upload-btn" @click="searchKnowledge">搜索资料</button>
        <div v-if="searchStatus" class="upload-status">{{ searchStatus }}</div>
        <div v-if="searchResults.length" class="mini-search-results">
          <div v-for="(result, resultIndex) in searchResults" :key="resultIndex" class="mini-search-card">
            <div class="mini-search-title">{{ result.title }}</div>
            <div class="mini-search-meta">
              {{ formatDocType(result.doc_type) }}<span v-if="result.project_name"> · {{ result.project_name }}</span>
            </div>
          </div>
        </div>
      </div>

      <div v-if="isAuthed && activeWorkspace === 'knowledge' && canManageKnowledge" class="upload-panel">
        <div class="quick-title">资料上传</div>
        <input class="upload-input" type="file" @change="handleFileChange" />
        <input v-model="uploadForm.title" class="upload-input" placeholder="资料标题（可选）" />
        <div class="upload-grid">
          <select v-model="uploadForm.docType" class="upload-input">
            <option value="lab_policy">规章制度</option>
            <option value="project_doc">项目资料</option>
            <option value="paper_note">论文笔记</option>
            <option value="env_setup">环境配置</option>
            <option value="meeting_minutes">组会记录</option>
            <option value="faq">FAQ</option>
            <option value="experiment_log">实验记录</option>
            <option value="onboarding">新人导览</option>
            <option value="general">通用资料</option>
          </select>
          <select v-model="uploadForm.visibility" class="upload-input">
            <option value="public">公共资料</option>
            <option value="project">项目组内</option>
            <option value="restricted">负责人可见</option>
          </select>
        </div>
        <input v-model="uploadForm.author" class="upload-input" placeholder="作者" />
        <input v-model="uploadForm.projectName" class="upload-input" placeholder="项目名" />
        <input v-model="uploadForm.researchDirection" class="upload-input" placeholder="研究方向" />
        <input v-model="uploadForm.tags" class="upload-input" placeholder="标签，逗号分隔" />
        <button class="upload-btn" :disabled="uploading" @click="uploadKnowledge">
          {{ uploading ? '上传中...' : '上传资料' }}
        </button>
        <button class="upload-btn secondary" :disabled="seedingSamples" @click="seedLabSamples">
          {{ seedingSamples ? '导入中...' : '一键导入实验室样例资料' }}
        </button>
        <div v-if="uploadStatus" class="upload-status">{{ uploadStatus }}</div>
      </div>

      <div v-if="isAuthed && activeWorkspace === 'knowledge'" class="ingestion-panel">
        <div class="quick-title">入库任务</div>
        <div class="ingestion-stats">
          <span>排队 {{ ingestionStats.pending || 0 }}</span>
          <span>处理中 {{ ingestionStats.running || 0 }}</span>
          <span>完成 {{ ingestionStats.completed || 0 }}</span>
          <span>失败 {{ ingestionStats.failed || 0 }}</span>
        </div>
        <div v-if="ingestionJobs.length" class="ingestion-list">
          <div v-for="job in ingestionJobs.slice(0, 6)" :key="job.job_id" class="ingestion-item">
            <div class="ingestion-item-top">
              <span>{{ job.filename }}</span>
              <b :class="`job-${job.status}`">{{ formatJobStatus(job.status) }}</b>
            </div>
            <small v-if="job.status === 'completed'">
              {{ job.result?.stored_chunks || 0 }} 个片段 · {{ job.result?.elapsed_seconds || 0 }}s
            </small>
            <small v-else-if="job.error">{{ job.error }}</small>
            <small v-else>{{ formatDocType(job.doc_type) }}</small>
          </div>
        </div>
        <div v-else class="upload-status">暂无入库任务</div>
      </div>

      <div v-if="isAuthed && activeWorkspace === 'knowledge'" class="feedback-panel">
        <div class="quick-title">知识缺口</div>
        <div class="feedback-stats">
          <span>总计 {{ feedbackStats.total }}</span>
          <span>有帮助 {{ feedbackStats.helpful }}</span>
          <span>不准确 {{ feedbackStats.incorrect }}</span>
          <span>缺资料 {{ feedbackStats.missing_material }}</span>
        </div>
        <div class="issue-status-tabs">
          <button :class="{ active: feedbackIssueStatus === 'open' }" @click="setFeedbackIssueStatus('open')">待处理</button>
          <button :class="{ active: feedbackIssueStatus === 'resolved' }" @click="setFeedbackIssueStatus('resolved')">已解决</button>
        </div>
        <div v-if="feedbackIssues.length" class="issue-list">
          <div v-for="issue in feedbackIssues" :key="issue.id" class="issue-card">
            <div class="issue-type">{{ issue.feedback_type === 'incorrect' ? '不准确' : '缺资料' }}</div>
            <div v-if="canReviewFeedback" class="issue-owner">来自 {{ issue.username }}</div>
            <div class="issue-question">{{ issue.question }}</div>
            <div v-if="issue.comment" class="issue-comment">{{ issue.comment }}</div>
            <div v-if="issue.resolution_note" class="issue-resolution">
              {{ issue.resolution_note }}<span v-if="issue.resolved_by"> · {{ issue.resolved_by }}</span>
            </div>
            <button
              v-if="canReviewFeedback"
              class="issue-action"
              @click="updateFeedbackIssue(issue, issue.status === 'open' ? 'resolved' : 'open')"
            >
              {{ issue.status === 'open' ? '标记已解决' : '重新打开' }}
            </button>
          </div>
        </div>
        <div v-else class="issue-empty">
          {{ feedbackIssueStatus === 'open' ? '暂无待处理问题' : '暂无已解决问题' }}
        </div>
      </div>

      <div v-if="isAuthed && activeWorkspace === 'research'" class="research-side-panel">
        <div class="quick-title">科研工作台</div>
        <div class="feedback-stats">
          <span>可见项目 {{ researchOverview.projects }}</span>
          <span>实验记录 {{ researchOverview.experiments }}</span>
          <span>活跃项目 {{ researchOverview.active_projects }}</span>
          <span>未完成待办 {{ researchOverview.open_tasks }}</span>
          <span>协作成员 {{ researchOverview.members }}</span>
        </div>
        <button v-if="canManageKnowledge" class="upload-btn" @click="showProjectForm = !showProjectForm">
          {{ showProjectForm ? '收起项目表单' : '新建项目空间' }}
        </button>
        <button v-if="canManageKnowledge" class="upload-btn secondary" @click="seedResearchSamples">
          初始化科研样例
        </button>
      </div>
    </aside>

    <!-- 主聊天区 -->
    <main class="chat-main">
      <header class="chat-header">
        <h1>{{ workspaceTitle }}</h1>
        <div class="header-actions">
          <!-- 用户信息区域 -->
          <div v-if="currentUser" class="user-info">
            <div class="user-avatar" :class="currentUser.role">
              {{ currentUser.username?.charAt(0)?.toUpperCase() || '?' }}
            </div>
            <div class="user-details">
              <span class="user-name">{{ currentUser.username }}</span>
              <span class="role-badge" :class="currentUser.role">
                {{ currentUser.role_display || currentUser.role }}
              </span>
              <span v-if="currentUser.department_name" class="dept-name">
                {{ currentUser.department_name }}
              </span>
            </div>
          </div>
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
            <p class="login-desc">注册新账号，开始使用实验室智能助手</p>
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

      <section v-if="isAuthed && activeWorkspace === 'knowledge'" class="knowledge-workspace">
        <div class="knowledge-hero">
          <div>
            <span class="eyebrow">LAB KNOWLEDGE WORKSPACE</span>
            <h2>课题组资料中心</h2>
            <p>集中浏览实验室制度、项目资料、论文笔记、组会纪要与环境配置。目录只展示当前账号有权访问的资料。</p>
          </div>
          <button class="refresh-btn" :disabled="knowledgeLoading" @click="loadKnowledgeWorkspace">
            {{ knowledgeLoading ? '刷新中...' : '刷新目录' }}
          </button>
        </div>

        <div class="overview-grid">
          <div class="overview-card accent">
            <span>资料总数</span>
            <strong>{{ knowledgeOverview.documents }}</strong>
          </div>
          <div class="overview-card">
            <span>检索片段</span>
            <strong>{{ knowledgeOverview.chunks }}</strong>
          </div>
          <div class="overview-card">
            <span>项目数量</span>
            <strong>{{ knowledgeOverview.projects }}</strong>
          </div>
          <div class="overview-card">
            <span>公共资料</span>
            <strong>{{ knowledgeOverview.public_documents }}</strong>
          </div>
        </div>

        <div class="library-toolbar">
          <input v-model="libraryFilters.query" placeholder="按标题、作者、项目或方向筛选" @keydown.enter="loadKnowledgeDocuments" />
          <select v-model="libraryFilters.docType" @change="loadKnowledgeDocuments">
            <option value="">全部类型</option>
            <option v-for="item in docTypeOptions" :key="item.value" :value="item.value">{{ item.label }}</option>
          </select>
          <select v-model="libraryFilters.visibility" @change="loadKnowledgeDocuments">
            <option value="">全部范围</option>
            <option value="public">公共资料</option>
            <option value="project">项目组内</option>
            <option value="restricted">负责人可见</option>
          </select>
          <button @click="loadKnowledgeDocuments">筛选</button>
        </div>

        <div v-if="knowledgeLoading" class="library-empty">正在加载资料目录...</div>
        <div v-else-if="knowledgeDocuments.length === 0" class="library-empty">当前筛选条件下暂无资料。</div>
        <div v-else class="document-grid">
          <article v-for="doc in knowledgeDocuments" :key="doc.source" class="document-card">
            <div class="document-card-top">
              <span class="source-type">{{ doc.doc_type_label || formatDocType(doc.doc_type) }}</span>
              <span class="visibility-badge" :class="doc.visibility">{{ formatVisibility(doc.visibility) }}</span>
            </div>
            <h3>{{ doc.title }}</h3>
            <p>{{ doc.summary || '该资料暂未填写简介，可通过资料检索或对话助手查看具体内容。' }}</p>
            <div class="document-meta">
              <span v-if="doc.author">{{ doc.author }}</span>
              <span v-if="doc.project_name">{{ doc.project_name }}</span>
              <span>{{ doc.chunk_count }} 个片段</span>
            </div>
            <div class="document-footer">
              <span>{{ doc.source }}</span>
              <button v-if="canManageKnowledge" class="document-delete" @click="deleteKnowledgeDocument(doc)">删除</button>
            </div>
          </article>
        </div>
      </section>

      <section v-else-if="isAuthed && activeWorkspace === 'research'" class="research-workspace">
        <div class="knowledge-hero research-hero">
          <div>
            <span class="eyebrow">RESEARCH OPERATIONS</span>
            <h2>科研项目空间</h2>
            <p>把研究计划、项目成员、环境、代码版本、数据集与实验结论放进同一条可追踪链路。</p>
          </div>
          <button class="refresh-btn" :disabled="researchLoading" @click="loadResearchWorkspace">
            {{ researchLoading ? '刷新中...' : '刷新项目' }}
          </button>
        </div>

        <div class="overview-grid">
          <div class="overview-card accent"><span>可见项目</span><strong>{{ researchOverview.projects }}</strong></div>
          <div class="overview-card"><span>实验记录</span><strong>{{ researchOverview.experiments }}</strong></div>
          <div class="overview-card"><span>活跃项目</span><strong>{{ researchOverview.active_projects }}</strong></div>
          <div class="overview-card"><span>未完成待办</span><strong>{{ researchOverview.open_tasks }}</strong></div>
        </div>

        <form v-if="showProjectForm" class="research-form" @submit.prevent="createResearchProject">
          <div class="research-form-heading">
            <div><span class="eyebrow">NEW PROJECT</span><h3>建立项目空间</h3></div>
            <button type="button" class="document-delete" @click="showProjectForm = false">关闭</button>
          </div>
          <div class="research-form-grid">
            <input v-model="projectForm.title" required placeholder="项目名称，例如 Distributed NUMA" />
            <input v-model="projectForm.researchDirection" placeholder="研究方向，例如 高性能网络" />
            <input v-model="projectForm.lead" placeholder="负责人账号，默认当前用户" />
            <select v-model="projectForm.visibility">
              <option value="project">项目成员可见</option>
              <option value="public">全实验室可见</option>
              <option value="restricted">负责人可见</option>
            </select>
          </div>
          <textarea v-model="projectForm.summary" placeholder="项目目标、当前阶段和评测重点"></textarea>
          <input v-model="projectForm.members" placeholder="成员账号，逗号分隔" />
          <button class="refresh-btn" type="submit">创建项目</button>
          <span v-if="projectFormError" class="login-error">{{ projectFormError }}</span>
        </form>

        <div class="research-layout">
          <div class="project-list">
            <div class="library-toolbar project-toolbar">
              <input v-model="projectQuery" placeholder="按项目名称、方向或简介筛选" @keydown.enter="loadResearchProjects" />
              <button @click="loadResearchProjects">筛选</button>
            </div>
            <div v-if="researchLoading" class="library-empty">正在加载科研项目...</div>
            <div v-else-if="researchProjects.length === 0" class="library-empty">当前账号暂无可见项目。</div>
            <template v-else>
              <article
                v-for="project in researchProjects"
                :key="project.id"
                class="project-card"
                :class="{ active: selectedProject?.id === project.id }"
                @click="selectResearchProject(project)"
              >
                <div class="document-card-top">
                  <span class="source-type">{{ project.research_direction || '科研项目' }}</span>
                  <span class="visibility-badge" :class="project.visibility">{{ formatVisibility(project.visibility) }}</span>
                </div>
                <h3>{{ project.title }}</h3>
                <p>{{ project.summary || '该项目暂未填写简介。' }}</p>
                <div class="project-card-foot">
                  <span>{{ formatProjectStatus(project.status) }}</span>
                  <span>{{ project.members.length }} 人 · {{ project.experiment_count }} 次实验</span>
                </div>
              </article>
            </template>
          </div>

          <div class="experiment-board">
            <div v-if="!selectedProject" class="library-empty">选择一个项目，查看结构化实验记录。</div>
            <template v-else>
              <div class="experiment-board-heading">
                <div>
                  <span class="eyebrow">{{ selectedProject.slug }}</span>
                  <h2>{{ selectedProject.title }}</h2>
                  <p>{{ selectedProject.lead }} 负责 · {{ selectedProject.members.map(item => item.username).join('、') }}</p>
                </div>
                <button class="refresh-btn" @click="showExperimentForm = !showExperimentForm">
                  {{ showExperimentForm ? '收起表单' : '记录实验' }}
                </button>
              </div>

              <form v-if="showExperimentForm" class="research-form compact" @submit.prevent="createExperiment">
                <div class="research-form-grid">
                  <input v-model="experimentForm.title" required placeholder="实验标题" />
                  <select v-model="experimentForm.status">
                    <option value="planned">待执行</option>
                    <option value="running">进行中</option>
                    <option value="completed">已完成</option>
                    <option value="failed">失败</option>
                  </select>
                  <input v-model="experimentForm.codeCommit" placeholder="Git commit" />
                  <input v-model="experimentForm.datasetVersion" placeholder="数据集 / workload 版本" />
                </div>
                <textarea v-model="experimentForm.environment" placeholder="环境：节点、CPU、NIC、OS、关键参数"></textarea>
                <textarea v-model="experimentForm.hypothesis" placeholder="实验假设"></textarea>
                <input v-model="experimentForm.metrics" placeholder='指标 JSON，例如 {"p99_us": 4.8, "bandwidth_gbps": 91.2}' />
                <textarea v-model="experimentForm.conclusion" placeholder="实验结论"></textarea>
                <textarea v-model="experimentForm.nextSteps" placeholder="后续动作"></textarea>
                <button class="refresh-btn" type="submit">保存实验记录</button>
                <span v-if="experimentFormError" class="login-error">{{ experimentFormError }}</span>
              </form>

              <section class="task-board">
                <div class="experiment-board-heading">
                  <div>
                    <span class="eyebrow">MEETING ACTIONS</span>
                    <h3>组会行动项</h3>
                  </div>
                  <span>{{ researchTasks.filter(item => item.status !== 'done').length }} 项待推进</span>
                </div>
                <div class="task-extractor">
                  <textarea
                    v-model="meetingTaskContent"
                    placeholder="- [ ] 补齐 RDMA 延迟矩阵 | 负责人: alice | 截止: 2026-06-10"
                  ></textarea>
                  <button class="refresh-btn" @click="extractMeetingTasks">从纪要提取待办</button>
                </div>
                <div v-if="researchTasks.length === 0" class="library-empty">尚无组会行动项。</div>
                <article v-for="task in researchTasks" :key="task.id" class="task-card" :class="task.status">
                  <div>
                    <h4>{{ task.title }}</h4>
                    <p>
                      <span v-if="task.assignee">负责人 {{ task.assignee }}</span>
                      <span v-if="task.due_date">截止 {{ task.due_date }}</span>
                      <span v-if="task.source">来源 {{ task.source }}</span>
                    </p>
                  </div>
                  <select :value="task.status" @change="updateResearchTask(task, $event)">
                    <option value="open">待处理</option>
                    <option value="in_progress">进行中</option>
                    <option value="done">已完成</option>
                  </select>
                </article>
              </section>

              <div v-if="researchExperiments.length === 0" class="library-empty">尚无实验记录。</div>
              <article v-for="experiment in researchExperiments" :key="experiment.id" class="experiment-card">
                <div class="document-card-top">
                  <span class="experiment-status" :class="experiment.status">{{ formatExperimentStatus(experiment.status) }}</span>
                  <time>{{ formatTimestamp(experiment.updated_at) }}</time>
                </div>
                <h3>{{ experiment.title }}</h3>
                <p v-if="experiment.hypothesis"><b>假设：</b>{{ experiment.hypothesis }}</p>
                <div class="experiment-facts">
                  <span v-if="experiment.code_commit">commit {{ experiment.code_commit }}</span>
                  <span v-if="experiment.dataset_version">{{ experiment.dataset_version }}</span>
                  <span v-for="(value, key) in experiment.metrics" :key="key">{{ key }}: {{ value }}</span>
                </div>
                <p v-if="experiment.conclusion"><b>结论：</b>{{ experiment.conclusion }}</p>
                <p v-if="experiment.next_steps"><b>下一步：</b>{{ experiment.next_steps }}</p>
              </article>
            </template>
          </div>
        </div>
      </section>

      <div v-else-if="isAuthed" class="messages-container" ref="messagesContainer">
        <div v-if="messages.length === 0" class="empty-state">
          <div class="empty-icon">💬</div>
          <h2>欢迎使用实验室智能助手</h2>
          <p>我可以帮您：</p>
          <ul>
            <li>📚 检索实验室知识资料</li>
            <li>🧭 给出新人入组导览</li>
            <li>📎 展示来源文档和证据片段</li>
            <li>🕐 查询当前时间</li>
            <li>🔢 执行数学计算</li>
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
            <div v-if="msg.sources?.length" class="source-list">
              <div class="source-list-title">引用来源</div>
              <div v-for="(source, sourceIndex) in msg.sources" :key="`${index}-${sourceIndex}`" class="source-card">
                <div class="source-card-header">
                  <span class="source-title">{{ source.title }}</span>
                  <span class="source-type">{{ formatDocType(source.doc_type) }}</span>
                </div>
                <div class="source-meta">
                  <span v-if="source.author">{{ source.author }}</span>
                  <span v-if="source.project_name">{{ source.project_name }}</span>
                  <span v-if="source.created_at">{{ source.created_at }}</span>
                </div>
                <div class="source-snippet">{{ source.snippet }}</div>
              </div>
            </div>
            <div v-if="msg.role === 'assistant'" class="feedback-actions">
              <button class="feedback-btn" :disabled="msg.feedbackSubmitted" @click="submitFeedback(msg, 'helpful')">有帮助</button>
              <button class="feedback-btn" :disabled="msg.feedbackSubmitted" @click="submitFeedback(msg, 'incorrect')">不准确</button>
              <button class="feedback-btn" :disabled="msg.feedbackSubmitted" @click="submitFeedback(msg, 'missing_material')">缺少资料</button>
            </div>
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
      
      <footer v-if="isAuthed && activeWorkspace === 'chat'" class="input-area">
        <div class="input-container">
          <textarea
            v-model="inputMessage"
            @keydown.enter.exact.prevent="sendMessage"
            placeholder="输入实验室问题，按 Enter 发送..."
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
import { ref, computed, nextTick, onMounted, onUnmounted, watch } from 'vue'
import axios from 'axios'
import { marked } from 'marked'

interface SourceItem {
  title: string
  snippet: string
  doc_type: string
  author?: string
  project_name?: string
  research_direction?: string
  created_at?: string
  score?: number | null
  source?: string
  visibility?: string
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  agent?: string
  sources?: SourceItem[]
  question?: string
  feedbackSubmitted?: boolean
}

interface Session {
  session_id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

interface KnowledgeDocument {
  source: string
  title: string
  doc_type: string
  doc_type_label: string
  author?: string
  project_name?: string
  research_direction?: string
  visibility: string
  created_at?: string
  summary?: string
  chunk_count: number
}

interface IngestionJob {
  job_id: string
  filename: string
  category: string
  doc_type: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'retrying'
  retry_count: number
  max_retries: number
  error?: string
  result?: { stored_chunks?: number; elapsed_seconds?: number; source?: string }
}

interface ProjectMember {
  username: string
  member_role: string
  created_at: number
}

interface ResearchProject {
  id: string
  slug: string
  title: string
  summary: string
  research_direction: string
  status: 'planned' | 'active' | 'paused' | 'completed'
  visibility: 'public' | 'project' | 'restricted'
  lead: string
  members: ProjectMember[]
  experiment_count: number
  open_task_count: number
}

interface ResearchExperiment {
  id: string
  project_id: string
  title: string
  hypothesis: string
  environment: string
  code_commit: string
  dataset_version: string
  metrics: Record<string, string | number | boolean>
  conclusion: string
  next_steps: string
  status: 'planned' | 'running' | 'completed' | 'failed'
  created_by: string
  updated_at: number
}

interface ResearchTask {
  id: string
  project_id: string
  title: string
  assignee: string
  due_date: string
  status: 'open' | 'in_progress' | 'done'
  source: string
}

interface FeedbackIssue {
  id: number
  username: string
  session_id: string
  feedback_type: 'incorrect' | 'missing_material'
  question: string
  comment?: string
  status: 'open' | 'resolved'
  resolution_note?: string
  resolved_by?: string
  resolved_at?: string
  created_at: string
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

// 当前登录用户信息（从后端 /me 或登录响应获取）
interface CurrentUser {
  username: string
  role: string
  role_display: string
  department: string
  department_name: string
  department_path: string
  permission_hint: string
}
const currentUser = ref<CurrentUser | null>(null)
const knowledgeFile = ref<File | null>(null)
const uploadForm = ref({
  title: '',
  docType: 'project_doc',
  author: '',
  projectName: '',
  researchDirection: '',
  visibility: 'public',
  tags: '',
})
const uploading = ref(false)
const uploadStatus = ref('')
const feedbackStats = ref({
  total: 0,
  helpful: 0,
  incorrect: 0,
  missing_material: 0,
})
const feedbackIssues = ref<FeedbackIssue[]>([])
const feedbackIssueStatus = ref<'open' | 'resolved'>('open')
const searchForm = ref({
  query: '',
  docType: '',
})
const searchResults = ref<SourceItem[]>([])
const searchStatus = ref('')
const seedingSamples = ref(false)
const activeWorkspace = ref<'chat' | 'knowledge' | 'research'>('chat')
const knowledgeLoading = ref(false)
const knowledgeDocuments = ref<KnowledgeDocument[]>([])
const knowledgeOverview = ref({
  documents: 0,
  chunks: 0,
  projects: 0,
  public_documents: 0,
  restricted_documents: 0,
  by_doc_type: {} as Record<string, number>,
  by_visibility: {} as Record<string, number>,
})
const libraryFilters = ref({
  query: '',
  docType: '',
  visibility: '',
})
const docTypeOptions = [
  { value: 'lab_policy', label: '规章制度' },
  { value: 'project_doc', label: '项目资料' },
  { value: 'paper_note', label: '论文笔记' },
  { value: 'env_setup', label: '环境配置' },
  { value: 'meeting_minutes', label: '组会记录' },
  { value: 'faq', label: 'FAQ' },
  { value: 'experiment_log', label: '实验记录' },
  { value: 'onboarding', label: '新人导览' },
]
const ingestionJobs = ref<IngestionJob[]>([])
const ingestionStats = ref<Record<string, number>>({})
let ingestionPollTimer: ReturnType<typeof setInterval> | null = null
const knownCompletedJobs = new Set<string>()
const researchLoading = ref(false)
const researchProjects = ref<ResearchProject[]>([])
const researchExperiments = ref<ResearchExperiment[]>([])
const researchTasks = ref<ResearchTask[]>([])
const selectedProject = ref<ResearchProject | null>(null)
const projectQuery = ref('')
const showProjectForm = ref(false)
const showExperimentForm = ref(false)
const projectFormError = ref('')
const experimentFormError = ref('')
const researchOverview = ref({
  projects: 0,
  experiments: 0,
  open_tasks: 0,
  members: 0,
  active_projects: 0,
  by_status: {} as Record<string, number>,
})
const projectForm = ref({
  title: '',
  researchDirection: '',
  summary: '',
  lead: '',
  members: '',
  visibility: 'project',
})
const experimentForm = ref({
  title: '',
  hypothesis: '',
  environment: '',
  codeCommit: '',
  datasetVersion: '',
  metrics: '',
  conclusion: '',
  nextSteps: '',
  status: 'planned',
})
const meetingTaskContent = ref('')

const isAuthed = computed(() => token.value.length > 0)
const canManageKnowledge = computed(() => {
  const role = currentUser.value?.role || 'student'
  return ['admin', 'pi', 'teacher', 'lab_admin', 'senior_student', 'editor', 'manager', 'hr', 'it_support'].includes(role)
})
const canReviewFeedback = computed(() => canManageKnowledge.value)

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
    const resp = await axios.post<{
      access_token: string
      token_type: string
      user_info?: {
        username: string
        role: string
        department: string
        department_name: string
        department_path: string
      }
    }>('/api/v1/auth/login', {
      username: loginUsername.value,
      password: loginPassword.value
    })
    token.value = resp.data.access_token
    localStorage.setItem('eka_token', token.value)
    applyAuthHeader()

    // 解析并保存用户信息
    const ui = resp.data.user_info
    if (ui) {
      const roleDisplayMap: Record<string, string> = {
        admin: '管理员',
        pi: '导师/PI',
        teacher: '教师',
        lab_admin: '实验室管理员',
        senior_student: '高年级成员',
        student: '研究生',
        assistant: '助研/本科生',
        editor: '资料维护者',
        viewer: '普通成员',
        manager: '项目负责人',
        hr: '实验室管理员',
        it_support: '平台支持',
        employee: '研究组成员',
      }
      const hintMap: Record<string, string> = {
        admin: '您可管理全部实验室资料与用户。',
        pi: '您可查看公共、项目组内和负责人可见资料。',
        teacher: '您可查看公共与项目组内资料。',
        lab_admin: '您可维护公共流程与资料入口。',
        senior_student: '您可查看公共与项目组内资料，并维护部分项目资料。',
        student: '您可查看实验室公共资料。',
        assistant: '您可查看公共资料与新人导览内容。',
        editor: '您可维护实验室公共与项目资料。',
        viewer: '您可查看实验室公共资料。',
        manager: '您可查看公共与项目组内资料。',
        hr: '您可维护公共流程资料。',
        it_support: '您可维护环境配置与平台说明资料。',
        employee: '您可查看实验室公共资料。',
      }
      currentUser.value = {
        username: ui.username,
        role: ui.role || 'student',
        role_display: roleDisplayMap[ui.role || 'student'] || ui.role || '研究生',
        department: ui.department || '',
        department_name: ui.department_name || '',
        department_path: ui.department_path || '',
        permission_hint: hintMap[ui.role || 'student'] || '',
      }
      localStorage.setItem('eka_current_user', JSON.stringify(currentUser.value))
    }

    loginPassword.value = ''
    await checkConnection()
    await loadSessions()
    await loadFeedbackStats()
    await loadFeedbackIssues()
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
  localStorage.removeItem('eka_current_user')
  applyAuthHeader()
  apiConnected.value = false
  messages.value = []
  sessions.value = []
  currentUser.value = null
  feedbackStats.value = { total: 0, helpful: 0, incorrect: 0, missing_material: 0 }
  feedbackIssues.value = []
  feedbackIssueStatus.value = 'open'
  searchResults.value = []
  searchStatus.value = ''
  activeWorkspace.value = 'chat'
  knowledgeDocuments.value = []
  knowledgeOverview.value = {
    documents: 0,
    chunks: 0,
    projects: 0,
    public_documents: 0,
    restricted_documents: 0,
    by_doc_type: {},
    by_visibility: {},
  }
  ingestionJobs.value = []
  ingestionStats.value = {}
  researchProjects.value = []
  researchExperiments.value = []
  researchTasks.value = []
  selectedProject.value = null
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
const currentAgent = ref('planner')
const apiConnected = ref(false)
const messagesContainer = ref<HTMLElement | null>(null)
const sessions = ref<Session[]>([])
const partialAnswer = ref('')      // 流式累积中的部分回答
const partialAnswerKey = ref<number | null>(null)  // 流式消息在 messages 中的索引
const pendingVersionSource = ref('')  // SSE 中收到的版本溯源（等待 done 后追加到消息）
const pendingSources = ref<SourceItem[]>([])

// 用于取消请求
let abortController: AbortController | null = null

const chatTitle = computed(() => {
  const session = sessions.value.find(s => s.session_id === sessionId.value)
  if (session) {
    return session.title
  }
  return sessionId.value === 'default' ? '默认会话' : `会话: ${sessionId.value}`
})
const workspaceTitle = computed(() => {
  if (activeWorkspace.value === 'knowledge') return '资料中心'
  if (activeWorkspace.value === 'research') return '科研空间'
  return chatTitle.value
})

const getAgentName = (agent?: string) => {
  const map: Record<string, string> = {
    planner: '路由调度',
    knowledge_agent: '资料检索',
    operation_agent: '任务执行',
    general_agent: '学术助手'
  }
  return map[agent || ''] || 'AI 助手'
}

const getAgentBadge = (agent: string) => {
  const map: Record<string, string> = {
    planner: '调度',
    knowledge_agent: '资料',
    operation_agent: '执行',
    general_agent: '辅助'
  }
  return map[agent] || agent
}

const formatDocType = (docType?: string) => {
  const map: Record<string, string> = {
    lab_policy: '规章制度',
    project_doc: '项目资料',
    paper_note: '论文笔记',
    env_setup: '环境配置',
    meeting_minutes: '组会记录',
    faq: 'FAQ',
    experiment_log: '实验记录',
    onboarding: '新人导览',
    general: '通用资料',
  }
  return map[docType || 'general'] || docType || '资料'
}

const formatVisibility = (visibility?: string) => {
  const map: Record<string, string> = {
    public: '公共',
    project: '项目组内',
    restricted: '负责人可见',
  }
  return map[visibility || 'public'] || visibility || '公共'
}

const formatProjectStatus = (status: ResearchProject['status']) => {
  return { planned: '规划中', active: '进行中', paused: '已暂停', completed: '已完成' }[status]
}

const formatExperimentStatus = (status: ResearchExperiment['status']) => {
  return { planned: '待执行', running: '进行中', completed: '已完成', failed: '失败' }[status]
}

const formatTimestamp = (timestamp: number) => {
  return new Date(timestamp * 1000).toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })
}

const formatJobStatus = (status: IngestionJob['status']) => {
  const map: Record<IngestionJob['status'], string> = {
    pending: '排队中',
    running: '处理中',
    retrying: '等待重试',
    completed: '已完成',
    failed: '失败',
  }
  return map[status]
}

const loadIngestionJobs = async () => {
  try {
    const response = await axios.get(`${API_BASE}/knowledge/ingestion/jobs`, { params: { limit: 20 } })
    ingestionJobs.value = response.data.jobs || []
    ingestionStats.value = response.data.stats || {}
    const newlyCompleted = ingestionJobs.value.filter(
      job => job.status === 'completed' && !knownCompletedJobs.has(job.job_id)
    )
    ingestionJobs.value
      .filter(job => job.status === 'completed')
      .forEach(job => knownCompletedJobs.add(job.job_id))
    if (newlyCompleted.length && activeWorkspace.value === 'knowledge') {
      await Promise.all([loadKnowledgeOverview(), loadKnowledgeDocuments()])
    }
  } catch (error) {
    console.error('加载入库任务失败:', error)
  }
}

const loadKnowledgeDocuments = async () => {
  knowledgeLoading.value = true
  try {
    const response = await axios.get(`${API_BASE}/knowledge/documents`, {
      params: {
        query: libraryFilters.value.query || undefined,
        doc_type: libraryFilters.value.docType || undefined,
        visibility: libraryFilters.value.visibility || undefined,
      },
    })
    knowledgeDocuments.value = response.data.documents || []
  } catch (error) {
    console.error('加载资料目录失败:', error)
  } finally {
    knowledgeLoading.value = false
  }
}

const loadKnowledgeOverview = async () => {
  try {
    const response = await axios.get(`${API_BASE}/knowledge/overview`)
    knowledgeOverview.value = response.data
  } catch (error) {
    console.error('加载资料概览失败:', error)
  }
}

const loadKnowledgeWorkspace = async () => {
  await Promise.all([
    loadKnowledgeOverview(),
    loadKnowledgeDocuments(),
    loadIngestionJobs(),
    loadFeedbackStats(),
    loadFeedbackIssues(),
  ])
}

const openKnowledgeWorkspace = async () => {
  activeWorkspace.value = 'knowledge'
  await loadKnowledgeWorkspace()
}

const loadResearchOverview = async () => {
  const response = await axios.get(`${API_BASE}/research/overview`)
  researchOverview.value = response.data
}

const loadResearchProjects = async () => {
  const response = await axios.get(`${API_BASE}/research/projects`, {
    params: { query: projectQuery.value || undefined },
  })
  researchProjects.value = response.data.projects || []
  if (selectedProject.value) {
    selectedProject.value =
      researchProjects.value.find(item => item.id === selectedProject.value?.id) || null
  }
}

const loadResearchWorkspace = async () => {
  researchLoading.value = true
  try {
    await Promise.all([loadResearchOverview(), loadResearchProjects()])
    if (selectedProject.value) {
      await Promise.all([
        loadResearchExperiments(selectedProject.value.id),
        loadResearchTasks(selectedProject.value.id),
      ])
    }
  } finally {
    researchLoading.value = false
  }
}

const openResearchWorkspace = async () => {
  activeWorkspace.value = 'research'
  await loadResearchWorkspace()
}

const selectResearchProject = async (project: ResearchProject) => {
  selectedProject.value = project
  showExperimentForm.value = false
  await Promise.all([loadResearchExperiments(project.id), loadResearchTasks(project.id)])
}

const loadResearchExperiments = async (projectId: string) => {
  const response = await axios.get(`${API_BASE}/research/projects/${projectId}/experiments`)
  researchExperiments.value = response.data.experiments || []
}

const loadResearchTasks = async (projectId: string) => {
  const response = await axios.get(`${API_BASE}/research/projects/${projectId}/tasks`)
  researchTasks.value = response.data.tasks || []
}

const extractMeetingTasks = async () => {
  if (!selectedProject.value || !meetingTaskContent.value.trim()) return
  await axios.post(`${API_BASE}/research/projects/${selectedProject.value.id}/tasks/extract`, {
    content: meetingTaskContent.value,
    source: '粘贴的组会纪要',
  })
  meetingTaskContent.value = ''
  await loadResearchWorkspace()
}

const updateResearchTask = async (task: ResearchTask, event: Event) => {
  const status = (event.target as HTMLSelectElement).value
  await axios.patch(`${API_BASE}/research/tasks/${task.id}`, { status })
  await loadResearchWorkspace()
}

const resetProjectForm = () => {
  projectForm.value = {
    title: '', researchDirection: '', summary: '', lead: '', members: '', visibility: 'project',
  }
}

const createResearchProject = async () => {
  projectFormError.value = ''
  try {
    await axios.post(`${API_BASE}/research/projects`, {
      title: projectForm.value.title,
      research_direction: projectForm.value.researchDirection,
      summary: projectForm.value.summary,
      lead: projectForm.value.lead || undefined,
      members: projectForm.value.members.split(',').map(item => item.trim()).filter(Boolean),
      visibility: projectForm.value.visibility,
    })
    resetProjectForm()
    showProjectForm.value = false
    await loadResearchWorkspace()
  } catch (error: any) {
    projectFormError.value = error?.response?.data?.detail || '创建项目失败'
  }
}

const seedResearchSamples = async () => {
  await axios.post(`${API_BASE}/research/seed-samples`)
  await loadResearchWorkspace()
}

const resetExperimentForm = () => {
  experimentForm.value = {
    title: '', hypothesis: '', environment: '', codeCommit: '', datasetVersion: '',
    metrics: '', conclusion: '', nextSteps: '', status: 'planned',
  }
}

const createExperiment = async () => {
  if (!selectedProject.value) return
  experimentFormError.value = ''
  let metrics = {}
  try {
    metrics = experimentForm.value.metrics.trim() ? JSON.parse(experimentForm.value.metrics) : {}
  } catch {
    experimentFormError.value = '指标必须是合法 JSON 对象'
    return
  }
  try {
    await axios.post(`${API_BASE}/research/projects/${selectedProject.value.id}/experiments`, {
      title: experimentForm.value.title,
      hypothesis: experimentForm.value.hypothesis,
      environment: experimentForm.value.environment,
      code_commit: experimentForm.value.codeCommit,
      dataset_version: experimentForm.value.datasetVersion,
      metrics,
      conclusion: experimentForm.value.conclusion,
      next_steps: experimentForm.value.nextSteps,
      status: experimentForm.value.status,
    })
    resetExperimentForm()
    showExperimentForm.value = false
    await loadResearchWorkspace()
  } catch (error: any) {
    experimentFormError.value = error?.response?.data?.detail || '保存实验记录失败'
  }
}

const deleteKnowledgeDocument = async (doc: KnowledgeDocument) => {
  if (!confirm(`确定删除资料“${doc.title}”吗？该操作会移除对应检索片段。`)) return
  try {
    await axios.delete(`${API_BASE}/knowledge/documents`, { params: { source: doc.source } })
    await loadKnowledgeWorkspace()
  } catch (error: any) {
    alert(error?.response?.data?.detail || '删除资料失败')
  }
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

const loadFeedbackStats = async () => {
  try {
    const response = await axios.get(`${API_BASE}/feedback/stats`)
    feedbackStats.value = response.data
  } catch (error) {
    console.error('加载反馈统计失败:', error)
  }
}

const loadFeedbackIssues = async () => {
  try {
    const response = await axios.get(`${API_BASE}/feedback/issues`, {
      params: { limit: 8, status: feedbackIssueStatus.value },
    })
    feedbackIssues.value = response.data.issues || []
  } catch (error) {
    console.error('加载反馈问题清单失败:', error)
  }
}

const setFeedbackIssueStatus = async (status: 'open' | 'resolved') => {
  feedbackIssueStatus.value = status
  await loadFeedbackIssues()
}

const updateFeedbackIssue = async (
  issue: FeedbackIssue,
  status: 'open' | 'resolved'
) => {
  let resolutionNote = ''
  if (status === 'resolved') {
    resolutionNote = window.prompt('请说明如何解决（例如：已补充哪份资料）', '')?.trim() || ''
    if (!resolutionNote) return
  }
  try {
    await axios.patch(`${API_BASE}/feedback/issues/${issue.id}`, {
      status,
      resolution_note: resolutionNote || undefined,
    })
    await Promise.all([loadFeedbackStats(), loadFeedbackIssues()])
  } catch (error: any) {
    window.alert(error?.response?.data?.detail || '更新反馈状态失败')
  }
}

const loadHistory = async (sid: string) => {
  try {
    const response = await axios.get(`${API_BASE}/history/${sid}`)
    if (response.data.messages) {
      let lastUserQuestion = ''
      messages.value = response.data.messages.map((m: any) => {
        if (m.role === 'user') {
          lastUserQuestion = m.content
        }
        return {
          role: m.role,
          content: m.content,
          agent: m.metadata?.agent || m.agent,
          sources: Array.isArray(m.metadata?.sources) ? m.metadata.sources : [],
          question: m.role === 'assistant' ? lastUserQuestion : undefined,
          feedbackSubmitted: false,
        }
      })
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

const handleFileChange = (event: Event) => {
  const target = event.target as HTMLInputElement
  knowledgeFile.value = target.files?.[0] || null
}

const resetUploadForm = () => {
  knowledgeFile.value = null
  uploadForm.value = {
    title: '',
    docType: 'project_doc',
    author: '',
    projectName: '',
    researchDirection: '',
    visibility: 'public',
    tags: '',
  }
}

const uploadKnowledge = async () => {
  if (!knowledgeFile.value) {
    uploadStatus.value = '请先选择文件'
    return
  }

  const formData = new FormData()
  formData.append('file', knowledgeFile.value)
  formData.append('category', uploadForm.value.docType)
  formData.append('doc_type', uploadForm.value.docType)
  formData.append('title', uploadForm.value.title)
  formData.append('author', uploadForm.value.author)
  formData.append('project_name', uploadForm.value.projectName)
  formData.append('research_direction', uploadForm.value.researchDirection)
  formData.append('visibility', uploadForm.value.visibility)
  formData.append('tags', uploadForm.value.tags)

  uploading.value = true
  uploadStatus.value = ''
  try {
    const resp = await axios.post('/api/v1/knowledge/add/file', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    const job = resp.data.job as IngestionJob
    uploadStatus.value = resp.data.duplicate
      ? `资料内容已存在：${job.filename}（${formatJobStatus(job.status)}）`
      : `已提交 ${job.filename}，任务 ${job.job_id.slice(0, 8)} 正在${formatJobStatus(job.status)}`
    resetUploadForm()
    await loadIngestionJobs()
  } catch (error: any) {
    uploadStatus.value = error?.response?.data?.detail || '上传失败'
  } finally {
    uploading.value = false
  }
}

const seedLabSamples = async () => {
  seedingSamples.value = true
  try {
    const response = await axios.post('/api/v1/knowledge/seed-lab-samples')
    uploadStatus.value = `${response.data.message}，共导入 ${response.data.inserted_files} 份资料`
    await loadKnowledgeWorkspace()
  } catch (error: any) {
    uploadStatus.value = error?.response?.data?.detail || '导入样例资料失败'
  } finally {
    seedingSamples.value = false
  }
}

const searchKnowledge = async () => {
  if (!searchForm.value.query.trim()) {
    searchStatus.value = '请输入检索词'
    return
  }

  searchStatus.value = ''
  try {
    const response = await axios.post('/api/v1/knowledge/search', {
      query: searchForm.value.query,
      top_k: 5,
      doc_type: searchForm.value.docType || undefined,
    })
    searchResults.value = (response.data.results || []).map((item: any) => ({
      title: item.metadata?.title || item.metadata?.document_title || item.metadata?.source || '未命名资料',
      snippet: String(item.content || '').slice(0, 160),
      doc_type: item.metadata?.doc_type || item.metadata?.category || 'general',
      author: item.metadata?.author,
      project_name: item.metadata?.project_name,
      research_direction: item.metadata?.research_direction,
      created_at: item.metadata?.created_at,
      source: item.metadata?.source,
      visibility: item.metadata?.visibility,
      score: item.score,
    }))
    searchStatus.value = `命中 ${searchResults.value.length} 条资料`
  } catch (error: any) {
    searchStatus.value = error?.response?.data?.detail || '检索失败'
  }
}

const submitFeedback = async (
  msg: Message,
  feedbackType: 'helpful' | 'incorrect' | 'missing_material'
) => {
  if (msg.feedbackSubmitted || !msg.question) return

  const comment =
    feedbackType === 'helpful'
      ? ''
      : window.prompt(
          feedbackType === 'incorrect' ? '哪里不准确？' : '还缺少哪些资料？',
          ''
        ) || ''

  try {
    await axios.post('/api/v1/feedback', {
      session_id: sessionId.value,
      question: msg.question,
      answer: msg.content,
      used_agent: msg.agent || 'unknown',
      feedback_type: feedbackType,
      comment,
    })
    msg.feedbackSubmitted = true
    await loadFeedbackStats()
    await loadFeedbackIssues()
  } catch (error) {
    console.error('提交反馈失败:', error)
  }
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
  currentAgent.value = 'planner'
  pendingSources.value = []
  pendingVersionSource.value = ''
  scrollToBottom()

  // 创建占位消息用于流式更新
  const placeholderKey = messages.value.length
  const placeholderMsg = {
    role: 'assistant' as const,
    content: '',
    agent: currentAgent.value,
    sources: [] as SourceItem[],
    question: text,
    feedbackSubmitted: false,
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
                agent: currentAgent.value,
                sources: pendingSources.value,
                question: text,
                feedbackSubmitted: false,
              }
              scrollToBottom()
            }
          } else if (event.type === 'thinking' && event.data) {
            // 可选：显示思考状态（暂时静默处理）
          } else if (event.type === 'used_agent' && event.data) {
            currentAgent.value = event.data
          } else if (event.type === 'user_profile' && event.data) {
            // 更新当前用户信息（优先使用服务端最新信息）
            currentUser.value = event.data
            localStorage.setItem('eka_current_user', JSON.stringify(event.data))
          } else if (event.type === 'sources' && event.data) {
            pendingSources.value = Array.isArray(event.data) ? event.data : []
          } else if (event.type === 'version_source' && event.data) {
            // 缓存版本溯源，等待流结束后追加到消息
            pendingVersionSource.value = event.data
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
      const lastMsg = messages.value[messages.value.length - 1]
      lastMsg.agent = currentAgent.value
      lastMsg.sources = pendingSources.value
      // 追加版本溯源信息到消息末尾
      if (pendingVersionSource.value) {
        lastMsg.content += '\n\n' + pendingVersionSource.value
        pendingVersionSource.value = ''
      }
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
        agent: 'error',
        question: text,
        feedbackSubmitted: false,
      })
    }
  } finally {
    loading.value = false
    partialAnswerKey.value = null
    partialAnswer.value = ''
    pendingVersionSource.value = ''  // 重置版本溯源缓存
    pendingSources.value = []
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
  // 页面加载时恢复用户信息
  const savedUser = localStorage.getItem('eka_current_user')
  if (savedUser) {
    try {
      currentUser.value = JSON.parse(savedUser)
    } catch {
      localStorage.removeItem('eka_current_user')
    }
  }

  // 页面加载时验证 token 是否有效
  if (isAuthed.value) {
    try {
      // 调用验证接口检查 token 是否有效，同时获取最新用户信息
      await axios.get(`${API_BASE}/sessions`, { timeout: 3000 })
      // 如果没有缓存的用户信息，调用 /me 获取
      if (!currentUser.value) {
        await fetchCurrentUser()
      }
      await checkConnection()
      await loadSessions()
      await loadFeedbackStats()
      await loadFeedbackIssues()
      await loadIngestionJobs()
      await loadHistory(sessionId.value)
      setInterval(checkConnection, 30000)
      ingestionPollTimer = setInterval(() => {
        if (activeWorkspace.value === 'knowledge') loadIngestionJobs()
      }, 4000)
    } catch (error) {
      // token 无效，清除登录状态
      console.log('Token 已过期，请重新登录')
      logout()
    }
  }
})

onUnmounted(() => {
  if (ingestionPollTimer) clearInterval(ingestionPollTimer)
})

// 获取当前用户详情（调用 /me 接口）
const fetchCurrentUser = async () => {
  try {
    const resp = await axios.get<{
      username: string; role: string; department: string
      department_name: string; department_path: string
      role_display_name: string; permission_hint: string
    }>(`${API_BASE}/auth/me`)
    const u = resp.data
    currentUser.value = {
      username: u.username,
      role: u.role || 'student',
      role_display: u.role_display_name || u.role || '研究生',
      department: u.department || '',
      department_name: u.department_name || '',
      department_path: u.department_path || '',
      permission_hint: u.permission_hint || '',
    }
    localStorage.setItem('eka_current_user', JSON.stringify(currentUser.value))
  } catch {
    // 获取失败，使用默认匿名用户
    currentUser.value = {
      username: 'anonymous',
      role: 'student',
      role_display: '研究生',
      department: '',
      department_name: '',
      department_path: '',
      permission_hint: '您可查看实验室公共资料与新人导览内容。',
    }
  }
}

// 监听 session 变化，重新加载历史
watch(sessionId, async (newSid) => {
  await loadHistory(newSid)
})
</script>
