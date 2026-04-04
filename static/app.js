/* ═══════════════════════════════════════════
   DocMind – App Logic (app.js)
   ═══════════════════════════════════════════ */

const API = '';  // same origin

// ── DOM refs ──
const dropZone      = document.getElementById('dropZone');
const fileInput     = document.getElementById('fileInput');
const uploadProgress= document.getElementById('uploadProgress');
const progressFill  = document.getElementById('progressFill');
const progressLabel = document.getElementById('progressLabel');
const docList       = document.getElementById('docList');
const docCount      = document.getElementById('docCount');
const clearBtn      = document.getElementById('clearBtn');
const messagesWrap  = document.getElementById('messagesWrap');
const welcomeScreen = document.getElementById('welcomeScreen');
const msgContainer  = document.getElementById('messagesContainer');
const chatInput     = document.getElementById('chatInput');
const sendBtn       = document.getElementById('sendBtn');
const newChatBtn    = document.getElementById('newChatBtn');
const statusDot     = document.getElementById('statusDot');
const statusText    = document.getElementById('statusText');
const sidebarToggle = document.getElementById('sidebarToggle');
const sidebar       = document.getElementById('sidebar');

let isTyping = false;

// ══════════════════════════════════════════
// Toast Notifications
// ══════════════════════════════════════════
const toastContainer = document.createElement('div');
toastContainer.className = 'toast-container';
document.body.appendChild(toastContainer);

function showToast(msg, type = 'info', duration = 4000) {
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  toastContainer.appendChild(t);
  setTimeout(() => {
    t.style.opacity = '0';
    t.style.transform = 'translateX(20px)';
    t.style.transition = 'all 0.3s ease';
    setTimeout(() => t.remove(), 300);
  }, duration);
}

// ══════════════════════════════════════════
// Health Check / Status
// ══════════════════════════════════════════
async function checkHealth() {
  try {
    const res = await fetch(`${API}/health`);
    if (res.ok) {
      const data = await res.json();
      statusDot.className  = 'status-dot online';
      statusText.textContent = `Online · ${data.documents} doc${data.documents !== 1 ? 's' : ''}`;
    } else {
      throw new Error('offline');
    }
  } catch {
    statusDot.className   = 'status-dot offline';
    statusText.textContent = 'Server offline';
  }
}

// ══════════════════════════════════════════
// Document List
// ══════════════════════════════════════════
async function loadDocuments() {
  try {
    const res  = await fetch(`${API}/documents`);
    const data = await res.json();
    renderDocList(data.documents || []);
  } catch {
    // silent
  }
}

function renderDocList(docs) {
  docCount.textContent = docs.length;
  docList.innerHTML = '';

  if (docs.length === 0) {
    docList.innerHTML = '<li class="doc-empty">No documents yet</li>';
    return;
  }

  docs.forEach(doc => {
    const li   = document.createElement('li');
    li.className = 'doc-item';
    const ext  = doc.name.split('.').pop().toLowerCase();
    const icon = ext === 'pdf' ? '📄' : ext === 'docx' ? '📝' : '📃';
    const size = formatBytes(doc.size);
    li.innerHTML = `
      <span class="doc-icon">${icon}</span>
      <div class="doc-info">
        <div class="doc-name" title="${doc.name}">${doc.name}</div>
        <div class="doc-meta">${doc.chunks} chunks · ${size}</div>
      </div>`;
    docList.appendChild(li);
  });
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes/1024).toFixed(1)} KB`;
  return `${(bytes/1024/1024).toFixed(1)} MB`;
}

// ══════════════════════════════════════════
// File Upload
// ══════════════════════════════════════════
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  uploadFiles(Array.from(e.dataTransfer.files));
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) {
    uploadFiles(Array.from(fileInput.files));
    fileInput.value = '';
  }
});

async function uploadFiles(files) {
  const allowed = ['.pdf', '.txt', '.docx'];
  const valid   = files.filter(f => allowed.some(ext => f.name.toLowerCase().endsWith(ext)));

  if (valid.length === 0) {
    showToast('Only PDF, DOCX, and TXT files are supported.', 'error');
    return;
  }

  uploadProgress.classList.remove('hidden');

  for (let i = 0; i < valid.length; i++) {
    const file = valid[i];
    const pct  = Math.round(((i) / valid.length) * 100);
    progressFill.style.width  = `${pct}%`;
    progressLabel.textContent = `Uploading ${file.name}…`;

    try {
      const formData = new FormData();
      formData.append('file', file);
      const res  = await fetch(`${API}/upload`, { method: 'POST', body: formData });
      const data = await res.json();

      if (res.ok) {
        showToast(`✓ "${file.name}" indexed (${data.chunks} chunks)`, 'success');
      } else {
        showToast(`✗ ${data.detail || 'Upload failed'}`, 'error');
      }
    } catch (e) {
      showToast(`✗ Network error uploading "${file.name}"`, 'error');
    }
  }

  progressFill.style.width  = '100%';
  progressLabel.textContent  = 'Done!';
  setTimeout(() => {
    uploadProgress.classList.add('hidden');
    progressFill.style.width = '0%';
  }, 1500);

  await loadDocuments();
  await checkHealth();
}

// ══════════════════════════════════════════
// Clear Documents
// ══════════════════════════════════════════
clearBtn.addEventListener('click', async () => {
  if (!confirm('Remove all documents and clear the vector store?')) return;
  try {
    await fetch(`${API}/documents`, { method: 'DELETE' });
    renderDocList([]);
    showToast('All documents cleared.', 'info');
    await checkHealth();
  } catch {
    showToast('Failed to clear documents.', 'error');
  }
});

// ══════════════════════════════════════════
// Chat
// ══════════════════════════════════════════
function showWelcome(show) {
  welcomeScreen.style.display  = show ? 'flex'  : 'none';
  msgContainer.style.display   = show ? 'none' : 'flex';
}

function appendMessage(role, text, toolsUsed = []) {
  showWelcome(false);

  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.textContent = role === 'user' ? 'U' : '🤖';

  const body = document.createElement('div');
  body.className = 'msg-body';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.innerHTML = formatMarkdown(text);
  body.appendChild(bubble);

  // Tool usage accordion (only for AI messages)
  if (role === 'ai' && toolsUsed.length > 0) {
    const accordion = buildToolAccordion(toolsUsed);
    body.appendChild(accordion);
  }

  wrap.appendChild(avatar);
  wrap.appendChild(body);
  msgContainer.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function buildToolAccordion(tools) {
  const wrap = document.createElement('div');
  wrap.className = 'tools-used';

  const header = document.createElement('div');
  header.className = 'tools-header';
  header.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
    ${tools.length} tool${tools.length > 1 ? 's' : ''} used · click to expand
  `;

  const body = document.createElement('div');
  body.className = 'tools-body';

  tools.forEach(step => {
    const s = document.createElement('div');
    s.className = 'tool-step';
    s.innerHTML = `
      <div class="tool-name-tag ${step.tool}">${toolIcon(step.tool)} ${step.tool}</div>
      <div class="tool-input-label">Input:</div>
      <div class="tool-input-val">${escapeHtml(step.input)}</div>
    `;
    body.appendChild(s);
  });

  header.addEventListener('click', () => {
    const open = body.classList.toggle('open');
    header.classList.toggle('open', open);
  });

  wrap.appendChild(header);
  wrap.appendChild(body);
  return wrap;
}

function toolIcon(toolName) {
  const icons = { document_search: '📄', calculator: '🧮', web_search: '🌐' };
  return icons[toolName] || '🔧';
}

function showTyping() {
  const wrap = document.createElement('div');
  wrap.className = 'message ai';
  wrap.id = 'typingMsg';

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.textContent = '🤖';

  const body = document.createElement('div');
  body.className = 'msg-body';

  const bubble = document.createElement('div');
  bubble.className = 'typing-indicator';
  bubble.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
  body.appendChild(bubble);

  wrap.appendChild(avatar);
  wrap.appendChild(body);
  msgContainer.appendChild(wrap);
  scrollToBottom();
}

function removeTyping() {
  const t = document.getElementById('typingMsg');
  if (t) t.remove();
}

async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text || isTyping) return;

  chatInput.value = '';
  chatInput.style.height = 'auto';
  sendBtn.disabled = true;
  isTyping = true;

  appendMessage('user', text);
  showTyping();

  try {
    const url = `${API}/chat`;
    console.log(`Sending POST request to: ${url}`);
    
    const res  = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    const data = await res.json();
    removeTyping();

    if (res.ok) {
      const answer = data.answer || '(No response from AI)';
      appendMessage('ai', answer, data.tools_used || []);
    } else {
      appendMessage('ai', `⚠️ Error: ${data.detail || 'Unknown error'}`, []);
    }
  } catch (e) {
    removeTyping();
    console.error('Fetch error details:', e);
    
    let helpMsg = `⚠️ Could not reach the server. [Error: ${e.message}]`;
    if (window.location.protocol === 'file:') {
      helpMsg = '⚠️ Error: You are opening the HTML file directly. Please visit http://127.0.0.1:8000 in your browser.';
    } else if (e.message.includes('Failed to fetch')) {
      helpMsg = `⚠️ Connection failed. The Python server might have crashed or been blocked. [Detail: ${e.message}]`;
    }
    
    appendMessage('ai', helpMsg, []);
  }

  isTyping = false;
  sendBtn.disabled = chatInput.value.trim() === '';
}

sendBtn.addEventListener('click', sendMessage);

chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

chatInput.addEventListener('input', () => {
  // Auto-grow textarea
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 180) + 'px';
  sendBtn.disabled = chatInput.value.trim() === '';
});

// Suggestion chips
document.querySelectorAll('.suggestion').forEach(btn => {
  btn.addEventListener('click', () => {
    chatInput.value = btn.dataset.text;
    chatInput.dispatchEvent(new Event('input'));
    sendMessage();
  });
});

// New chat
newChatBtn.addEventListener('click', () => {
  msgContainer.innerHTML = '';
  showWelcome(true);
});

// Sidebar toggle
sidebarToggle.addEventListener('click', () => {
  sidebar.classList.toggle('collapsed');
  sidebar.classList.toggle('open');
});

function scrollToBottom() {
  messagesWrap.scrollTop = messagesWrap.scrollHeight;
}

// ══════════════════════════════════════════
// Markdown + formatting helpers
// ══════════════════════════════════════════
function formatMarkdown(text) {
  // Guard against null/undefined to prevent 'Cannot read properties of undefined' crash
  if (!text) return '';
  const safeText = String(text);
  if (typeof marked !== 'undefined') {
    return marked.parse(safeText);
  }
  // Fallback if marked didn't load
  return safeText
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*/g, '•')
    .replace(/\n/g, '<br>');
}

function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  if (typeof str !== 'string') str = JSON.stringify(str);
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ══════════════════════════════════════════
// Init
// ══════════════════════════════════════════
(async () => {
  showWelcome(true);
  await checkHealth();
  await loadDocuments();
  setInterval(checkHealth, 30000);
})();
