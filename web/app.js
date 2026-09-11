const state = {
  sessionId: localStorage.getItem('native-agent-session') || crypto.randomUUID(),
  lastRunId: null,
  busy: false,
};

const $ = (selector) => document.querySelector(selector);
const messages = $('#messages');
const welcome = $('#welcome');
const prompt = $('#prompt');
const send = $('#send');

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

function simpleMarkdown(value) {
  return escapeHtml(value).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/`(.+?)`/g, '<code>$1</code>');
}

function addMessage(role, text, className = '') {
  welcome.hidden = true;
  const element = document.createElement('div');
  element.className = `message ${role} ${className}`;
  element.innerHTML = `<div class="avatar">${role === 'user' ? 'ME' : 'N'}</div><div class="bubble">${simpleMarkdown(text)}</div>`;
  messages.appendChild(element);
  element.scrollIntoView({behavior:'smooth', block:'end'});
  return element;
}

function addApproval(approval) {
  const card = document.createElement('div');
  card.className = 'approval-card';
  card.innerHTML = `
    <span class="approval-label">APPROVAL REQUIRED · ${escapeHtml(approval.risk)}</span>
    <h3>${escapeHtml(approval.tool)} 실행을 승인할까요?</h3>
    <p>${escapeHtml(approval.reason)}</p>
    <div class="approval-code">${escapeHtml(JSON.stringify(approval.arguments, null, 2))}</div>
    <div class="approval-actions"><button class="deny">거절</button><button class="allow">승인하고 실행</button></div>`;
  messages.appendChild(card);
  card.scrollIntoView({behavior:'smooth', block:'end'});
  card.querySelector('.deny').onclick = () => resolveApproval(approval.id, false, card);
  card.querySelector('.allow').onclick = () => resolveApproval(approval.id, true, card);
}

async function request(path, options = {}) {
  const response = await fetch(path, {headers:{'Content-Type':'application/json'}, ...options});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || '요청에 실패했습니다.');
  return payload;
}

async function submitMessage(text) {
  if (!text.trim() || state.busy) return;
  state.busy = true; send.disabled = true;
  addMessage('user', text.trim());
  prompt.value = ''; resizePrompt();
  const thinking = addMessage('agent', '처리 중<span class="dots"><span>.</span><span>.</span><span>.</span></span>', 'thinking');
  thinking.querySelector('.bubble').innerHTML = '처리 중<span class="dots"><span>.</span><span>.</span><span>.</span></span>';
  try {
    const reply = await request('/api/chat', {method:'POST', body:JSON.stringify({session_id:state.sessionId, message:text})});
    thinking.remove(); handleReply(reply);
    localStorage.setItem('native-agent-session', state.sessionId);
    loadSessions();
  } catch (error) {
    thinking.remove(); addMessage('agent', `오류: ${error.message}`);
  } finally {
    state.busy = false; send.disabled = false; prompt.focus();
  }
}

function handleReply(reply) {
  state.lastRunId = reply.run_id;
  $('#trace-button').disabled = false;
  if (reply.status === 'approval_required') addApproval(reply.approval);
  else addMessage('agent', reply.message || '응답이 없습니다.');
}

async function resolveApproval(id, approved, card) {
  card.querySelectorAll('button').forEach(button => button.disabled = true);
  try {
    const reply = await request(`/api/approvals/${id}`, {method:'POST', body:JSON.stringify({approved})});
    card.remove();
    addMessage('user', approved ? '승인했습니다.' : '거절했습니다.');
    handleReply(reply);
  } catch (error) {
    addMessage('agent', `승인 처리 오류: ${error.message}`);
  }
}

async function loadConfig() {
  try {
    const config = await request('/api/config');
    $('#provider-name').textContent = config.provider === 'demo' ? 'Demo Runtime' : 'OpenAI Runtime';
    $('#model-name').textContent = config.model;
  } catch (_) {}
}

async function loadSessions() {
  try {
    const payload = await request('/api/sessions');
    $('#sessions').innerHTML = payload.sessions.slice(0, 8).map(session =>
      `<button class="session-item ${session.id === state.sessionId ? 'active':''}" title="${escapeHtml(session.title)}">${escapeHtml(session.title)}</button>`
    ).join('');
  } catch (_) {}
}

async function showTrace() {
  if (!state.lastRunId) return;
  try {
    const run = await request(`/api/runs/${state.lastRunId}`);
    $('#trace-summary').innerHTML = `<strong>${escapeHtml(run.status)}</strong> · ${escapeHtml(run.provider)}<br>${escapeHtml(run.id)}`;
    $('#trace-events').innerHTML = run.events.map(event => `<div class="trace-event"><b>${escapeHtml(event.type)}</b><pre>${escapeHtml(JSON.stringify(event.data, null, 2))}</pre></div>`).join('');
    $('#trace-panel').classList.add('open');
    $('#trace-panel').setAttribute('aria-hidden','false');
  } catch (error) { addMessage('agent', `추적 조회 오류: ${error.message}`); }
}

function resizePrompt() { prompt.style.height = 'auto'; prompt.style.height = `${Math.min(prompt.scrollHeight,140)}px`; }
$('#composer').addEventListener('submit', event => { event.preventDefault(); submitMessage(prompt.value); });
prompt.addEventListener('input', resizePrompt);
prompt.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); submitMessage(prompt.value); } });
document.querySelectorAll('[data-prompt]').forEach(button => button.onclick = () => submitMessage(button.dataset.prompt));
$('#new-chat').onclick = () => { state.sessionId = crypto.randomUUID(); state.lastRunId = null; messages.innerHTML=''; welcome.hidden=false; $('#trace-button').disabled=true; localStorage.setItem('native-agent-session',state.sessionId); };
$('#trace-button').onclick = showTrace;
$('#trace-close').onclick = () => $('#trace-panel').classList.remove('open');
loadConfig(); loadSessions(); prompt.focus();

