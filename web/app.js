const state = {
  sessionId: localStorage.getItem('native-agent-session') || crypto.randomUUID(),
  lastRunId: null,
  busy: false,
  model: null,
  provider: null,
  loadingModels: false,
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
  if (!text.trim() || state.busy || state.loadingModels || !state.model) return;
  state.busy = true; syncControls();
  addMessage('user', text.trim());
  prompt.value = ''; resizePrompt();
  const thinking = addMessage('agent', '처리 중<span class="dots"><span>.</span><span>.</span><span>.</span></span>', 'thinking');
  thinking.querySelector('.bubble').innerHTML = '처리 중<span class="dots"><span>.</span><span>.</span><span>.</span></span>';
  try {
    const reply = await request('/api/chat', {method:'POST', body:JSON.stringify({session_id:state.sessionId, message:text, model:state.model})});
    thinking.remove(); handleReply(reply);
    localStorage.setItem('native-agent-session', state.sessionId);
    loadSessions();
  } catch (error) {
    thinking.remove(); addMessage('agent', `오류: ${error.message}`);
  } finally {
    state.busy = false; syncControls(); prompt.focus();
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
    state.provider = config.provider;
    $('#provider-name').textContent = ({demo: 'Demo Runtime', openai: 'OpenAI Runtime', ollama: 'Ollama · 로컬 AI'})[config.provider] || config.provider;
    $('#model-name').textContent = config.model;
    $('#runtime-label').textContent = `도구 · 승인 · 메모리 · 추적 · ${config.provider === 'ollama' ? 'Ollama ' + config.model : config.provider}`;
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
    $('#trace-summary').innerHTML = `<strong>${escapeHtml(run.status)}</strong> · ${escapeHtml(run.provider)} · ${escapeHtml(run.model || "")}<br>${escapeHtml(run.id)}`;
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
loadConfig().then(loadModels); loadSessions(); prompt.focus();

function syncControls() {
  send.disabled = state.busy || state.loadingModels || !state.model;
  $('#model-select').disabled = state.busy || state.loadingModels || !state.model;
  $('#refresh-models').disabled = state.busy || state.loadingModels;
}
function displayModel() {
  $('#model-name').textContent = state.model || '선택 가능한 모델 없음';
  $('#runtime-label').textContent = `도구 · 승인 · 메모리 · 추적 · ${state.provider || ''} ${state.model || ''}`;
}
async function loadModels() {
  state.loadingModels = true; syncControls();
  $('#model-status').textContent = '설치 모델 확인 중…';
  try {
    const data = await request('/api/models');
    const select = $('#model-select');
    select.replaceChildren();
    for (const model of data.models) {
      const option = document.createElement('option');
      option.value = model.name;
      option.textContent = `${model.name}${model.size_gb ? ' · ' + model.size_gb + ' GB' : ''}${model.available ? '' : ' · ' + model.reason}`;
      option.disabled = !model.available;
      select.appendChild(option);
    }
    const available = data.models.filter(model => model.available);
    const saved = state.model || localStorage.getItem('copilot-model');
    state.model = available.find(model => model.name === saved)?.name || available.find(model => model.name === data.default_model)?.name || available[0]?.name || null;
    if (state.model) select.value = state.model;
    else select.replaceChildren(new Option('사용 가능한 모델이 없습니다', ''));
    $('#model-status').textContent = state.model ? `${available.length}개 사용 가능 · 다음 요청부터 적용` : 'Ollama에 도구 호출 지원 모델을 설치하세요.';
    displayModel();
  } catch (error) {
    state.model = null;
    $('#model-select').replaceChildren(new Option('모델 목록을 불러오지 못했습니다', ''));
    $('#model-status').textContent = error.message;
    displayModel();
  } finally { state.loadingModels = false; syncControls(); }
}
$('#model-select').onchange = () => {
  state.model = $('#model-select').value;
  localStorage.setItem('copilot-model', state.model);
  displayModel();
  $('#model-status').textContent = '다음 요청부터 적용 · 승인 대기 작업은 원래 모델로 처리';
};
$('#refresh-models').onclick = loadModels;
syncControls();
