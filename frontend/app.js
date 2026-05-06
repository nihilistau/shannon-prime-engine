// Shannon-Prime Frontend — App Logic
// Connects to the sp-engine OpenAI-compatible HTTP server.

(function () {
    'use strict';

    // ── Configuration ────────────────────────────────────────────────
    const DEFAULT_ENDPOINT = window.location.origin;
    const HEALTH_POLL_MS   = 8000;

    let apiEndpoint = localStorage.getItem('sp_endpoint') || DEFAULT_ENDPOINT;
    let maxTokens   = parseInt(localStorage.getItem('sp_max_tokens') || '128', 10);
    let isGenerating = false;

    // ── DOM refs ─────────────────────────────────────────────────────
    const chatArea      = document.getElementById('chat-area');
    const welcome       = document.getElementById('welcome');
    const userInput     = document.getElementById('user-input');
    const sendBtn       = document.getElementById('send-btn');
    const statusInd     = document.getElementById('status-indicator');
    const statusText    = statusInd.querySelector('.status-text');
    const modelNameEl   = document.getElementById('model-name');
    const latencyEl     = document.getElementById('latency-value');
    const settingsOvl   = document.getElementById('settings-overlay');
    const apiEndpointEl = document.getElementById('api-endpoint');
    const maxTokensEl   = document.getElementById('max-tokens');
    const settingsClose = document.getElementById('settings-close');

    // ── Messages state ───────────────────────────────────────────────
    const messages = [];

    // ── Health check loop ────────────────────────────────────────────
    async function checkHealth() {
        try {
            const resp = await fetch(`${apiEndpoint}/v1/models`, {
                signal: AbortSignal.timeout(5000)
            });
            if (!resp.ok) throw new Error(resp.statusText);
            const data = await resp.json();
            statusInd.className = 'header-status connected';
            statusText.textContent = 'Connected';
            if (data.data && data.data.length > 0) {
                modelNameEl.textContent = data.data[0].id;
            }
        } catch (e) {
            statusInd.className = 'header-status error';
            statusText.textContent = 'Offline';
            modelNameEl.textContent = '—';
        }
    }

    checkHealth();
    setInterval(checkHealth, HEALTH_POLL_MS);

    // ── Chat rendering ───────────────────────────────────────────────
    function appendMessage(role, content, meta) {
        if (welcome) welcome.style.display = 'none';

        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;

        const avatar = document.createElement('div');
        avatar.className = 'msg-avatar';
        avatar.textContent = role === 'user' ? 'U' : '⚡';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'msg-content';
        contentDiv.textContent = content;

        if (meta) {
            const metaDiv = document.createElement('div');
            metaDiv.className = 'msg-meta';
            metaDiv.textContent = meta;
            contentDiv.appendChild(metaDiv);
        }

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(contentDiv);
        chatArea.appendChild(msgDiv);
        chatArea.scrollTop = chatArea.scrollHeight;

        return contentDiv;
    }

    function appendThinking() {
        if (welcome) welcome.style.display = 'none';

        const msgDiv = document.createElement('div');
        msgDiv.className = 'message assistant';
        msgDiv.id = 'thinking-msg';

        const avatar = document.createElement('div');
        avatar.className = 'msg-avatar';
        avatar.textContent = '⚡';

        const thinkDiv = document.createElement('div');
        thinkDiv.className = 'thinking';
        thinkDiv.innerHTML = '<span></span><span></span><span></span>';

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(thinkDiv);
        chatArea.appendChild(msgDiv);
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    function removeThinking() {
        const el = document.getElementById('thinking-msg');
        if (el) el.remove();
    }

    // ── Send message ─────────────────────────────────────────────────
    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text || isGenerating) return;

        isGenerating = true;
        sendBtn.disabled = true;
        userInput.value = '';
        autoResizeInput();

        messages.push({ role: 'user', content: text });
        appendMessage('user', text);
        appendThinking();

        const t0 = performance.now();

        try {
            const resp = await fetch(`${apiEndpoint}/v1/chat/completions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: modelNameEl.textContent || 'default',
                    messages: messages,
                    max_tokens: maxTokens
                })
            });

            const elapsed = performance.now() - t0;
            removeThinking();

            if (!resp.ok) {
                const err = await resp.text();
                appendMessage('assistant', `Error: ${resp.status} — ${err}`, `${elapsed.toFixed(0)}ms`);
                return;
            }

            const data = await resp.json();
            const reply = data.choices?.[0]?.message?.content || '(empty response)';
            const meta = `${elapsed.toFixed(0)}ms`;

            messages.push({ role: 'assistant', content: reply });
            appendMessage('assistant', reply, meta);
            latencyEl.textContent = `${elapsed.toFixed(0)}ms`;

        } catch (e) {
            const elapsed = performance.now() - t0;
            removeThinking();
            appendMessage('assistant', `Connection error: ${e.message}`, `${elapsed.toFixed(0)}ms`);
        } finally {
            isGenerating = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    }

    // ── Input handling ───────────────────────────────────────────────
    function autoResizeInput() {
        userInput.style.height = 'auto';
        userInput.style.height = Math.min(userInput.scrollHeight, 150) + 'px';
    }

    userInput.addEventListener('input', autoResizeInput);

    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);

    // ── Quick prompts ────────────────────────────────────────────────
    document.querySelectorAll('.quick-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            userInput.value = btn.dataset.prompt;
            autoResizeInput();
            sendMessage();
        });
    });

    // ── Settings ─────────────────────────────────────────────────────
    apiEndpointEl.value = apiEndpoint;
    maxTokensEl.value   = maxTokens;

    settingsClose.addEventListener('click', () => {
        apiEndpoint = apiEndpointEl.value.replace(/\/+$/, '');
        maxTokens   = parseInt(maxTokensEl.value, 10) || 128;
        localStorage.setItem('sp_endpoint', apiEndpoint);
        localStorage.setItem('sp_max_tokens', String(maxTokens));
        settingsOvl.classList.remove('open');
        checkHealth();
    });

    // Clicking outside closes settings
    settingsOvl.addEventListener('click', (e) => {
        if (e.target === settingsOvl) settingsOvl.classList.remove('open');
    });

})();
