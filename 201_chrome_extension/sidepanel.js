// ===== Configuration =====
const CONFIG = {
    serverUrl: 'http://localhost:8000/api/extract',
    timeout: 30000 // 30 seconds
};

// ===== DOM Elements =====
const elements = {
    pageUrl: document.getElementById('pageUrl'),
    extractBtn: document.getElementById('extractBtn'),
    btnLoader: document.getElementById('btnLoader'),
    status: document.getElementById('status'),
    responseContent: document.getElementById('responseContent'),
    clearBtn: document.getElementById('clearBtn'),
    debugContent: document.getElementById('debugContent')
};

// ===== State =====
let isLoading = false;

// ===== Initialize =====
document.addEventListener('DOMContentLoaded', init);

async function init() {
    // Get current tab info
    await updatePageInfo();

    // Set up event listeners
    elements.extractBtn.addEventListener('click', handleExtract);
    elements.clearBtn.addEventListener('click', clearResponse);

    // Listen for tab changes
    chrome.tabs.onActivated.addListener(updatePageInfo);
    chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
        if (changeInfo.url) updatePageInfo();
    });
}

// ===== Update Page Info =====
async function updatePageInfo() {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tab) {
            elements.pageUrl.textContent = tab.url || 'Unknown';
            elements.pageUrl.title = tab.url || '';
        }
    } catch (error) {
        elements.pageUrl.textContent = 'Unable to get page info';
        logDebug('Error getting tab info:', error);
    }
}

// ===== Handle Extract =====
async function handleExtract() {
    if (isLoading) return;

    setLoading(true);
    setStatus('Extracting page content...', 'loading');
    clearResponse();

    try {
        // Get current tab
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        if (!tab || !tab.id) {
            throw new Error('No active tab found');
        }

        // Check if we can inject into this page
        if (tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://')) {
            throw new Error('Cannot extract from Chrome system pages');
        }

        logDebug('Extracting from tab:', { id: tab.id, url: tab.url });

        // Inject content script and get HTML
        const results = await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: () => {
                return {
                    html: document.documentElement.outerHTML,
                    url: window.location.href,
                    title: document.title
                };
            }
        });

        if (!results || !results[0] || !results[0].result) {
            throw new Error('Failed to extract page content');
        }

        const pageData = results[0].result;
        logDebug('Extracted HTML length:', pageData.html.length);

        setStatus('Sending to LLM server...', 'loading');

        // Send to server
        const response = await sendToServer(pageData);

        setStatus('Success!', 'success');
        displayResponse(response);

    } catch (error) {
        console.error('Extract error:', error);
        setStatus(`Error: ${error.message}`, 'error');
        logDebug('Error details:', error);
    } finally {
        setLoading(false);
    }
}

// ===== Send to Server =====
async function sendToServer(pageData) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), CONFIG.timeout);

    try {
        logDebug('Sending request to:', CONFIG.serverUrl);
        logDebug('Payload size:', JSON.stringify(pageData).length, 'bytes');

        const response = await fetch(CONFIG.serverUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                html: pageData.html,
                url: pageData.url,
                title: pageData.title
            }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error (${response.status}): ${errorText}`);
        }

        const data = await response.json();
        logDebug('Server response:', data);

        return data;

    } catch (error) {
        clearTimeout(timeoutId);

        if (error.name === 'AbortError') {
            throw new Error('Request timed out');
        }

        if (error.message.includes('Failed to fetch')) {
            throw new Error('Cannot connect to server. Is it running on localhost:8000?');
        }

        throw error;
    }
}

// ===== Display Response =====
function displayResponse(data) {
    if (typeof data === 'string') {
        elements.responseContent.innerHTML = `<pre>${escapeHtml(data)}</pre>`;
    } else {
        elements.responseContent.innerHTML = `<pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre>`;
    }
}

// ===== Clear Response =====
function clearResponse() {
    elements.responseContent.innerHTML = `
    <div class="placeholder">
      Click "Extract Page" to send this page to your local LLM server for analysis.
    </div>
  `;
}

// ===== Set Loading State =====
function setLoading(loading) {
    isLoading = loading;
    elements.extractBtn.disabled = loading;
    elements.extractBtn.classList.toggle('loading', loading);
}

// ===== Set Status =====
function setStatus(message, type = '') {
    elements.status.textContent = message;
    elements.status.className = `status ${type}`;
}

// ===== Debug Logging =====
function logDebug(...args) {
    const timestamp = new Date().toLocaleTimeString();
    const message = args.map(arg =>
        typeof arg === 'object' ? JSON.stringify(arg, null, 2) : arg
    ).join(' ');

    const existing = elements.debugContent.querySelector('pre').textContent;
    const newContent = existing === 'No debug info yet.'
        ? `[${timestamp}] ${message}`
        : `${existing}\n[${timestamp}] ${message}`;

    elements.debugContent.innerHTML = `<pre>${escapeHtml(newContent)}</pre>`;
}

// ===== Utility: Escape HTML =====
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
