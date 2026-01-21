const promptInput = document.getElementById('prompt-input');
const submitBtn = document.getElementById('submit-btn');
const responseDisplay = document.getElementById('response-display');

const API_BASE_URL = '/api';

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatResponse(data) {
    let html = '';

    // Show routing metadata
    html += '<div class="metadata">';
    html += `<span class="badge model">${escapeHtml(data.model)}</span>`;
    html += `<span class="badge plan-type">${escapeHtml(data.plan_type)}</span>`;
    if (data.usage) {
        html += `<span class="badge tokens">${data.usage.total_tokens || 0} tokens</span>`;
    }
    html += '</div>';

    // Show steps for multi-step responses
    if (data.steps && data.steps.length > 0) {
        html += '<div class="steps">';
        for (const step of data.steps) {
            html += `<div class="step ${step.status}">`;
            html += `<div class="step-header">Step ${step.step}: ${escapeHtml(step.task)}</div>`;
            html += `<div class="step-output">${escapeHtml(step.output || 'No output')}</div>`;
            html += '</div>';
        }
        html += '</div>';
    } else {
        // Single response
        html += `<div class="response-content">${escapeHtml(data.response)}</div>`;
    }

    return html;
}

function setResponse(content, className = 'content') {
    responseDisplay.innerHTML = `<p class="${className}">${content}</p>`;
}

function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    if (isLoading) {
        setResponse('Processing... (classifying query and selecting model)', 'loading');
    }
}

async function submitPrompt() {
    const prompt = promptInput.value.trim();

    if (!prompt) {
        setResponse('Please enter a prompt.', 'error');
        return;
    }

    setLoading(true);

    try {
        const response = await fetch(`${API_BASE_URL}/prompt`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error: ${response.status}`);
        }

        const data = await response.json();
        responseDisplay.innerHTML = formatResponse(data);
    } catch (error) {
        setResponse(`Error: ${error.message}`, 'error');
    } finally {
        setLoading(false);
    }
}

submitBtn.addEventListener('click', submitPrompt);

promptInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && event.ctrlKey) {
        submitPrompt();
    }
});
