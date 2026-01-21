const promptInput = document.getElementById('prompt-input');
const submitBtn = document.getElementById('submit-btn');
const executionFlow = document.getElementById('execution-flow');
const promptDisplay = document.getElementById('prompt-display');
const routingDisplay = document.getElementById('routing-display');
const executionDisplay = document.getElementById('execution-display');
const responseDisplay = document.getElementById('response-display');
const errorDisplay = document.getElementById('error-display');

const API_BASE_URL = '/api';

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatLatency(ms) {
    if (ms < 1000) {
        return `${Math.round(ms)}ms`;
    }
    return `${(ms / 1000).toFixed(2)}s`;
}

function extractModelName(fullModel) {
    const parts = fullModel.split('/');
    return parts[parts.length - 1];
}

function showError(message) {
    errorDisplay.textContent = message;
    errorDisplay.classList.remove('hidden');
    executionFlow.classList.add('hidden');
}

function hideError() {
    errorDisplay.classList.add('hidden');
}

function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    submitBtn.textContent = isLoading ? 'Processing...' : 'Submit';
}

function renderPromptStep(prompt) {
    promptDisplay.innerHTML = `<p class="prompt-text">${escapeHtml(prompt)}</p>`;
}

function renderRoutingStep(data) {
    const selection = data.selection;
    const classification = data.metadata?.classification || {};

    let html = `
        <div class="routing-info">
            <div class="info-row">
                <span class="label">Plan Type:</span>
                <span class="value plan-type-${data.plan_type}">${data.plan_type.replace('_', ' ')}</span>
            </div>
            <div class="info-row">
                <span class="label">Selected Model:</span>
                <span class="value model-name">${extractModelName(selection.model)}</span>
            </div>
            <div class="info-row">
                <span class="label">Provider:</span>
                <span class="value">${selection.provider}</span>
            </div>
    `;

    if (selection.reasoning) {
        html += `
            <div class="info-row">
                <span class="label">Reasoning:</span>
                <span class="value reasoning">${escapeHtml(selection.reasoning)}</span>
            </div>
        `;
    }

    if (selection.confidence !== null && selection.confidence !== undefined) {
        const confidencePercent = Math.round(selection.confidence * 100);
        html += `
            <div class="info-row">
                <span class="label">Confidence:</span>
                <span class="value">${confidencePercent}%</span>
            </div>
        `;
    }

    if (classification.reasoning) {
        html += `
            <div class="info-row">
                <span class="label">Classification:</span>
                <span class="value reasoning">${escapeHtml(classification.reasoning)}</span>
            </div>
        `;
    }

    html += '</div>';
    routingDisplay.innerHTML = html;
}

function renderExecutionStep(data) {
    const metadata = data.metadata || {};

    if (data.plan_type === 'single_shot') {
        const latency = metadata.latency_ms;
        let html = `
            <div class="execution-single">
                <div class="info-row">
                    <span class="label">Execution Type:</span>
                    <span class="value">Single Shot</span>
                </div>
        `;

        if (latency) {
            html += `
                <div class="info-row">
                    <span class="label">Latency:</span>
                    <span class="value latency">${formatLatency(latency)}</span>
                </div>
            `;
        }

        if (data.usage) {
            html += `
                <div class="info-row">
                    <span class="label">Tokens Used:</span>
                    <span class="value">${data.usage.total_tokens} (${data.usage.prompt_tokens} prompt + ${data.usage.completion_tokens} completion)</span>
                </div>
            `;
        }

        html += '</div>';
        executionDisplay.innerHTML = html;
    } else {
        // Multi-step execution
        let html = `
            <div class="execution-multi">
                <div class="info-row">
                    <span class="label">Execution Type:</span>
                    <span class="value">Multi-Step (${data.steps.length} steps)</span>
                </div>
        `;

        if (metadata.total_latency_ms) {
            html += `
                <div class="info-row">
                    <span class="label">Total Latency:</span>
                    <span class="value latency">${formatLatency(metadata.total_latency_ms)}</span>
                </div>
            `;
        }

        html += '<div class="steps-list">';

        for (const step of data.steps) {
            const stepLatency = metadata.step_latencies_ms?.[step.step_number];
            const statusClass = step.status === 'completed' ? 'success' : step.status === 'failed' ? 'failed' : 'pending';

            html += `
                <div class="step-item ${statusClass}">
                    <div class="step-item-header">
                        <span class="step-number">Step ${step.step_number}</span>
                        <span class="step-status status-${statusClass}">${step.status}</span>
                    </div>
                    <div class="step-item-task">${escapeHtml(step.task)}</div>
                    <div class="step-item-meta">
                        <span class="step-model">${extractModelName(step.model)}</span>
            `;

            if (stepLatency) {
                html += `<span class="step-latency">${formatLatency(stepLatency)}</span>`;
            }

            html += '</div>';

            if (step.output) {
                html += `<div class="step-item-output"><pre>${escapeHtml(step.output)}</pre></div>`;
            }

            html += '</div>';
        }

        html += '</div></div>';
        executionDisplay.innerHTML = html;
    }
}

function renderResponseStep(data) {
    let html = '';

    if (data.plan_type === 'single_shot' && data.content) {
        html = `<div class="final-response"><pre>${escapeHtml(data.content)}</pre></div>`;
    } else if (data.steps && data.steps.length > 0) {
        // For multi-step, show the final step's output as the response
        const completedSteps = data.steps.filter(s => s.status === 'completed');
        if (completedSteps.length > 0) {
            const lastStep = completedSteps[completedSteps.length - 1];
            html = `
                <div class="final-response">
                    <p class="response-note">Final output from Step ${lastStep.step_number}:</p>
                    <pre>${escapeHtml(lastStep.output || 'No output')}</pre>
                </div>
            `;
        } else {
            html = '<p class="no-response">No completed steps</p>';
        }
    } else {
        html = '<p class="no-response">No response generated</p>';
    }

    responseDisplay.innerHTML = html;
}

function renderExecutionFlow(prompt, data) {
    hideError();
    executionFlow.classList.remove('hidden');

    renderPromptStep(prompt);
    renderRoutingStep(data);
    renderExecutionStep(data);
    renderResponseStep(data);
}

async function submitPrompt() {
    const prompt = promptInput.value.trim();

    if (!prompt) {
        showError('Please enter a prompt.');
        return;
    }

    setLoading(true);
    hideError();
    executionFlow.classList.add('hidden');

    try {
        const response = await fetch(`${API_BASE_URL}/route`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                messages: [
                    { role: 'user', content: prompt }
                ]
            }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error: ${response.status}`);
        }

        const data = await response.json();
        renderExecutionFlow(prompt, data);
    } catch (error) {
        showError(`Error: ${error.message}`);
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
