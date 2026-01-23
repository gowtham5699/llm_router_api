const promptInput = document.getElementById('prompt-input');
const submitBtn = document.getElementById('submit-btn');
const executionFlow = document.getElementById('execution-flow');
const promptDisplay = document.getElementById('prompt-display');
const routingDisplay = document.getElementById('routing-display');
const executionDisplay = document.getElementById('execution-display');
const responseDisplay = document.getElementById('response-display');
const errorDisplay = document.getElementById('error-display');

const API_BASE_URL = 'http://localhost:8000/api';

// State management
let currentState = {
    phase: 'input', // 'input', 'classified', 'model-selection', 'executing', 'complete'
    originalPrompt: '',
    classification: null,
    classifierInteraction: null,
    availableModels: [],
    selectedModelName: null,
};

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
    if (!fullModel) return 'unknown';
    const parts = fullModel.split('/');
    return parts[parts.length - 1];
}

function showError(message) {
    errorDisplay.textContent = message;
    errorDisplay.classList.remove('hidden');
}

function hideError() {
    errorDisplay.classList.add('hidden');
}

function setLoading(isLoading, message = 'Processing...') {
    submitBtn.disabled = isLoading;
    submitBtn.textContent = isLoading ? message : 'Submit';
}

function resetState() {
    currentState = {
        phase: 'input',
        originalPrompt: '',
        classification: null,
        classifierInteraction: null,
        availableModels: [],
        selectedModelName: null,
    };
    executionFlow.classList.add('hidden');
    hideError();
}

// ============== PHASE 1: Classification ==============

function renderClassifierInteraction() {
    const interaction = currentState.classifierInteraction;
    const classification = currentState.classification;

    let html = '<div class="routing-info">';

    // Classifier Interaction
    html += `
        <div class="routing-section">
            <h4>1. Classifier Interaction (gpt-4o-mini)</h4>
            <div class="classifier-details">
                <div class="detail-block">
                    <div class="detail-label">Prompt Sent to Classifier:</div>
                    <pre class="classifier-prompt">${escapeHtml(interaction.prompt_sent)}</pre>
                </div>
                <div class="detail-block">
                    <div class="detail-label">Raw Classifier Response:</div>
                    <pre class="classifier-response">${escapeHtml(JSON.stringify(interaction.raw_response, null, 2))}</pre>
                </div>
                ${interaction.latency_ms ? `<div class="classifier-latency">Classification took ${formatLatency(interaction.latency_ms)}</div>` : ''}
            </div>
            <div class="classification-summary">
                <div class="info-row">
                    <span class="label">Plan Type:</span>
                    <span class="value">${classification.plan_type || 'single_shot'}</span>
                </div>
                <div class="info-row">
                    <span class="label">Confidence:</span>
                    <span class="value">${Math.round((classification.confidence || 0.7) * 100)}%</span>
                </div>
                <div class="info-row">
                    <span class="label">Reasoning:</span>
                    <span class="value reasoning">${escapeHtml(classification.reasoning || '')}</span>
                </div>
            </div>
        </div>
    `;

    // Action buttons
    html += `
        <div class="routing-section action-section">
            <h4>2. Choose Execution Mode</h4>
            <div class="action-buttons">
                <button id="select-model-btn" class="action-btn primary">Select a Model</button>
                <button id="tournament-btn" class="action-btn secondary">Tournament Mode</button>
            </div>
        </div>
    `;

    html += '</div>';
    routingDisplay.innerHTML = html;

    // Add event listeners
    document.getElementById('select-model-btn').addEventListener('click', showModelSelection);
    document.getElementById('tournament-btn').addEventListener('click', runTournamentMode);
}

async function classifyPrompt() {
    const prompt = promptInput.value.trim();

    if (!prompt) {
        showError('Please enter a prompt.');
        return;
    }

    currentState.originalPrompt = prompt;
    setLoading(true, 'Classifying...');
    hideError();
    executionFlow.classList.remove('hidden');

    // Show prompt
    promptDisplay.innerHTML = `<p class="prompt-text">${escapeHtml(prompt)}</p>`;
    routingDisplay.innerHTML = '<p class="loading">Classifying prompt...</p>';
    executionDisplay.innerHTML = '';
    responseDisplay.innerHTML = '';

    try {
        const response = await fetch(`${API_BASE_URL}/classify`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error: ${response.status}`);
        }

        const data = await response.json();
        console.log('Classification result:', data);

        currentState.phase = 'classified';
        currentState.classification = data.classification;
        currentState.classifierInteraction = data.classifier_interaction;
        currentState.availableModels = data.available_models;

        renderClassifierInteraction();
    } catch (error) {
        console.error('Classification error:', error);
        showError(`Error: ${error.message}`);
    } finally {
        setLoading(false);
    }
}

// ============== PHASE 2A: Select a Model ==============

function showModelSelection() {
    currentState.phase = 'model-selection';
    const isMultiStep = currentState.classification.plan_type === 'multi_step';
    const steps = currentState.classification.steps || [];
    
    console.log('showModelSelection - isMultiStep:', isMultiStep, 'steps:', steps);

    let html = '<div class="routing-info">';

    // Show classifier summary (collapsed)
    html += `
        <div class="routing-section collapsed">
            <h4>1. Classification Complete</h4>
            <div class="classification-summary">
                <div class="info-row">
                    <span class="label">Plan Type:</span>
                    <span class="value">${currentState.classification.plan_type || 'single_shot'}</span>
                </div>
            </div>
        </div>
    `;

    if (isMultiStep) {
        // Multi-step: Show per-step model selection
        if (steps.length === 0) {
            // Classifier said multi-step but didn't provide steps - show warning
            html += `
                <div class="routing-section">
                    <h4>2. Multi-Step Execution</h4>
                    <p class="section-note warning">Classifier detected multi-step but no steps were defined. Using single model for all steps.</p>
                    <div class="models-selection-grid">
            `;
            for (const model of currentState.availableModels) {
                html += `
                    <div class="model-select-option" data-model="${escapeHtml(model.name)}">
                        <div class="model-select-name">${escapeHtml(model.name)}</div>
                        <div class="model-select-desc">${escapeHtml(model.description)}</div>
                        <div class="model-select-meta">
                            <span class="model-economy">${escapeHtml(model.economy)}</span>
                        </div>
                    </div>
                `;
            }
            html += `
                    </div>
                    <textarea id="preference-input" class="preference-input" placeholder="Describe any preferences..."></textarea>
                </div>
            `;
        } else {
            html += `
                <div class="routing-section">
                    <h4>2. Configure Each Step</h4>
                    <p class="section-note">Select a model and optionally add preferences for each step:</p>
                    <div class="steps-config-list">
            `;

        for (const step of steps) {
            html += `
                <div class="step-config-item" data-step="${step.step_number}">
                    <div class="step-config-header">
                        <span class="step-config-number">Step ${step.step_number}</span>
                        <span class="step-config-task">${escapeHtml(step.task)}</span>
                    </div>
                    <div class="step-config-body">
                        <div class="step-model-select">
                            <label>Model:</label>
                            <select class="step-model-dropdown" data-step="${step.step_number}">
                                <option value="">Auto-select (semantic)</option>
            `;
            for (const model of currentState.availableModels) {
                html += `<option value="${escapeHtml(model.name)}">${escapeHtml(model.name)}</option>`;
            }
            html += `
                            </select>
                        </div>
                        <div class="step-preference-input">
                            <label>Preference:</label>
                            <input type="text" class="step-preference" data-step="${step.step_number}" 
                                   placeholder="e.g., 'fast response', 'detailed output'...">
                        </div>
                    </div>
                </div>
            `;
        }

        html += `
                </div>
            </div>
        `;
        }  // Close the else block for steps.length > 0
    } else {
        // Single-shot: Show single model selection
        html += `
            <div class="routing-section">
                <h4>2. Select a Model</h4>
                <div class="models-selection-grid">
        `;

        for (const model of currentState.availableModels) {
            html += `
                <div class="model-select-option" data-model="${escapeHtml(model.name)}">
                    <div class="model-select-name">${escapeHtml(model.name)}</div>
                    <div class="model-select-desc">${escapeHtml(model.description)}</div>
                    <div class="model-select-meta">
                        <span class="model-economy">${escapeHtml(model.economy)}</span>
                        <span class="model-responsiveness">${escapeHtml(model.responsiveness || model.latency || '')}</span>
                    </div>
                </div>
            `;
        }

        html += `
                </div>
                <div class="auto-select-note">
                    Or leave unselected for automatic semantic selection
                </div>
            </div>
        `;

        // Second prompt input for single-shot
        html += `
            <div class="routing-section">
                <h4>3. Additional Preferences (Optional)</h4>
                <textarea id="preference-input" class="preference-input" placeholder="Describe any preferences for model selection (e.g., 'prefer fast response', 'need high accuracy', 'budget friendly')..."></textarea>
            </div>
        `;
    }

    // Execute buttons
    html += `
        <div class="routing-section">
            <div class="execute-actions">
                <button id="execute-selected-btn" class="action-btn primary">Execute</button>
                <button id="back-btn" class="action-btn secondary">Back</button>
            </div>
        </div>
    `;

    html += '</div>';
    routingDisplay.innerHTML = html;

    // Add event listeners for model selection (single-shot or multi-step without steps)
    if (!isMultiStep || steps.length === 0) {
        document.querySelectorAll('.model-select-option').forEach(el => {
            el.addEventListener('click', () => {
                document.querySelectorAll('.model-select-option').forEach(opt => opt.classList.remove('selected'));
                el.classList.add('selected');
                currentState.selectedModelName = el.dataset.model;
            });
        });
    }

    document.getElementById('execute-selected-btn').addEventListener('click', executeWithSelectedModel);
    document.getElementById('back-btn').addEventListener('click', () => {
        currentState.phase = 'classified';
        currentState.selectedModelName = null;
        renderClassifierInteraction();
    });
}

async function executeWithSelectedModel() {
    const isMultiStep = currentState.classification.plan_type === 'multi_step';
    const steps = currentState.classification.steps || [];
    
    let requestBody = {
        original_prompt: currentState.originalPrompt,
        classification: currentState.classification,
    };

    if (isMultiStep && steps.length > 0) {
        // Collect per-step selections
        const stepSelections = [];
        for (const step of steps) {
            const stepNum = step.step_number;
            const modelDropdown = document.querySelector(`.step-model-dropdown[data-step="${stepNum}"]`);
            const preferenceInput = document.querySelector(`.step-preference[data-step="${stepNum}"]`);
            
            stepSelections.push({
                step_number: stepNum,
                selected_model_name: modelDropdown?.value || null,
                user_preference: preferenceInput?.value || '',
            });
        }
        requestBody.step_selections = stepSelections;
    } else {
        // Single-shot: use global selection
        const preference = document.getElementById('preference-input')?.value || '';
        requestBody.selected_model_name = currentState.selectedModelName;
        requestBody.user_preference = preference;
    }

    setLoading(true, 'Executing...');
    executionDisplay.innerHTML = '<p class="loading">Selecting model and executing...</p>';
    responseDisplay.innerHTML = '';

    try {
        const response = await fetch(`${API_BASE_URL}/select-model`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error: ${response.status}`);
        }

        const data = await response.json();
        console.log('Execution result:', data);

        currentState.phase = 'complete';
        renderExecutionResult(data);
    } catch (error) {
        console.error('Execution error:', error);
        showError(`Error: ${error.message}`);
    } finally {
        setLoading(false);
    }
}

// ============== PHASE 2B: Tournament Mode ==============

async function runTournamentMode() {
    setLoading(true, 'Running tournament...');
    executionDisplay.innerHTML = '<p class="loading">Running tournament across all models...</p>';
    responseDisplay.innerHTML = '';

    try {
        const response = await fetch(`${API_BASE_URL}/route`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: [{ role: 'user', content: currentState.originalPrompt }],
                tournament: true,
                tournament_per_step: true,
                judge_model: 'gpt-4o-mini',
            }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error: ${response.status}`);
        }

        const data = await response.json();
        console.log('Tournament result:', data);

        currentState.phase = 'complete';
        renderExecutionResult(data);
    } catch (error) {
        console.error('Tournament error:', error);
        showError(`Error: ${error.message}`);
    } finally {
        setLoading(false);
    }
}

// ============== Execution Result Rendering ==============

function renderExecutionResult(data) {
    // Update routing display with final selection info
    let routingHtml = '<div class="routing-info">';

    // Classification summary
    routingHtml += `
        <div class="routing-section">
            <h4>1. Classification</h4>
            <div class="classification-summary">
                <div class="info-row">
                    <span class="label">Plan Type:</span>
                    <span class="value">${data.plan_type}</span>
                </div>
            </div>
        </div>
    `;

    // Model selection
    routingHtml += `
        <div class="routing-section">
            <h4>2. Model Selection</h4>
            <div class="info-row">
                <span class="label">Selected Model:</span>
                <span class="value model-name">${extractModelName(data.selection.model)}</span>
            </div>
            ${data.selection.reasoning ? `
            <div class="info-row">
                <span class="label">Reasoning:</span>
                <span class="value reasoning">${escapeHtml(data.selection.reasoning)}</span>
            </div>
            ` : ''}
        </div>
    `;

    // Semantic selection details (if available)
    if (data.metadata?.semantic_selection) {
        const ss = data.metadata.semantic_selection;
        routingHtml += `
            <div class="routing-section">
                <h4>3. Semantic Selection Details</h4>
                <div class="info-row">
                    <span class="label">Method:</span>
                    <span class="value">${escapeHtml(ss.method)}</span>
                </div>
                <div class="info-row">
                    <span class="label">Selected:</span>
                    <span class="value">${escapeHtml(ss.selected_model)}</span>
                </div>
                ${ss.raw_response ? `
                <div class="detail-block">
                    <div class="detail-label">Selector Response:</div>
                    <pre class="classifier-response">${escapeHtml(JSON.stringify(ss.raw_response, null, 2))}</pre>
                </div>
                ` : ''}
            </div>
        `;
    }

    // Tournament results (if available)
    if (data.tournament) {
        routingHtml += renderTournamentSection(data.tournament);
    }

    routingHtml += '</div>';
    routingDisplay.innerHTML = routingHtml;

    // Render execution
    renderExecutionStep(data);

    // Render response
    renderResponseStep(data);
}

function renderTournamentSection(tournament) {
    const judge = tournament.judge;
    const winnerModel = judge?.winner_model;

    let html = `
        <div class="routing-section">
            <h4>Tournament Results</h4>
    `;

    if (judge) {
        html += `
            <div class="judge-block">
                <div class="info-row">
                    <span class="label">Judge:</span>
                    <span class="value">${escapeHtml(extractModelName(judge.model))}</span>
                </div>
                <div class="info-row">
                    <span class="label">Winner:</span>
                    <span class="value winner-highlight">${escapeHtml(extractModelName(judge.winner_model))}</span>
                </div>
                <div class="detail-block">
                    <div class="detail-label">Judge Reasoning:</div>
                    <pre class="judge-reasoning">${escapeHtml(judge.reasoning || '')}</pre>
                </div>
            </div>
        `;
    }

    html += '<div class="candidates-list">';
    for (const c of tournament.candidates || []) {
        const isWinner = winnerModel && c.model === winnerModel;
        const status = c.error ? 'failed' : 'completed';
        html += `
            <div class="candidate-item ${status} ${isWinner ? 'winner' : ''}">
                <div class="candidate-header">
                    <span class="candidate-model">${escapeHtml(extractModelName(c.model))}${isWinner ? ' ✓ WINNER' : ''}</span>
                    ${c.latency_ms ? `<span class="candidate-latency">${formatLatency(c.latency_ms)}</span>` : ''}
                    ${c.usage?.total_tokens ? `<span class="candidate-tokens">${c.usage.total_tokens} tokens</span>` : ''}
                </div>
                ${c.error ? `<div class="candidate-error">${escapeHtml(c.error)}</div>` : `<pre class="candidate-output">${escapeHtml(c.content || '')}</pre>`}
            </div>
        `;
    }
    html += '</div></div>';

    return html;
}

function renderExecutionStep(data) {
    const metadata = data.metadata || {};

    if (data.plan_type === 'single_shot') {
        let html = `
            <div class="execution-single">
                <div class="info-row">
                    <span class="label">Execution Type:</span>
                    <span class="value">Single Shot</span>
                </div>
        `;

        if (metadata.latency_ms) {
            html += `
                <div class="info-row">
                    <span class="label">Latency:</span>
                    <span class="value latency">${formatLatency(metadata.latency_ms)}</span>
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
                    <span class="value">Multi-Step (${data.steps?.length || 0} steps)</span>
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

        for (const step of data.steps || []) {
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
                        <span class="step-model-label">Model:</span>
                        <span class="step-model">${extractModelName(step.model)}</span>
                        ${stepLatency ? `<span class="step-latency">${formatLatency(stepLatency)}</span>` : ''}
                    </div>
                </div>
            `;
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
        const completedSteps = data.steps.filter(s => s.status === 'completed');
        if (completedSteps.length > 0) {
            html = '<div class="final-response">';
            html += `<p class="response-note">Output from ${completedSteps.length} step(s):</p>`;
            for (const step of completedSteps) {
                html += `
                    <div class="step-output-block">
                        <div class="step-output-header">Step ${step.step_number}: ${escapeHtml(step.task)}</div>
                        <pre>${escapeHtml(step.output || 'No output')}</pre>
                    </div>
                `;
            }
            html += '</div>';
        } else {
            html = '<p class="no-response">No completed steps</p>';
        }
    } else {
        html = '<p class="no-response">No response generated</p>';
    }

    responseDisplay.innerHTML = html;
}

// ============== Event Listeners ==============

submitBtn.addEventListener('click', classifyPrompt);

promptInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && event.ctrlKey) {
        classifyPrompt();
    }
});
