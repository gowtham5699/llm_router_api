const promptInput = document.getElementById('prompt-input');
const submitBtn = document.getElementById('submit-btn');
const responseDisplay = document.getElementById('response-display');

const API_BASE_URL = '/api';

function setResponse(content, className = 'content') {
    responseDisplay.innerHTML = `<p class="${className}">${content}</p>`;
}

function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    if (isLoading) {
        setResponse('Processing...', 'loading');
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
            throw new Error(`HTTP error: ${response.status}`);
        }

        const data = await response.json();
        setResponse(data.response || JSON.stringify(data, null, 2));
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
