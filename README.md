# LLM Router API

An intelligent LLM routing system that automatically classifies queries and selects the most appropriate model based on complexity.

## Features

- **Intelligent Query Classification**: Automatically analyzes prompts to determine complexity (simple, standard, complex, code)
- **Cost-Optimized Routing**: Routes simple queries to cheaper models, complex queries to more capable models
- **Multi-Step Execution**: Supports breaking down complex tasks into sequential steps with dependency tracking
- **Model Tiers**:
  - Simple queries -> Llama 3.1 8B (fast, economical)
  - Standard queries -> Claude 3.5 Sonnet (balanced)
  - Complex queries -> Claude 3 Opus (most capable)
  - Code queries -> DeepSeek Coder (specialized)

## Setup

### Prerequisites

- Python 3.11+
- OpenAI API key (for meta-router classification)
- OpenRouter API key (for LLM execution)

### Installation

1. Clone the repository and navigate to the project directory:

```bash
cd llm_router_api
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=sk-your-openai-key
OPENROUTER_API_KEY=sk-or-your-openrouter-key
```

### Running the Server

Start the development server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Access the application:
- **Frontend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## API Endpoints

### POST /api/prompt

Simple endpoint for the frontend. Accepts a prompt string and returns the routed response.

**Request:**
```json
{
  "prompt": "What is 2 + 2?"
}
```

**Response:**
```json
{
  "response": "2 + 2 = 4",
  "model": "openrouter/meta-llama/llama-3.1-8b-instruct",
  "plan_type": "single_shot",
  "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35}
}
```

### POST /route

Full routing endpoint with advanced options.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Write a Python function to sort a list"}
  ],
  "temperature": 0.7,
  "max_tokens": 2000
}
```

## Demo Prompts

Try these prompts to see the router in action:

1. **Simple Q&A** (routes to Llama):
   - "What is the capital of France?"
   - "What year did World War II end?"

2. **Code Generation** (routes to DeepSeek):
   - "Write a Python function to check if a number is prime"
   - "Create a TypeScript interface for a user profile"

3. **Complex Analysis** (routes to Claude Opus):
   - "Analyze the pros and cons of microservices vs monolithic architecture for a startup"
   - "Explain the implications of quantum computing on current encryption standards"

## Architecture

```
User Prompt
    │
    ▼
┌─────────────────┐
│  Meta-Router    │  ← Classifies query complexity
│  (GPT-4o-mini)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Selection │  ← Maps complexity to model tier
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Executor     │  ← Runs selected model via OpenRouter
│  (OpenRouter)   │
└────────┬────────┘
         │
         ▼
    Response
```

## AI Usage Note

This project was developed with AI assistance. The codebase includes:
- Automated query classification using GPT-4o-mini
- LLM execution through OpenRouter's unified API
- Cost optimization through intelligent model selection

The routing logic aims to balance response quality with cost efficiency by matching query complexity to appropriate model capabilities.

## License

See LICENSE file for details.
