# LangGraph Advanced Chatbot

## Setup

1. Clone this repository or copy the project files.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root and add your API keys:
   ```env
   TAVILY_API_KEY=your_tavily_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

Run the chatbot from the command line:
```bash
python main.py
```

## Features
- Conversational AI using LangChain and OpenAI
- Tool integration (e.g., TavilySearch)
- Human approval for tool usage

## Deployment
- Ensure all environment variables are set in `.env`
- Use a process manager (e.g., supervisord) for production

## Troubleshooting
- If you see errors about missing API keys, check your `.env` file.
- For dependency issues, re-run `pip install -r requirements.txt`.

