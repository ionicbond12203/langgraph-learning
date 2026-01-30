# LangGraph Learning - Travel Agent

A multi-agent travel planning system built with LangGraph and Ollama (Qwen 2.5).

## Features

- **Manager Agent**: Parses user requests and extracts travel details
- **Flight Agent**: Searches for real-time flight prices via DuckDuckGo
- **Hotel Agent**: Finds hotel prices from Booking.com/Agoda
- **Guide Agent**: Retrieves travel guides, attractions, and food recommendations
- **Reporter Agent**: Compiles a comprehensive travel itinerary

## Architecture

```
User Request
     │
     ▼
  Manager
     │
     ├──────────────────┬──────────────────┐
     ▼                  ▼                  ▼
Flight Agent      Hotel Agent        Guide Agent
     │                  │                  │
     └──────────────────┴──────────────────┘
                        │
                        ▼
                    Reporter
                        │
                        ▼
               Final Travel Plan
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) with Qwen 2.5 model installed
- LangGraph CLI

## Installation

```bash
# Clone the repository
git clone https://github.com/ionicbond12203/langgraph-learning.git
cd langgraph-learning

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e ".[dev]"

# Install DuckDuckGo search
pip install -U ddgs
```

## Configuration

1. Copy `.env.example` to `.env`
2. Configure your LangSmith API key (optional)

```bash
cp .env.example .env
```

## Running

```bash
# Make sure Ollama is running with Qwen 2.5
ollama run qwen2.5:14b

# Start the LangGraph dev server
langgraph dev
```

Then open the Studio UI: https://smith.langchain.com/studio

## Usage Example

Input:
```
I want to travel to Tokyo next month
```

The system will:
1. Parse your request (origin: Kuala Lumpur, destination: Tokyo, date: Next Month)
2. Search for flight prices
3. Search for hotel prices
4. Find travel guides and attractions
5. Generate a comprehensive travel plan

## Project Structure

```
├── src/
│   └── agent/
│       ├── travel.py      # Travel agent graph (flights, hotels, guides)
│       └── graph.py       # Base agent graph
├── langgraph.json         # LangGraph configuration
├── pyproject.toml         # Python dependencies
└── .env.example           # Environment template
```

## License

MIT
