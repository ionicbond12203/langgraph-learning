# LangGraph Learning - Multi-Agent Travel Planner

A complete tutorial project demonstrating how to build a **multi-agent system** using [LangGraph](https://github.com/langchain-ai/langgraph) with a local LLM (Ollama + Qwen 2.5).

This project creates a travel planning assistant that coordinates multiple AI agents to search for flights, hotels, and travel guides, then compiles everything into a comprehensive travel itinerary.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Step-by-Step Installation](#step-by-step-installation)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [How It Works](#how-it-works)
- [Code Walkthrough](#code-walkthrough)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Request                             │
│              "I want to travel to Tokyo next month"             │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Manager Agent  │  ← Parses request, extracts
                    │   (Qwen 2.5)    │    origin, destination, date
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
  │ Flight Agent  │  │  Hotel Agent  │  │  Guide Agent  │
  │ (DuckDuckGo)  │  │ (DuckDuckGo)  │  │ (DuckDuckGo)  │
  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
          │                  │                  │
          │     Real-time Internet Search       │
          └──────────────────┼──────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Reporter Agent  │  ← Compiles all data into
                    │   (Qwen 2.5)    │    a detailed travel plan
                    └────────┬────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │   Final Travel Itinerary │
              │   (Budget, Day-by-Day)   │
              └──────────────────────────┘
```

**Key Features:**
- 🤖 **Multi-Agent Coordination**: Manager delegates tasks to specialized agents
- 🌐 **Real-Time Search**: Uses DuckDuckGo to fetch live flight/hotel prices
- 🏠 **100% Local LLM**: Runs entirely on your machine with Ollama
- 📊 **LangGraph Studio**: Visual debugging and monitoring UI

---

## Prerequisites

Before starting, make sure you have:

### 1. Python 3.10 or higher
```bash
python --version  # Should show 3.10+
```

### 2. Ollama (Local LLM Runtime)

Ollama allows you to run large language models locally.

**Installation:**
- **Windows**: Download from [ollama.ai/download](https://ollama.ai/download)
- **Mac**: `brew install ollama`
- **Linux**: `curl -fsSL https://ollama.ai/install.sh | sh`

**Pull the Qwen 2.5 model (14B recommended):**
```bash
ollama pull qwen2.5:14b
```

> **Note**: The 14B model requires ~10GB RAM. For lower-end machines, try `qwen2.5:7b` instead.

### 3. Git
```bash
git --version
```

---

## Step-by-Step Installation

### Step 1: Clone this repository

```bash
git clone https://github.com/ionicbond12203/langgraph-learning.git
cd langgraph-learning
```

### Step 2: Create a virtual environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
# Install the project in editable mode
pip install -e .

# Install the DuckDuckGo search package (required for web search)
pip install -U ddgs

# Install LangGraph CLI for the dev server
pip install "langgraph-cli[inmem]"
```

### Step 4: Verify installation

```bash
# Check if LangGraph CLI is installed
langgraph --version
```

---

## Configuration

### Step 1: Create your environment file

```bash
# Copy the example file
cp .env.example .env
```

### Step 2: (Optional) Configure LangSmith

LangSmith provides observability and debugging. It's optional but highly recommended.

1. Sign up at [smith.langchain.com](https://smith.langchain.com)
2. Get your API key from Settings
3. Add to `.env`:

```env
LANGSMITH_API_KEY=your_api_key_here
LANGSMITH_PROJECT=langgraph-travel-agent
```

---

## Running the Project

### Step 1: Start Ollama (if not running)

Make sure the Ollama service is running:
```bash
# On Windows, Ollama runs as a service automatically
# On Mac/Linux, you may need to start it:
ollama serve
```

Verify the model is available:
```bash
ollama list  # Should show qwen2.5:14b
```

### Step 2: Start the LangGraph dev server

```bash
langgraph dev
```

You should see output like:
```
        Welcome to

╦  ┌─┐┌┐┌┌─┐╔═╗┬─┐┌─┐┌─┐┬ ┬
║  ├─┤││││ ┬║ ╦├┬┘├─┤├─┘├─┤
╩═╝┴ ┴┘└┘└─┘╚═╝┴└─┴ ┴┴  ┴ ┴

- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

### Step 3: Open LangGraph Studio

Click the Studio UI link or go to:
```
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### Step 4: Test the Travel Agent

1. In the Studio, select the **"travel"** graph
2. Click **"New Thread"**
3. Enter a travel request:
   ```
   I want to travel to Tokyo next month
   ```
4. Watch the agents work in real-time!

---

## How It Works

### The Multi-Agent Flow

1. **Manager Agent** receives your request and extracts:
   - `origin`: Departure city (defaults to "Kuala Lumpur")
   - `destination`: Target city
   - `date`: Travel date

2. **Three specialized agents run in parallel:**
   - **Flight Agent**: Searches for cheap flights on Skyscanner/Trip.com
   - **Hotel Agent**: Searches for hotels on Booking.com/Agoda
   - **Guide Agent**: Finds attractions, food, and itineraries

3. **Reporter Agent** takes all the search results and creates a comprehensive travel plan with:
   - Budget estimation
   - Day-by-day itinerary
   - Food recommendations

### LangGraph State Management

The agents communicate through a shared `TravelState`:

```python
class TravelState(TypedDict, total=False):
    request: str        # Original user request
    origin: str         # Departure city
    destination: str    # Target city
    date: str           # Travel date
    flight_info: str    # Flight search results
    hotel_info: str     # Hotel search results
    guide_info: str     # Guide search results
    final_plan: str     # Final compiled report
```

---

## Code Walkthrough

### Key File: `src/agent/travel.py`

#### 1. Tool Definitions (Lines 29-77)

```python
@tool
def search_flights(origin: str, destination: str, date: str):
    """Search for real-time flight prices using DuckDuckGo."""
    query = f"cheap flight ticket price from {origin} to {destination} on {date}"
    return web_search.invoke(query)

@tool
def search_hotels(city: str, check_in: str):
    """Search for hotel prices on Booking.com/Agoda."""
    query = f"budget hotel prices in {city} on {check_in} booking.com agoda"
    return web_search.invoke(query)
```

#### 2. Agent Nodes (Lines 105-206)

Each agent is a function that:
- Takes the current state
- Performs its task (LLM reasoning or tool call)
- Returns updated state

```python
def flight_agent_node(state: TravelState):
    agent = llm.bind_tools(flight_tools)
    msg = f"Find cheap flights from {state['origin']} to {state['destination']}"
    response = agent.invoke(msg)
    
    # Failsafe: Force tool call if LLM doesn't call it
    if response.tool_calls:
        result = search_flights.invoke(response.tool_calls[0]['args'])
    else:
        result = search_flights.invoke({...})
    
    return {"flight_info": result}
```

#### 3. Graph Construction (Lines 211-234)

```python
builder = StateGraph(TravelState)

# Add nodes
builder.add_node("manager", manager_node)
builder.add_node("flight_agent", flight_agent_node)
builder.add_node("hotel_agent", hotel_agent_node)
builder.add_node("guide_agent", guide_agent_node)
builder.add_node("reporter", reporter_node)

# Add edges (parallel execution)
builder.add_edge(START, "manager")
builder.add_edge("manager", "flight_agent")
builder.add_edge("manager", "hotel_agent")
builder.add_edge("manager", "guide_agent")

# All agents converge to reporter
builder.add_edge("flight_agent", "reporter")
builder.add_edge("hotel_agent", "reporter")
builder.add_edge("guide_agent", "reporter")

builder.add_edge("reporter", END)

graph = builder.compile()
```

---

## Customization

### Change the LLM Model

Edit `src/agent/travel.py` line 19:
```python
# Use a smaller model for lower-end machines
llm = ChatOllama(model="qwen2.5:7b", temperature=0)

# Or use a different model entirely
llm = ChatOllama(model="llama3.2:3b", temperature=0)
```

### Modify Default Origin City

Edit the `manager_node` function (line 111):
```python
# Change default origin from 'Kuala Lumpur' to your city
"origin": data.get("origin", "Singapore")
```

### Add More Search Sources

Modify the search queries in the tool functions:
```python
query = f"cheap flight ticket price from {origin} to {destination} kayak expedia"
```

---

## Troubleshooting

### Error: `Could not import ddgs python package`

```bash
pip install -U ddgs
```

### Error: `Connection refused` when running langgraph dev

Make sure Ollama is running:
```bash
ollama serve
```

### Error: Model not found

Pull the required model:
```bash
ollama pull qwen2.5:14b
```

### Slow response times

The 14B model can be slow on older hardware. Try:
```bash
ollama pull qwen2.5:7b
```
Then update `travel.py` to use `qwen2.5:7b`.

### Search results are empty or timeout

DuckDuckGo may rate-limit requests. Wait a few minutes and try again.

---

## Project Structure

```
langgraph-learning/
├── src/
│   └── agent/
│       ├── __init__.py       # Package init
│       ├── travel.py         # Main travel agent (flights, hotels, guides)
│       └── graph.py          # Base agent template
├── langgraph.json            # LangGraph configuration
├── pyproject.toml            # Python dependencies
├── .env.example              # Environment template
├── .gitignore                # Git ignore rules
├── Makefile                  # Build commands
└── README.md                 # This file
```

---

## Next Steps

Once you understand this project, try:

1. **Add more agents**: Weather agent, currency exchange agent
2. **Add memory**: Use LangGraph checkpointing to remember preferences
3. **Deploy to cloud**: Use LangGraph Cloud for production deployment
4. **Switch to OpenAI**: Replace Ollama with `ChatOpenAI` for faster responses

---

## License

MIT License - feel free to use this for learning and commercial projects.

---

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [Ollama Models](https://ollama.ai/library)
- [LangSmith](https://smith.langchain.com)
