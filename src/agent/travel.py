import json
import operator
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END

# 1. Load Environment Variables
load_dotenv()

# 2. Initialize Model (Qwen 2.5)
# temperature=0 ensures the tool calling is precise and stable.
llm = ChatOllama(model="qwen2.5:14b", temperature=0)

# Initialize the Search Tool (Our "Eyes")
web_search = DuckDuckGoSearchRun()


# ==========================================
# 3. Define Tools (Real-Time Internet Search)
# ==========================================

@tool
def search_flights(origin: str, destination: str, date: str):
    """
    Search for real-time flight ticket prices using the internet.
    Args:
        origin: Departure city (e.g., Kuala Lumpur)
        destination: Destination city (e.g., Tokyo)
        date: Travel date (e.g., Next Month)
    """
    query = f"cheap flight ticket price from {origin} to {destination} on {date} skyscanner trip.com"
    print(f"\n✈️ [Internet] Searching flights: {query}...")
    try:
        # Actually search on DuckDuckGo
        return web_search.invoke(query)
    except Exception as e:
        return f"Search timed out. Error: {e}"


@tool
def search_hotels(city: str, check_in: str):
    """
    Search for real-time hotel prices using the internet.
    Args:
        city: Destination city
        check_in: Check-in date
    """
    query = f"budget hotel prices in {city} on {check_in} booking.com agoda"
    print(f"\n🏨 [Internet] Searching hotels: {query}...")
    try:
        return web_search.invoke(query)
    except Exception as e:
        return f"Search timed out. Error: {e}"


@tool
def get_travel_guide(city: str):
    """
    Search for travel guides, attractions, and food.
    """
    query = f"{city} travel guide must visit places best food 3 days itinerary"
    print(f"\n🗺️ [Internet] Searching guide: {city}...")
    try:
        result = web_search.invoke(query)
        # ✂️ Truncate if result is too long to prevent local model crash
        if len(result) > 2000:
            return result[:2000] + "...(content truncated)"
        return result
    except Exception as e:
        return "Search timed out."


# Bind tools to a list
flight_tools = [search_flights]
hotel_tools = [search_hotels]
guide_tools = [get_travel_guide]


# ==========================================
# 4. Define State
# ==========================================
# total=False allows us to start with an empty state
class TravelState(TypedDict, total=False):
    request: str  # Original user request
    origin: str  # Departure city
    destination: str  # Destination city
    date: str  # Travel date
    flight_info: str  # Search result for flights
    hotel_info: str  # Search result for hotels
    guide_info: str  # Search result for guides
    final_plan: str  # Final report


# ==========================================
# 5. Define Agents (Nodes)
# ==========================================

def manager_node(state: TravelState):
    print("\n🤖 Manager is analyzing the request...")
    prompt = f"""
    User Request: "{state['request']}"

    Please extract the following information and return it in JSON format:
    1. origin (Departure city. If not specified, default to 'Kuala Lumpur')
    2. destination (Target city)
    3. date (Travel date. If not specified, default to 'Next Month')

    Return ONLY JSON. Do not include Markdown formatting like ```json.
    """
    try:
        response = llm.invoke(prompt).content
        # Clean up potential markdown code blocks
        clean_json = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        return {
            "origin": data.get("origin", "Kuala Lumpur"),
            "destination": data.get("destination"),
            "date": data.get("date")
        }
    except:
        print("⚠️ Manager failed to parse JSON. Using default values...")
        return {"origin": "Kuala Lumpur", "destination": "Tokyo", "date": "Next Month"}


def flight_agent_node(state: TravelState):
    agent = llm.bind_tools(flight_tools)
    msg = f"Find cheap flights from {state['origin']} to {state['destination']} on {state['date']}."

    response = agent.invoke(msg)

    # === Failsafe Mechanism ===
    # If the AI forgets to call the tool, we force it here.
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        result = search_flights.invoke(tool_call['args'])
    else:
        print("⚠️ Flight Agent forgot to call tool. Forcing execution...")
        result = search_flights.invoke({
            "origin": state['origin'],
            "destination": state['destination'],
            "date": state['date']
        })

    return {"flight_info": result}


def hotel_agent_node(state: TravelState):
    agent = llm.bind_tools(hotel_tools)
    msg = f"Find cheap hotels in {state['destination']} on {state['date']}."

    response = agent.invoke(msg)

    # === Failsafe Mechanism ===
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        result = search_hotels.invoke(tool_call['args'])
    else:
        print("⚠️ Hotel Agent forgot to call tool. Forcing execution...")
        result = search_hotels.invoke({
            "city": state['destination'],
            "check_in": state['date']
        })

    return {"hotel_info": result}


def guide_agent_node(state: TravelState):
    # For the guide, we skip LLM thinking and force a search directly.
    # This is faster and prevents errors.
    result = get_travel_guide.invoke({"city": state['destination']})
    return {"guide_info": result}


def reporter_node(state: TravelState):
    print("\n📝 Writing detailed travel plan (Watch your terminal)...")

    prompt = f"""
    You are a Senior Travel Planner. 
    Based on the REAL search data below, write a **detailed** travel itinerary.

    【Flight Data】: {state['flight_info']}
    【Hotel Data】: {state['hotel_info']}
    【Guide Data】: {state['guide_info']}

    Requirements:
    1. **Budget Estimation**: Calculate the total estimated cost based on the numbers found in the search results. If currencies are mixed (e.g., USD, MYR, RUB), try to convert or mention them clearly.
    2. **Detailed Itinerary**: Do NOT just list bullet points. Describe the trip day-by-day.
       - For each day, include: Morning Activity, Afternoon Activity, Evening Activity, and Food Recommendations.
       - Use the 'Guide Data' to find real attraction names.
    3. **Tone**: Enthusiastic, professional, and helpful.
    4. **Correction**: If the flight search results look weird (e.g., wrong location), mention it in the report: "Note: Search results might be inaccurate for this specific route, please check Skyscanner manually."

    Please write a comprehensive report (at least 600 words):
    """

    # Stream the output to the terminal
    response = llm.invoke(prompt, config={"callbacks": [StreamingStdOutCallbackHandler()]})

    return {"final_plan": response.content}


# ==========================================
# 6. Graph Construction
# ==========================================
builder = StateGraph(TravelState)

# Add Nodes
builder.add_node("manager", manager_node)
builder.add_node("flight_agent", flight_agent_node)
builder.add_node("hotel_agent", hotel_agent_node)
builder.add_node("guide_agent", guide_agent_node)
builder.add_node("reporter", reporter_node)

# Add Edges (Manager -> 3 Agents -> Reporter)
builder.add_edge(START, "manager")
builder.add_edge("manager", "flight_agent")
builder.add_edge("manager", "hotel_agent")
builder.add_edge("manager", "guide_agent")

builder.add_edge("flight_agent", "reporter")
builder.add_edge("hotel_agent", "reporter")
builder.add_edge("guide_agent", "reporter")

builder.add_edge("reporter", END)

# Compile
graph = builder.compile()