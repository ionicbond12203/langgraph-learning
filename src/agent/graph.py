import os
import operator
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv

# LangChain / LangGraph 核心组件
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition  # <--- 预构建的工具节点和路由条件

# Alpaca 组件
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

# 加载环境变量
load_dotenv()


# ==========================================
# 1. 定义工具 (The Tools)
# ==========================================
@tool
def get_stock_price(symbol: str):
    """
    查询股票的当前价格。
    Args:
        symbol: 股票代码，例如 NVDA, AAPL, TSLA (必须大写)
    """
    try:
        # 初始化 Alpaca 客户端
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            return "错误：未找到 Alpaca API Key，请检查 .env 文件。"

        client = StockHistoricalDataClient(api_key, secret_key)

        # 获取最新报价
        request_params = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = client.get_stock_latest_quote(request_params)

        # 提取价格
        price = quote[symbol].ask_price
        return f"{symbol} 的当前价格是 ${price}"

    except Exception as e:
        return f"查询失败: {str(e)}"


# 将工具放入列表
tools = [get_stock_price]

# ==========================================
# 2. 定义模型并绑定工具 (Bind Tools)
# ==========================================
# 你的本地 Qwen 模型
llm = ChatOllama(model="qwen2.5:14b", temperature=0)  # 温度设低点，调用工具更准

# 关键步骤：告诉 LLM 它有哪些工具可用
llm_with_tools = llm.bind_tools(tools)


# ==========================================
# 3. 定义图的状态与节点
# ==========================================
class State(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage | ToolMessage], operator.add]


def chatbot_node(state: State):
    # 这里我们调用的是 "绑定了工具的 LLM"
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# ==========================================
# 4. 构建图 (The Graph)
# ==========================================
builder = StateGraph(State)

# 添加节点
builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools))  # <--- LangGraph 自带的工具执行节点

# 设置连线
builder.add_edge(START, "chatbot")

# 关键路由：tools_condition
# 它会自动判断：
# 1. 如果 LLM 想调用工具 -> 走到 "tools" 节点
# 2. 如果 LLM 只是普通聊天 -> 走到 END
builder.add_conditional_edges("chatbot", tools_condition)

# 如果执行完工具，把结果扔回给 chatbot，让它生成最终回复
builder.add_edge("tools", "chatbot")

# 编译
graph = builder.compile()