# app1.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import faiss

# ==================== IMPORTS ====================
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
from operator import add
from duckduckgo_search import DDGS

app = Flask(__name__)

# ==================== CONFIG ====================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "dataset_indexq.faiss"
METADATA_CSV = "metadata_with_embeddings1.csv"
OLLAMA_MODEL = "llama3.2"

df = pd.read_csv(METADATA_CSV)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
index = faiss.read_index(FAISS_INDEX_PATH)

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.7)

chat_history = []

# ==================== TOOLS ====================
@tool
def retrieve_and_format(query: str, top_k: int = 5) -> str:
    """Retrieve top matching Kaggle datasets using semantic search. Always return this output."""
    query_emb = np.array([embeddings.embed_query(query)], dtype="float32")
    _, indices = index.search(query_emb, top_k)
    top_results = df.iloc[indices[0]]

    header = "🔍 **Top 5 Matching Kaggle Datasets (Semantic Search)**\n\n"
    blocks = []
    for i, (_, row) in enumerate(top_results.iterrows(), 1):
        name = row.get('dataset_name', 'Unknown')
        url = row.get('dataset_url', 'N/A')
        cat = row.get('category2', 'Unknown')
        desc = row.get('about_dataset', 'No description')[:220] + "..."
        block = f"{i}. **{name}**\n   🔗 {url}\n   🗂️ {cat}\n   📝 {desc}\n"
        blocks.append(block)
    return header + "\n".join(blocks)

@tool
def web_search(query: str) -> str:
    """Search the web for tutorials, code examples, ML techniques."""
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=6)]
    formatted = "\n".join([f"- {r['title']}: {r['href']}" for r in results])
    return f"🌐 **Web Search Results** for '{query}':\n{formatted or 'No good results.'}"

tools = [retrieve_and_format, web_search]

# ==================== SYSTEM PROMPT – VERY STRICT ====================
system_prompt = """
You are an expert Data Science Project Assistant.

MANDATORY RULES – YOU MUST OBEY THESE EVERY TIME:
1. For ANY project, idea, dataset request → ALWAYS call the retrieve_and_format tool FIRST
2. ALWAYS include the FULL output of the retrieve_and_format tool at the VERY TOP of your final answer
3. Never summarize or skip the top 5 datasets — copy-paste the exact output
4. After the top 5 list, explain which one is best and why
5. Then give complete step-by-step project guidance with code snippets

Response structure MUST start like this:
🔍 **Top 5 Matching Kaggle Datasets (Semantic Search)**

1. **Name**
   🔗 url
   ...

Best Dataset: ...
Project Steps:
...
"""

# ==================== LANGGRAPH AGENT ====================
class AgentState(TypedDict):
    messages: Annotated[List, add]

def agent_node(state):
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state):
    results = []
    last_msg = state["messages"][-1]
    for call in last_msg.tool_calls:
        tool_name = call["name"]
        args = call["args"]
        tool_func = next((t for t in tools if t.name == tool_name), None)
        if tool_func:
            try:
                result = tool_func.invoke(args)
                results.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
            except Exception as e:
                results.append(ToolMessage(content=f"Tool error: {str(e)}", tool_call_id=call["id"]))
        else:
            results.append(ToolMessage(content=f"Tool '{tool_name}' not found", tool_call_id=call["id"]))
    return {"messages": results}

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls else END)
graph.add_edge("tools", "agent")
agent_app = graph.compile()

# ==================== FOLLOW-UP PREDICTION ====================
prediction_chain = (ChatPromptTemplate.from_messages([
    ("system", "Suggest 3 short, natural follow-up questions. Number them 1-3."),
    ("placeholder", "{chat_history}"),
    ("human", "User: {input}\nAssistant: {response}")
]) | llm)

# ==================== ROUTES ====================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history
    data = request.json
    query = data.get("query", "")
    if data.get("new_chat"):
        chat_history = []
        return jsonify({"response": "New chat started! 🚀 What project should we build?"})

    chat_history.append(HumanMessage(content=query))
    full_msgs = [SystemMessage(content=system_prompt)] + chat_history

    result = agent_app.invoke({"messages": full_msgs})

    # Extract final response
    final_msg = result["messages"][-1]
    response_text = final_msg.content

    # If the model forgot to include the top 5, force-add it (safety net)
    if "Top 5 Matching Kaggle Datasets" not in response_text:
        for msg in reversed(result["messages"]):
            if isinstance(msg, ToolMessage) and "Top 5 Matching Kaggle Datasets" in msg.content:
                response_text = msg.content + "\n\n" + response_text
                break

    chat_history.append(AIMessage(content=response_text))

    followup = prediction_chain.invoke({
        "input": query,
        "response": response_text,
        "chat_history": chat_history
    }).content

    return jsonify({
        "response": f"{response_text}\n\n**Next Questions:**\n{followup}"
    })

if __name__ == "__main__":
    app.run(debug=True, port=5006)
