import os
import operator
from typing import TypedDict, List, Annotated

import pandas as pd
import streamlit as st

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_data():
    """
    Expects these files in the repo (same folder as app.py):
      - BasePlans.csv
      - Riders.csv
      - Junction.csv
      - PolicyHolders.csv
      - PolicyRiders.csv
    """
    base_plans = pd.read_csv("BasePlans.csv").set_index("PolicyID")
    riders = pd.read_csv("Riders.csv").set_index("RiderID")
    junction = pd.read_csv("Junction.csv")
    policy_holders = pd.read_csv("PolicyHolders.csv").set_index("PolicyHolderID")
    policy_riders = pd.read_csv("PolicyRiders.csv").set_index("PolicyHolderID")

    return base_plans, riders, junction, policy_holders, policy_riders


def build_domain_functions():
    base_plans, riders, junction, policy_holders, policy_riders = load_data()

    def get_available_riders(base_policy_id: int) -> str:
        riders_for_base = junction[junction["PolicyID"] == base_policy_id]["RiderID"].tolist()
        rider_details = riders.loc[riders_for_base]

        rider_summary = "\n".join(
            f"- {row['RiderName']}: {row['Description']}"
            for _, row in rider_details.iterrows()
        )

        return f"Policy {base_plans.loc[base_policy_id].PolicyName} Riders:\n{rider_summary}"

    def get_available_policies_for_user(user_id: int) -> str:
        policy_holder_name = policy_holders.loc[user_id].Name
        rider_summary = get_available_riders(policy_holders.loc[user_id].BasePolicyID)
        return f"{policy_holder_name}:\n{rider_summary}"

    def check_current_coverage(user_id: int) -> List[str]:
        # policy_riders.loc[user_id] may return Series or DataFrame depending on number of riders
        pr = policy_riders.loc[user_id]
        if isinstance(pr, pd.Series):
            rider_ids = [pr["RiderID"]]
        else:
            rider_ids = pr["RiderID"].tolist()

        rider_names = riders.loc[rider_ids]["RiderName"]
        if isinstance(rider_names, pd.Series):
            return rider_names.tolist()
        return list(rider_names)

    def estimate_new_premium(user_id: int, additional_riders: List[str]) -> float:
        user_base_plan = policy_holders.loc[user_id]["BasePolicyID"]
        current_riders = check_current_coverage(user_id)
        base_premium = float(base_plans.loc[user_base_plan, "BasePremium"])

        current_rider_costs = float(riders.loc[riders["RiderName"].isin(current_riders), "PricePerYear"].sum())
        additional_rider_costs = float(riders.loc[riders["RiderName"].isin(additional_riders), "PricePerYear"].sum())
        total_rider_cost = current_rider_costs + additional_rider_costs

        total_riders = len(set(current_riders + additional_riders))
        discount = 0.0
        if total_riders == 2:
            discount = 0.15
        elif total_riders >= 3:
            discount = 0.25

        discounted_rider_cost = total_rider_cost * (1 - discount)
        return base_premium + discounted_rider_cost

    return get_available_policies_for_user, check_current_coverage, estimate_new_premium


# -----------------------------
# Tools (LangChain tool wrappers)
# -----------------------------
@st.cache_resource
def build_tools_and_graph():
    get_available_policies_for_user, check_current_coverage, estimate_new_premium = build_domain_functions()

    @tool
    def get_available_policies_for_user_tool(user_id: int) -> str:
        """Lookup a policyholder's base policy and all available riders by policyholder ID."""
        return get_available_policies_for_user(user_id)

    @tool
    def check_current_coverage_tool(user_id: int) -> List[str]:
        """Lookup current rider coverage for a specific policyholder ID."""
        return check_current_coverage(user_id)

    @tool
    def estimate_new_premium_tool(user_id: int, additional_riders: List[str]) -> float:
        """Estimate a new premium given a policyholder ID and a list of additional rider names."""
        return estimate_new_premium(user_id, additional_riders)

    tools = [get_available_policies_for_user_tool, check_current_coverage_tool, estimate_new_premium_tool]

    system = SystemMessage(content="""
You are AdvisorBot, an intelligent and empathetic assistant that helps insurance policyholders
understand their coverage and explore options for additional protection.

Goals:
- Clarify what the user is trying to do (understand current coverage vs. explore new coverage)
- Ask for a PolicyHolderID when needed
- Use tools to retrieve coverage and calculate premiums
- Be clear about what is known from the data vs. what is general guidance

Tool guidance:
- Use check_current_coverage_tool to see what riders a policyholder already has.
- Use get_available_policies_for_user_tool to list riders available for their base policy.
- Use estimate_new_premium_tool to compute a premium estimate for additional riders.

When asking for additional riders, request rider *names* exactly as listed.
""".strip())

    api_key = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Add it to Streamlit secrets or environment variables.")

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=api_key).bind_tools(tools)

    class State(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]

    def assistant_node(state: State):
        ai_msg = llm.invoke(state["messages"])
        return {"messages": [ai_msg]}

    tools_node = ToolNode(tools)

    def route(state: State) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "end"

    graph = StateGraph(State)
    graph.add_node("assistant", assistant_node)
    graph.add_node("tools", tools_node)

    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", route, {"tools": "tools", "end": END})
    graph.add_edge("tools", "assistant")

    app = graph.compile()
    return app, system


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AdvisorBot (LangGraph)", layout="centered")
st.title("AdvisorBot")
st.caption("LangGraph + tools + CSV-backed insurance data (demo)")

# Try to build graph; fail gracefully if missing key or CSVs
try:
    graph_app, SYSTEM = build_tools_and_graph()
except Exception as e:
    st.error(f"App setup failed: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [SYSTEM]

# Render chat history
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        continue
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

prompt = st.chat_input("Ask about coverage, riders, or premium estimates…")
if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))

    # Run the LangGraph loop using the accumulated messages
    result = graph_app.invoke({"messages": st.session_state.messages})
    new_messages = result["messages"]

    # Append only the newly produced messages (AI + tool messages)
    for m in new_messages:
        # LangGraph returns full list; avoid duplicating SYSTEM and history
        # We'll append only messages that are not already in session_state by identity check.
        st.session_state.messages.append(m)

    st.rerun()