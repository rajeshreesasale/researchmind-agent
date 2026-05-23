import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Updated to the currently active production 8B model: llama-3.1-8b-instant
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ── Custom Lightweight Agent Runner (Version-Agnostic) ──
class SimpleAgentRunner:
    """
    A simple, reliable execution engine that invokes tools manually.
    Bypasses all shifting LangChain version import bugs entirely.
    """
    def __init__(self, llm_model, tools_list):
        self.llm_with_tools = llm_model.bind_tools(tools_list)
        self.tools_map = {tool.name: tool for tool in tools_list}

    def invoke(self, inputs):
        user_msg = inputs.get("messages", "")
        
        # Let the model decide which tool and parameters to use
        ai_msg = self.llm_with_tools.invoke(user_msg)
        
        # Check if the model requested any tool calls
        if ai_msg.tool_calls:
            tool_call = ai_msg.tool_calls[0] 
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # Execute the tool safely matching its map key
            if tool_name in self.tools_map:
                tool_output = self.tools_map[tool_name].invoke(tool_args)
                
                # Feed the tool's raw result back to the model for final answer formatting
                final_prompt = f"""You are a helpful research assistant. 
Here is the user query: {user_msg}

The tool '{tool_name}' returned the following information:
{tool_output}

Synthesize this data perfectly to answer the user query."""
                
                final_response = llm.invoke(final_prompt)
                return {"output": final_response.content}
                
        return {"output": ai_msg.content}


# ── 1st Agent: Search ──
def build_search_agent():
    return SimpleAgentRunner(llm, [web_search])

# ── 2nd Agent: Reader ──
def build_reader_agent():
    return SimpleAgentRunner(llm, [scrape_url])

# ── 3rd Agent: Writer Chain ──
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured, and insightful reports based exactly on the research provided."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual, and professional."""),
])
writer_chain = writer_prompt | llm | StrOutputParser()

# ── 4th Agent: Critic Chain ──
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...

Areas to Improve:
- ...

One line verdict:
..."""),
])
critic_chain = critic_prompt | llm | StrOutputParser()
