from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# FIX: Swapped out decommissioned specdec model for the current production versatile model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ── 1st Agent: Search ──
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="You are a helpful research assistant. You MUST use the web_search tool to find accurate, recent information."
    )

# ── 2nd Agent: Reader ──
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="You are a deep-reading assistant. You MUST use the scrape_url tool to gather deep body text from URLs."
    )

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