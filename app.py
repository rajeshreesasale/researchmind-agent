import streamlit as st
import time
from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain

st.set_page_config(
    page_title="ResearchMind · AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Injecting your custom CSS styles ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: #e8e4dc; }
.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(255,140,50,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(255,80,30,0.08) 0%, transparent 55%);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }
.hero { text-align: center; padding: 3.5rem 0 2.5rem; position: relative; }
.hero-eyebrow { font-family: 'DM Mono', monospace; font-size: 0.7rem; font-weight: 500; letter-spacing: 0.25em; text-transform: uppercase; color: #ff8c32; margin-bottom: 1rem; opacity: 0.9; }
.hero h1 { font-family: 'Syne', sans-serif; font-size: clamp(2.8rem, 6vw, 5rem); font-weight: 800; line-height: 1.0; letter-spacing: -0.03em; color: #f0ebe0; margin: 0 0 1rem; }
.hero h1 span { color: #ff8c32; }
.hero-sub { font-size: 1.05rem; font-weight: 300; color: #a09890; max-width: 520px; margin: 0 auto; line-height: 1.65; }
.divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(255,140,50,0.3), transparent); margin: 2rem 0; }
.input-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,140,50,0.15); border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 2rem; backdrop-filter: blur(8px); }
.stTextInput > div > div > input { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,140,50,0.25) !important; border-radius: 10px !important; color: #f0ebe0 !important; font-size: 1rem !important; padding: 0.75rem 1rem !important; }
.stTextInput > label { font-family: 'DM Mono', monospace !important; font-size: 0.72rem !important; letter-spacing: 0.15em !important; text-transform: uppercase !important; color: #ff8c32 !important; }
.stButton > button { background: linear-gradient(135deg, #ff8c32 0%, #ff5a1a 100%) !important; color: #0a0a0f !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; border-radius: 10px !important; width: 100%; box-shadow: 0 4px 20px rgba(255,140,50,0.3) !important; }
.step-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; padding: 1.5rem 1.8rem; margin-bottom: 1.2rem; position: relative; }
.step-card.active { border-color: rgba(255,140,50,0.4); background: rgba(255,140,50,0.04); }
.step-card.done { border-color: rgba(80,200,120,0.3); background: rgba(80,200,120,0.03); }
.step-header { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.3rem; }
.step-num { font-family: 'DM Mono', monospace; color: #ff8c32; font-size: 0.68rem; }
.step-title { font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 700; color: #f0ebe0; }
.step-status { margin-left: auto; font-family: 'DM Mono', monospace; font-size: 0.68rem; }
.status-waiting { color: #555; }
.status-running { color: #ff8c32; }
.status-done { color: #50c878; }
.report-panel { background: rgba(255,255,255,0.025); border: 1px solid rgba(255,140,50,0.2); border-radius: 16px; padding: 2rem 2.5rem; margin-top: 1rem; }
.feedback-panel { background: rgba(255,255,255,0.025); border: 1px solid rgba(80,200,120,0.2); border-radius: 16px; padding: 2rem 2.5rem; margin-top: 1rem; }
.panel-label { font-family: 'DM Mono', monospace; font-size: 0.7rem; text-transform: uppercase; margin-bottom: 1.2rem; }
.panel-label.orange { color: #ff8c32; border-bottom: 1px solid rgba(255,140,50,0.15); }
.panel-label.green { color: #50c878; border-bottom: 1px solid rgba(80,200,120,0.15); }
.section-heading { font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 700; color: #f0ebe0; margin: 2rem 0 1rem; }
.notice { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #605850; text-align: center; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)

def step_card(num: str, title: str, state: str, desc: str = ""):
    status_map = {
        "waiting": ("WAITING", "status-waiting"),
        "running": ("● RUNNING", "status-running"),
        "done":     ("✓ DONE",   "status-done"),
    }
    label, cls = status_map.get(state, ("", ""))
    card_cls = {"running": "active", "done": "done"}.get(state, "")
    st.markdown(f"""
    <div class="step-card {card_cls}">
        <div class="step-header">
            <span class="step-num">{num}</span>
            <span class="step-title">{title}</span>
            <span class="step-status {cls}">{label}</span>
        </div>
        {"<div style='font-size:0.82rem;color:#706860;margin-top:0.3rem;'>"+desc+"</div>" if desc else ""}
    </div>
    """, unsafe_allow_html=True)

if "results" not in st.session_state: st.session_state.results = {}
if "running" not in st.session_state: st.session_state.running = False
if "done" not in st.session_state: st.session_state.done = False

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Multi-Agent AI System</div>
    <h1>Research<span>Mind</span></h1>
    <p class="hero-sub">Four specialized AI agents collaborate — searching, scraping, writing, and critiquing.</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

col_input, col_spacer, col_pipeline = st.columns([5, 0.5, 4])

with col_input:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    topic = st.text_input("Research Topic", placeholder="e.g. AI updates in 2026", key="topic_input")
    run_btn = st.button("⚡   Run Research Pipeline", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_pipeline:
    st.markdown('<div class="section-heading">Pipeline Status</div>', unsafe_allow_html=True)
    r = st.session_state.results
    
    def s(step):
        if not st.session_state.running and not r: return "waiting"
        if step in r: return "done"
        steps = ["search", "reader", "writer", "critic"]
        if st.session_state.running:
            for k in steps:
                if k not in r: return "running" if k == step else "waiting"
        return "waiting"

    step_card("01", "Search Agent",  s("search"), "Gathers recent web information")
    step_card("02", "Reader Agent",  s("reader"), "Scrapes & extracts deep content")
    step_card("03", "Writer Chain",  s("writer"), "Drafts the full research report")
    step_card("04", "Critic Chain",  s("critic"), "Reviews & scores the report")

# ── Dynamic Chain Trigger ──
if run_btn:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.results = {}
        st.session_state.running = True
        st.session_state.done = False
        st.session_state.current_topic = topic
        st.rerun()

if st.session_state.running and not st.session_state.done:
    topic_val = st.session_state.get('current_topic', '')

    # ── Step 1: Search Agent ──
    if "search" not in st.session_state.results:
        with st.spinner("🔍  Search Agent is collecting web records..."):
            search_agent = build_search_agent()
            sr = search_agent.invoke({"messages": f"Find recent, reliable and detailed information about: {topic_val}"})
            
            # Safe Fallback Check
            if "output" in sr:
                search_text = sr["output"]
            elif "text" in sr:
                search_text = sr["text"]
            elif "messages" in sr and len(sr["messages"]) > 0:
                search_text = getattr(sr["messages"][-1], "content", str(sr["messages"][-1]))
            else:
                search_text = str(sr)
                
            st.session_state.results["search"] = search_text
            st.rerun()

    # ── Step 2: Reader Agent ──
    if "reader" not in st.session_state.results:
        with st.spinner("📄  Reader Agent is extracting deep resource texts..."):
            search_ctx = st.session_state.results["search"]
            reader_agent = build_reader_agent()
            rr = reader_agent.invoke({"messages": f"Based on these results about '{topic_val}', scrape the best link:\n\n{search_ctx[:800]}"})
            
            # Safe Fallback Check
            if "output" in rr:
                reader_text = rr["output"]
            elif "text" in rr:
                reader_text = rr["text"]
            elif "messages" in rr and len(rr["messages"]) > 0:
                reader_text = getattr(rr["messages"][-1], "content", str(rr["messages"][-1]))
            else:
                reader_text = str(rr)
                
            st.session_state.results["reader"] = reader_text
            st.rerun()

    # ── Step 3: Writer Chain ──
    if "writer" not in st.session_state.results:
        with st.spinner("✍️  Writer Chain is compiling report draft..."):
            combined = f"SEARCH:\n{st.session_state.results['search']}\n\nSCRAPED:\n{st.session_state.results['reader']}"
            report_out = writer_chain.invoke({"topic": topic_val, "research": combined})
            st.session_state.results["writer"] = report_out
            st.rerun()

    # ── Step 4: Critic Chain ──
    if "critic" not in st.session_state.results:
        with st.spinner("🧐  Critic Chain is scoring content..."):
            critic_out = critic_chain.invoke({"report": st.session_state.results["writer"]})
            st.session_state.results["critic"] = critic_out

    st.session_state.running = False
    st.session_state.done = True
    st.rerun()

# ── Output Panel Rendering ──
r = st.session_state.results
if r and st.session_state.done:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Generated Deliverables</div>', unsafe_allow_html=True)

    if "writer" in r:
        st.markdown('<div class="report-panel"><div class="panel-label orange">📝 Final Research Report</div>', unsafe_allow_html=True)
        st.markdown(r["writer"])
        st.markdown('</div>', unsafe_allow_html=True)
        st.download_button(label="⬇   Download Report (.md)", data=r["writer"], file_name="research_report.md", mime="text/markdown")
        
    if "critic" in r:
        st.markdown('<div class="feedback-panel"><div class="panel-label green">🧐 Critic Evaluation Verdict</div>', unsafe_allow_html=True)
        st.markdown(r["critic"])
        st.markdown('</div>', unsafe_allow_html=True)