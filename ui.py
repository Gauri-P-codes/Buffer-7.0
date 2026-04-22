"""
ui.py
-----
Streamlit Interactive Dashboard for the Crowd & Service Flow Optimizer.

Layout:
  ┌─────────────────────────────────────────┐
  │          Header + System Status          │
  ├──────────────┬──────────────────────────┤
  │  Controls    │   Queue Visualizations   │
  │  (buttons)   │   (per-counter cards)    │
  ├──────────────┴──────────────────────────┤
  │        Graph: Queue sizes over time      │
  ├─────────────────────────────────────────┤
  │              Event Log                   │
  └─────────────────────────────────────────┘
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time

from queue_manager import QueueManager,Customer
from scheduler import Scheduler
from simulation import (
    simulate_single_arrival,
    run_auto_simulation_step,
    seed_initial_customers,
)

def hex_to_rgba(hex_color, alpha=0.1):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"

# ──────────────────────────────────────────────
# Page config — must be first Streamlit call
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Queue Optimizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# Custom CSS — clean dark dashboard aesthetic
# ──────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
  }

  /* ── Background ── */
  .stApp {
    background: linear-gradient(135deg, #0d0f1a 0%, #111827 50%, #0d1117 100%);
    color: #e2e8f0;
  }

  /* ── Header ── */
  .header-box {
    background: linear-gradient(90deg, #1e3a5f 0%, #0f2744 50%, #1a1f3a 100%);
    border: 1px solid #2d4a6b;
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 20px;
    box-shadow: 0 4px 24px rgba(59,130,246,0.15);
  }
  .header-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
  }
  .header-subtitle {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
  }

  /* ── Metric cards ── */
  .metric-card {
    background: #1e2535;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: #60a5fa; }
  .metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #60a5fa;
    font-family: 'JetBrains Mono', monospace;
  }
  .metric-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
  }

  /* ── Counter cards ── */
  .counter-card {
    background: #1a2235;
    border: 1px solid #2d3748;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
    transition: all 0.2s;
  }
  .counter-card:hover { border-color: #3b82f6; transform: translateY(-1px); }
  .counter-title {
    font-size: 1rem;
    font-weight: 600;
    color: #93c5fd;
    margin-bottom: 8px;
  }
  .counter-stat {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #cbd5e1;
  }
  .vip-badge {
    background: linear-gradient(90deg, #f59e0b, #d97706);
    color: #1a1a1a;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 20px;
    margin-left: 6px;
  }
  .norm-badge {
    background: #334155;
    color: #94a3b8;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-left: 4px;
  }

  /* ── Queue bar ── */
  .queue-bar-wrap {
    background: #0f172a;
    border-radius: 8px;
    height: 10px;
    width: 100%;
    margin: 8px 0;
    overflow: hidden;
  }
  .queue-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.4s ease;
  }

  /* ── Status badges ── */
  .badge-normal  { background:#1e3a5f; color:#60a5fa; border:1px solid #3b82f6; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600; }
  .badge-sjf     { background:#2d1b5e; color:#a78bfa; border:1px solid #7c3aed; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600; }
  .badge-peak    { background:#4c1d1d; color:#f87171; border:1px solid #ef4444; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600; animation: pulse 1s infinite; }

  @keyframes pulse {
    0%,100% { opacity:1; }
    50%      { opacity:0.6; }
  }

  /* ── Log ── */
  .log-box {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 16px;
    height: 220px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #7dd3fc;
  }
  .log-entry { padding: 2px 0; border-bottom: 1px solid #1a2235; }
            
  /* ── Toggle / Checkbox labels ── */
  .stToggle label, 
  [data-testid="stToggle"] label,
  div[data-testid="stToggle"] p,
  .stCheckbox label,
  [data-testid="stCheckbox"] label,
  [data-testid="stToggle"] span {
    color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
  }
 /* ── Toggle label fix ── */
  label, label p, label span {
    color: #93c5fd !important;
  }
  /* ── Buttons ── */
  .stButton > button {
    background: #1a2235 !important;
    color: #93c5fd !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: all 0.15s !important;
    border: 1px solid #2d3748 !important;
    width: 100% !important;
  }
  .stButton > button:hover {
    background: #1e2d45 !important;
    color: #60a5fa !important;
    border-color: #3b82f6 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2) !important;
  }
  .stButton > button:active {
    transform: translateY(0px) !important;
    box-shadow: none !important;
  }
  /* ── Section title ── */
  .section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #cbd5e1;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e2d3d;
  }

  div[data-testid="stMetric"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State — initialise once per session
# ──────────────────────────────────────────────
def init_state():
    if "manager" not in st.session_state:
        num_counters = st.session_state.get("num_counters_cfg", 4)
        manager   = QueueManager(num_counters=num_counters, avg_service_time=3)
        scheduler = Scheduler(manager)
        seed_initial_customers(scheduler, num_customers=6, vip_probability=0.25)
        manager.record_snapshot()
        st.session_state.manager   = manager
        st.session_state.scheduler = scheduler
        st.session_state.last_served = None
        st.session_state.auto_run    = False
        st.session_state.step_count  = 0

init_state()

manager: QueueManager = st.session_state.manager
scheduler: Scheduler  = st.session_state.scheduler


# ──────────────────────────────────────────────
# Helper: colour for queue fill bar
# ──────────────────────────────────────────────
def queue_color(length: int, max_len: int = 10) -> str:
    ratio = min(length / max(max_len, 1), 1.0)
    if ratio < 0.4:
        return "#22c55e"   # green — light load
    elif ratio < 0.7:
        return "#f59e0b"   # amber — moderate
    else:
        return "#ef4444"   # red   — heavy


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
summary = scheduler.status_summary()

mode_badge = (
    '<span class="badge-sjf">⚡ FCFS MODE</span>' if summary["mode"] == "FCFS"
    else '<span class="badge-normal">🔵 NORMAL MODE</span>'
)
peak_badge = '<span class="badge-peak">🚨 PEAK LOAD</span>' if summary["is_peak"] else ""

st.markdown(f"""
<div class="header-box">
  <p class="header-title">🎯 AI-Inspired Queue & Service Flow Optimizer</p>
  <p class="header-subtitle">
    DSA-powered • Multi-Counter • Priority Queues • Real-time Rebalancing
    &nbsp;&nbsp;|&nbsp;&nbsp;
    {mode_badge} {peak_badge}
  </p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# TOP METRICS ROW
# ──────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)

total_served = sum(c.served_count for c in manager.counters)
max_wait     = max(summary["wait_times"]) if summary["wait_times"] else 0
avg_wait     = (sum(summary["wait_times"]) / len(summary["wait_times"])
                if summary["wait_times"] else 0)

for col, val, label in [
    (m1, summary["total_customers"], "Customers Waiting"),
    (m2, total_served,               "Total Served"),
    (m3, len(manager.counters),      "Active Counters"),
    (m4, f"{avg_wait:.1f}s",         "Avg Wait Time"),
    (m5, f"{max_wait:.1f}s",         "Max Wait Time"),
]:
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-value">{val}</div>
      <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MAIN LAYOUT: Controls  |  Queues
# ──────────────────────────────────────────────
ctrl_col, queue_col = st.columns([1, 2], gap="large")

# ── LEFT: Controls ────────────────────────────
with ctrl_col:
    st.markdown('<div class="section-title">🕹️ Controls</div>', unsafe_allow_html=True)

    # Add Normal Customer
    if st.button("👤 Add Normal Customer", use_container_width=True):
        cust = simulate_single_arrival(scheduler, is_vip=False)
        st.session_state.last_served = None
        manager.record_snapshot()

    # Add VIP Customer
    if st.button("⭐ Add VIP Customer", use_container_width=True):
        cust = simulate_single_arrival(scheduler, is_vip=True)
        st.session_state.last_served = None
        manager.record_snapshot()

    # Serve Next
    if st.button("✅ Serve Next Customer", use_container_width=True):
        customer, counter_id = scheduler.serve_next_global()
        if customer:
            st.session_state.last_served = (customer, counter_id)
        else:
            manager.log("⚠️ No customers to serve!")
            st.session_state.last_served = None
        manager.record_snapshot()

    # Rebalance
    if st.button("🔄 Trigger Rebalancing", use_container_width=True):
        moved = scheduler.rebalance()
        manager.record_snapshot()

    # Emergency
    if st.button("🚨 Emergency Injection", use_container_width=True):
        cid = scheduler.inject_emergency()
        manager.record_snapshot()

    # Burst arrival
    if st.button("💥 Burst Arrival (+5)", use_container_width=True):
        from simulation import simulate_arrival_burst
        simulate_arrival_burst(scheduler, count=5, vip_probability=0.2)
        manager.record_snapshot()

    st.markdown("---")

    # Auto-simulate toggle
    st.markdown('<div class="section-title">🤖 Auto Simulation</div>', unsafe_allow_html=True)
    auto = st.toggle("Enable Auto-Step", value=st.session_state.auto_run)
    st.session_state.auto_run = auto

    speed = st.slider("Step Interval (s)", 0.5, 3.0, 1.5, 0.5)

    if st.button("▶ Run One Step", use_container_width=True):
        run_auto_simulation_step(scheduler)
        st.session_state.step_count += 1

    # Reset system
    st.markdown("---")
    num_c = st.selectbox("Number of Counters", [3, 4, 5], index=1)
    if st.button("🔁 Reset System", use_container_width=True):
        Customer._id_counter = 0
        del st.session_state["manager"]
        del st.session_state["scheduler"]
        st.session_state.num_counters_cfg = num_c
        st.rerun()

    # Last served info
    if st.session_state.last_served:
        customer, counter_id = st.session_state.last_served
        tag = "⭐ VIP" if customer.is_vip else "👤 Normal"
        st.success(f"Just served: {tag} **{customer.name}** from Counter-{counter_id}")


# ── RIGHT: Queue Visualisation ─────────────────
with queue_col:
    st.markdown('<div class="section-title">📊 Live Queue Status</div>',
                unsafe_allow_html=True)

    max_q = max(summary["queue_lengths"]) if summary["queue_lengths"] else 1

    for i, counter in enumerate(manager.counters):
        length     = counter.total_length()
        wait_est   = counter.estimated_wait_time()
        pred_time  = scheduler.predict_service_time(counter)
        all_custs  = counter.all_customers()
        fill_pct   = min(int(length / max(max_q, 8) * 100), 100)
        bar_color  = queue_color(length)

        # Customer pills
        pills_html = ""
        for c in all_custs[:10]:    # show max 10 to avoid overflow
            if c.is_vip:
                pills_html += f'<span class="vip-badge">⭐{c.name}</span> '
            else:
                pills_html += f'<span class="norm-badge">{c.name}</span> '
        if len(all_custs) > 10:
            pills_html += f'<span class="norm-badge">+{len(all_custs)-10} more</span>'
        if not all_custs:
            pills_html = '<span style="color:#475569;font-size:0.8rem;">— empty —</span>'

        st.markdown(f"""
        <div class="counter-card">
          <div class="counter-title">
            🏢 Counter {counter.counter_id}
            <span style="font-size:0.75rem;color:#475569;margin-left:8px;">
              served: {counter.served_count}
            </span>
          </div>
          <div class="counter-stat">
            Queue: <b style="color:#f1f5f9">{length}</b> &nbsp;|&nbsp;
            Est. Wait: <b style="color:#fbbf24">{wait_est:.1f}s</b> &nbsp;|&nbsp;
            Predicted: <b style="color:#34d399">{pred_time:.1f}s/cust</b>
          </div>
          <div class="queue-bar-wrap">
            <div class="queue-bar-fill" style="width:{fill_pct}%;background:{bar_color};"></div>
          </div>
          <div style="margin-top:8px;">{pills_html}</div>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# CHARTS ROW
# ──────────────────────────────────────────────
st.markdown('<div class="section-title">📈 Queue History & Analytics</div>',
            unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2, gap="medium")

# ── Chart 1: Queue size over time (line chart) ──
# ── Chart 1: Queue size over time (line chart) ──
with chart_col1:
    history = manager.history
    if len(history) >= 2:
        df = pd.DataFrame(
            history,
            columns=[f"Counter {i+1}" for i in range(len(manager.counters))]
        )
        df.index.name = "Tick"
        df = df.reset_index()

        fig = go.Figure()
        colors = ["#60a5fa", "#a78bfa", "#34d399", "#fbbf24", "#f87171"]

        for idx, col in enumerate([c for c in df.columns if c != "Tick"]):
            color = colors[idx % len(colors)]

            fig.add_trace(go.Scatter(
                x=df["Tick"], y=df[col],
                name=col,
                line=dict(color=color, width=2.5),
                fill="tozeroy",
                fillcolor=hex_to_rgba(color, 0.05),  # ✅ FIXED
                mode="lines+markers",
                marker=dict(size=4),
            ))

        fig.update_layout(
            title="Queue Lengths Over Time",
            plot_bgcolor="#0d1117",
            paper_bgcolor="#1a2235",
            font=dict(color="#94a3b8", family="Space Grotesk"),
            legend=dict(bgcolor="#1a2235", bordercolor="#2d3748"),
            xaxis=dict(gridcolor="#1e2d3d", title="Simulation Tick"),
            yaxis=dict(gridcolor="#1e2d3d", title="Queue Length"),
            height=300,
            margin=dict(t=40, b=20, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Interact with the system to generate history data.")
# ── Chart 2: Current queue comparison (bar chart) ──
with chart_col2:
    lengths  = summary["queue_lengths"]
    waits    = summary["wait_times"]
    labels   = [f"Counter {i+1}" for i in range(len(lengths))]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=labels, y=lengths,
        name="Queue Length",
        marker_color=["#60a5fa", "#a78bfa", "#34d399", "#fbbf24"][:len(lengths)],
        text=lengths,
        textposition="auto",
    ))
    fig2.add_trace(go.Scatter(
        x=labels, y=waits,
        name="Est. Wait (s)",
        yaxis="y2",
        line=dict(color="#f87171", width=2, dash="dot"),
        mode="lines+markers",
        marker=dict(size=7, color="#f87171"),
    ))

    fig2.update_layout(
        title="Current Queue Status",
        plot_bgcolor="#0d1117",
        paper_bgcolor="#1a2235",
        font=dict(color="#94a3b8", family="Space Grotesk"),
        legend=dict(bgcolor="#1a2235", bordercolor="#2d3748"),
        xaxis=dict(gridcolor="#1e2d3d"),
        yaxis=dict(gridcolor="#1e2d3d", title="Queue Length"),
        yaxis2=dict(title="Wait Time (s)", overlaying="y", side="right", gridcolor="#1e2d3d"),
        height=300,
        margin=dict(t=40, b=20, l=10, r=10),
        barmode="group",
    )
    st.plotly_chart(fig2, use_container_width=True)


# ──────────────────────────────────────────────
# EVENT LOG
# ──────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Event Log</div>', unsafe_allow_html=True)

log_html = "".join(
    f'<div class="log-entry">{entry}</div>'
    for entry in reversed(manager.event_log[-30:])
)
st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DSA INFO EXPANDER
# ──────────────────────────────────────────────
with st.expander("📚 DSA Concepts Used in This System"):
    st.markdown("""
    | Feature | Data Structure / Algorithm | Complexity |
    |---|---|---|
    | Normal Queue | `collections.deque` | O(1) enqueue / dequeue |
    | VIP Queue | `heapq` (min-heap) | O(log n) push / pop |
    | Smart Assignment | Greedy (min queue scan) | O(k) counters |
    | Queue Rebalancing | Linear scan + deque ops | O(k × moves) |
    | Global FCFS Serving | heapq (arrival time across counters) | O(k log k) |
    | Peak Detection | Sliding Window | O(w) window size |
    | Service Time Prediction | Weighted moving average | O(h) history |
    | Emergency Injection | heapq push | O(log n) |
    """)

# ──────────────────────────────────────────────
# AUTO-SIMULATION (runs if toggle is ON)
# ──────────────────────────────────────────────
if st.session_state.auto_run:
    time.sleep(speed)
    run_auto_simulation_step(scheduler)
    st.session_state.step_count += 1
    st.rerun()