"""
app.py
------
Stock Analysis Chatbot — ask questions about any stock in plain English.

How it works:
  1. Fetch real stock data via yfinance (no API key needed)
  2. Convert each day's data into a text description
  3. Embed descriptions into FAISS vector index (RAG)
  4. User asks a question → retrieve relevant days → Groq LLM answers

Run with:  streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

from rag.fetcher   import fetch_stock_data
from rag.embedder  import build_index, search_index
from rag.answerer  import answer_question

load_dotenv()

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Chatbot", page_icon="📈", layout="wide")
st.title("📈 Stock Analysis Chatbot")
st.caption("Ask questions about any stock in plain English — powered by RAG + Groq")

# ── Session state ─────────────────────────────────────────────────────────────
if "index"        not in st.session_state: st.session_state.index    = None
if "chunks"       not in st.session_state: st.session_state.chunks   = []
if "df"           not in st.session_state: st.session_state.df       = None
if "ticker"       not in st.session_state: st.session_state.ticker   = ""
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# ── Sidebar: stock loader ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Load a Stock")
    ticker  = st.text_input("Ticker symbol", value="AAPL", placeholder="e.g. TSLA, MSFT")
    period  = st.selectbox("Time period", ["3mo", "6mo", "1y", "2y"], index=2)
    load_btn = st.button("Load Stock", use_container_width=True)

    if load_btn:
        if not ticker.strip():
            st.error("Please enter a ticker.")
        else:
            with st.spinner(f"Fetching {ticker.upper()} data…"):
                df, chunks = fetch_stock_data(ticker.upper(), period)

            if df is None:
                st.error("Could not fetch data. Check the ticker and try again.")
            else:
                with st.spinner("Building search index…"):
                    index = build_index(chunks)

                st.session_state.df       = df
                st.session_state.chunks   = chunks
                st.session_state.index    = index
                st.session_state.ticker   = ticker.upper()
                st.session_state.chat_history = []
                st.success(f"✅ Loaded {len(df)} trading days for {ticker.upper()}")

    if st.session_state.ticker:
        st.divider()
        st.markdown(f"**Loaded:** {st.session_state.ticker}")
        st.caption(f"{len(st.session_state.df)} trading days indexed")

    st.divider()
    st.markdown("**Try asking:**")
    examples = [
        "What was the highest closing price?",
        "How did the stock perform in the last month?",
        "When was the biggest single-day drop?",
        "What was the average volume?",
        "Was the overall trend bullish or bearish?",
        "Which month had the best performance?",
    ]
    for q in examples:
        if st.button(q, key=q, use_container_width=True):
            st.session_state._prefill = q

# ── Main area ─────────────────────────────────────────────────────────────────
if st.session_state.df is None:
    st.info("👈 Enter a ticker in the sidebar and click **Load Stock** to begin.")
    st.stop()

df     = st.session_state.df
ticker = st.session_state.ticker

# Price chart
col1, col2, col3, col4 = st.columns(4)
latest   = df["Close"].iloc[-1]
prev     = df["Close"].iloc[-2]
change   = latest - prev
pct      = (change / prev) * 100
high52   = df["Close"].max()
low52    = df["Close"].min()
avg_vol  = df["Volume"].mean()

col1.metric("Latest Close",  f"${latest:.2f}", f"{change:+.2f} ({pct:+.1f}%)")
col2.metric("52-Week High",  f"${high52:.2f}")
col3.metric("52-Week Low",   f"${low52:.2f}")
col4.metric("Avg Volume",    f"{avg_vol/1e6:.1f}M")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df.index, y=df["Close"],
    mode="lines", name="Close",
    line=dict(color="#185FA5", width=2),
    fill="tozeroy", fillcolor="rgba(24,95,165,0.08)"
))
fig.update_layout(
    title=f"{ticker} Price History",
    xaxis_title="Date", yaxis_title="Price (USD)",
    margin=dict(l=0, r=0, t=40, b=0),
    height=300,
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Chat interface ────────────────────────────────────────────────────────────
st.subheader("💬 Ask anything about this stock")

# Show chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Prefill from sidebar example buttons
prefill = st.session_state.pop("_prefill", "")
question = st.chat_input(
    placeholder="e.g. When was the biggest single-day drop?",
)
if prefill and not question:
    question = prefill

if question:
    if not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not found. Add it to your .env file.")
        st.stop()

    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching data and generating answer…"):
            answer, sources = answer_question(
                question,
                st.session_state.index,
                st.session_state.chunks,
                ticker,
            )
        st.markdown(answer)

        if sources:
            with st.expander("📎 Data points used"):
                for s in sources:
                    st.caption(s)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
