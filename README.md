# 📈 Stock Analysis Chatbot

> Ask questions about any stock in plain English — powered by RAG + Groq AI

A conversational chatbot that fetches real stock data and lets you ask natural language questions about it, like "When was the biggest single-day drop?" or "How did AAPL perform last quarter?"

---

## 🎯 What it does

- **Fetches real stock data** for any ticker using `yfinance` (no API key needed)
- **Indexes the data** using FAISS vector search (RAG pipeline)
- **Answers your questions** using Groq's free LLM API (llama3-70b)
- **Shows an interactive price chart** with key metrics

---

## 🏗 How it works

```
User enters ticker (e.g. AAPL)
        ↓
yfinance fetches 1 year of OHLCV data
        ↓
Each trading day → plain English description
  "On June 3 2024, AAPL opened at $192, closed at $194..."
        ↓
sentence-transformers embeds all descriptions → FAISS index
        ↓
User asks a question
        ↓
FAISS finds the most relevant trading days
        ↓
Groq LLM (llama3-70b) answers based on retrieved data
```

---

## 🚀 Setup

### 1. Clone and install
```bash
git clone https://github.com/SaiManikanth27/financial-rag-tool.git
cd financial-rag-tool
pip install -r requirements.txt
```

### 2. Get a free Groq API key
Sign up at [console.groq.com](https://console.groq.com) — free, no credit card.

```bash
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY
```

### 3. Run
```bash
streamlit run app.py
```

A browser window opens at `http://localhost:8501`

---

## 📂 Project structure

```
stock-chatbot/
├── app.py              # Streamlit UI
├── rag/
│   ├── fetcher.py      # yfinance data fetching + text conversion
│   ├── embedder.py     # sentence-transformers + FAISS index
│   └── answerer.py     # RAG query + Groq LLM answer generation
├── requirements.txt
├── .env.example
└── README.md
```

---

## 💬 Example questions

| Question | What it demonstrates |
|---|---|
| "What was the highest closing price?" | Retrieval of peak price data |
| "When was the biggest single-day drop?" | Detecting negative events |
| "How did the stock perform in January?" | Time-range analysis |
| "Was the overall trend bullish or bearish?" | Trend summarisation |
| "What was the average daily volume?" | Aggregation queries |

---

## 🛠 Tech stack

| Component | Tool | Why |
|---|---|---|
| Stock data | `yfinance` | Free, real data, no API key |
| Embeddings | `sentence-transformers` | Free, local, fast |
| Vector search | `FAISS` | No server, lightweight |
| LLM | `Groq (llama3-70b)` | Free tier, fast responses |
| UI | `Streamlit` | Clean, easy to demo |

---

## 📄 License

MIT
