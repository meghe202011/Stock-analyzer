"""
AI Stock Analyzer
-----------------
Requirements:
    pip install yfinance anthropic pandas ta streamlit plotly requests

Usage:
    streamlit run stock_analyzer.py

Set your Anthropic API key as an environment variable:
    export ANTHROPIC_API_KEY="your-key-here"
Or enter it in the sidebar when the app launches.
"""

import os
import json
import anthropic
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Try importing the 'ta' library ──────────────────────────────────────────
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .main { background: #0d0f14; color: #e2e8f0; }

  h1, h2, h3 { font-family: 'Space Mono', monospace; }

  .metric-card {
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
  }
  .metric-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; }
  .metric-value { font-size: 1.6rem; font-weight: 600; font-family: 'Space Mono', monospace; margin-top: 0.2rem; }
  .positive { color: #22c55e; }
  .negative { color: #ef4444; }
  .neutral  { color: #94a3b8; }

  .ai-box {
    background: linear-gradient(135deg, #1a1d27 0%, #0f1629 100%);
    border: 1px solid #3b4fd8;
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1rem;
    line-height: 1.8;
  }
  .ai-box h3 { color: #818cf8; margin-top: 1.2rem; }
  .ai-box strong { color: #c7d2fe; }

  .badge {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
  }
  .badge-buy  { background: #14532d; color: #4ade80; }
  .badge-sell { background: #450a0a; color: #f87171; }
  .badge-hold { background: #1c1917; color: #fbbf24; }

  div[data-testid="stSidebar"] {
    background: #0d0f14;
    border-right: 1px solid #1e2030;
  }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# DATA FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, period: str = "1y"):
    """Download OHLCV data and company info from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    df    = stock.history(period=period)
    info  = stock.info
    return df, info


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, Bollinger Bands, and moving averages."""
    df = df.copy()
    close = df["Close"]

    if TA_AVAILABLE:
        df["RSI"]      = ta.momentum.RSIIndicator(close, window=14).rsi()
        macd_obj       = ta.trend.MACD(close)
        df["MACD"]     = macd_obj.macd()
        df["MACD_sig"] = macd_obj.macd_signal()
        bb             = ta.volatility.BollingerBands(close)
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()
        df["BB_mid"]   = bb.bollinger_mavg()
    else:
        # Manual fallback
        delta     = close.diff()
        gain      = delta.clip(lower=0).rolling(14).mean()
        loss      = (-delta.clip(upper=0)).rolling(14).mean()
        rs        = gain / loss.replace(0, float("nan"))
        df["RSI"] = 100 - (100 / (1 + rs))

        ema12          = close.ewm(span=12, adjust=False).mean()
        ema26          = close.ewm(span=26, adjust=False).mean()
        df["MACD"]     = ema12 - ema26
        df["MACD_sig"] = df["MACD"].ewm(span=9, adjust=False).mean()

        sma20          = close.rolling(20).mean()
        std20          = close.rolling(20).std()
        df["BB_upper"] = sma20 + 2 * std20
        df["BB_lower"] = sma20 - 2 * std20
        df["BB_mid"]   = sma20

    df["SMA_50"]  = close.rolling(50).mean()
    df["SMA_200"] = close.rolling(200).mean()
    return df


# ════════════════════════════════════════════════════════════════════════════
# CHART
# ════════════════════════════════════════════════════════════════════════════

def build_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.04,
    )

    # ── Candlestick + Bollinger Bands ────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        name="Price",
    ), row=1, col=1)

    if "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_upper"], line=dict(color="#6366f1", width=1, dash="dot"),
            name="BB Upper", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_lower"], line=dict(color="#6366f1", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(99,102,241,0.07)",
            name="BB Lower", showlegend=False,
        ), row=1, col=1)

    if "SMA_50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_50"],
            line=dict(color="#f59e0b", width=1.5), name="SMA 50",
        ), row=1, col=1)
    if "SMA_200" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_200"],
            line=dict(color="#ec4899", width=1.5), name="SMA 200",
        ), row=1, col=1)

    # ── Volume ────────────────────────────────────────────────────────────
    colors = ["#22c55e" if c >= o else "#ef4444"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], marker_color=colors,
        name="Volume", showlegend=False,
    ), row=2, col=1)

    # ── RSI ───────────────────────────────────────────────────────────────
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"],
            line=dict(color="#818cf8", width=2), name="RSI",
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#0d0f14",
        font=dict(family="DM Sans", color="#94a3b8"),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text=f"{ticker} — Technical Chart", font=dict(family="Space Mono", size=14)),
        height=620,
    )
    fig.update_yaxes(gridcolor="#1e2030", zerolinecolor="#1e2030")
    fig.update_xaxes(gridcolor="#1e2030", zerolinecolor="#1e2030")
    return fig


# ════════════════════════════════════════════════════════════════════════════
# AI ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def build_summary(ticker: str, df: pd.DataFrame, info: dict) -> dict:
    """Collect key metrics into a clean dict for the AI prompt."""
    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest

    change_pct = ((latest["Close"] - prev["Close"]) / prev["Close"] * 100
                  if prev["Close"] else 0)

    return {
        "ticker":        ticker,
        "company":       info.get("longName", ticker),
        "sector":        info.get("sector", "N/A"),
        "industry":      info.get("industry", "N/A"),
        "current_price": round(float(latest["Close"]), 2),
        "change_pct_1d": round(change_pct, 2),
        "52w_high":      round(float(df["High"].max()), 2),
        "52w_low":       round(float(df["Low"].min()), 2),
        "avg_volume_30d":int(df["Volume"].tail(30).mean()),
        "rsi_14":        round(float(df["RSI"].iloc[-1]), 1) if "RSI" in df else None,
        "macd":          round(float(df["MACD"].iloc[-1]), 4) if "MACD" in df else None,
        "macd_signal":   round(float(df["MACD_sig"].iloc[-1]), 4) if "MACD_sig" in df else None,
        "sma_50":        round(float(df["SMA_50"].iloc[-1]), 2) if "SMA_50" in df else None,
        "sma_200":       round(float(df["SMA_200"].iloc[-1]), 2) if "SMA_200" in df else None,
        "pe_ratio":      info.get("trailingPE"),
        "forward_pe":    info.get("forwardPE"),
        "pb_ratio":      info.get("priceToBook"),
        "market_cap":    info.get("marketCap"),
        "dividend_yield":info.get("dividendYield"),
        "beta":          info.get("beta"),
        "analyst_rating":info.get("recommendationMean"),
        "target_price":  info.get("targetMeanPrice"),
        "description":   (info.get("longBusinessSummary") or "")[:600],
    }


def ai_analyze(summary: dict, api_key: str) -> str:
    """Send summary to Claude and return the analysis text."""
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""You are a professional equity analyst. Analyze the following stock data and produce a clear, structured investment report.

DATA:
{json.dumps(summary, indent=2)}

Your report must include these sections (use ### headings):

### Company Overview
Brief description of what the company does and its market position.

### Technical Analysis
Interpret RSI, MACD, and moving averages. Note any bullish/bearish signals or crossovers.

### Valuation
Comment on P/E, P/B, target price vs current price, and whether the stock looks cheap, fair, or expensive.

### Key Risks
List 3–4 specific risks relevant to this company/sector.

### Opportunities
List 2–3 growth catalysts or tailwinds.

### Outlook
Short-term (1–3 months) and long-term (6–12 months) view.

### Recommendation
State clearly: **BUY**, **HOLD**, or **SELL** with a one-sentence justification.

Be direct and specific. Avoid vague language. Base everything on the data provided.
"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def fmt_large(n):
    if n is None:
        return "N/A"
    if n >= 1e12:
        return f"${n/1e12:.2f}T"
    if n >= 1e9:
        return f"${n/1e9:.2f}B"
    if n >= 1e6:
        return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"


def color_class(val, good_if_positive=True):
    if val is None:
        return "neutral"
    if val > 0:
        return "positive" if good_if_positive else "negative"
    if val < 0:
        return "negative" if good_if_positive else "positive"
    return "neutral"


def extract_recommendation(text: str) -> str:
    upper = text.upper()
    if "**BUY**" in upper or "RECOMMENDATION**\nBUY" in upper or ": BUY" in upper:
        return "BUY"
    if "**SELL**" in upper or ": SELL" in upper:
        return "SELL"
    if "**HOLD**" in upper or ": HOLD" in upper:
        return "HOLD"
    return "HOLD"


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📈 AI Stock Analyzer")
    st.markdown("---")

    api_key = st.text_input(
        "Anthropic API Key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Get yours at console.anthropic.com",
    )

    ticker = st.text_input("Ticker Symbol", value="AAPL").upper().strip()

    period = st.selectbox(
        "Data Period",
        options=["3mo", "6mo", "1y", "2y", "5y"],
        index=2,
    )

    analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("**Popular Tickers**")
    cols = st.columns(3)
    quick_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD"]
    for i, qt in enumerate(quick_tickers):
        if cols[i % 3].button(qt, key=f"q_{qt}"):
            ticker = qt
            analyze_btn = True

    st.markdown("---")
    st.caption("⚠️ Not financial advice. For educational use only.")


# ════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════════════════════════════

st.markdown("# 📊 Stock Analyzer")

if not ticker:
    st.info("Enter a ticker symbol in the sidebar and click **Analyze**.")
    st.stop()

if analyze_btn or st.session_state.get("last_ticker") == ticker:

    if not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
        st.stop()

    # ── Fetch data ───────────────────────────────────────────────────────
    with st.spinner(f"Fetching data for **{ticker}**…"):
        try:
            df, info = fetch_stock_data(ticker, period)
        except Exception as e:
            st.error(f"Could not fetch data: {e}")
            st.stop()

    if df.empty:
        st.error(f"No data found for ticker **{ticker}**. Check the symbol and try again.")
        st.stop()

    df = add_indicators(df)
    summary = build_summary(ticker, df, info)
    st.session_state["last_ticker"] = ticker

    # ── Header metrics ───────────────────────────────────────────────────
    price     = summary["current_price"]
    chg       = summary["change_pct_1d"]
    chg_class = color_class(chg)
    chg_arrow = "▲" if chg >= 0 else "▼"

    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{summary['company']} · {summary['sector']}</div>
      <div style="display:flex; align-items:baseline; gap:1rem;">
        <span class="metric-value">${price:,.2f}</span>
        <span class="metric-value {chg_class}" style="font-size:1.1rem;">
          {chg_arrow} {abs(chg):.2f}%
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ──────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    def kpi(col, label, value, css_class="neutral"):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value {css_class}">{value}</div>
        </div>""", unsafe_allow_html=True)

    rsi_val   = summary.get("rsi_14")
    rsi_class = "positive" if rsi_val and rsi_val < 30 else ("negative" if rsi_val and rsi_val > 70 else "neutral")

    kpi(k1, "Market Cap",    fmt_large(summary["market_cap"]))
    kpi(k2, "P/E (TTM)",     f"{summary['pe_ratio']:.1f}" if summary["pe_ratio"] else "N/A")
    kpi(k3, "52-Wk High",   f"${summary['52w_high']:,.2f}")
    kpi(k4, "52-Wk Low",    f"${summary['52w_low']:,.2f}")
    kpi(k5, "RSI (14)",     f"{rsi_val:.1f}" if rsi_val else "N/A", rsi_class)
    kpi(k6, "Beta",         f"{summary['beta']:.2f}" if summary["beta"] else "N/A")

    # ── Chart ────────────────────────────────────────────────────────────
    st.plotly_chart(build_chart(df, ticker), use_container_width=True)

    # ── AI Analysis ──────────────────────────────────────────────────────
    st.markdown("## 🤖 AI Analysis")

    with st.spinner("Claude is analyzing the stock…"):
        try:
            analysis = ai_analyze(summary, api_key)
        except Exception as e:
            st.error(f"AI analysis failed: {e}")
            st.stop()

    rec = extract_recommendation(analysis)
    badge_cls = {"BUY": "badge-buy", "SELL": "badge-sell", "HOLD": "badge-hold"}.get(rec, "badge-hold")

    st.markdown(
        f'<span class="badge {badge_cls}">{rec}</span>',
        unsafe_allow_html=True,
    )

    # Render markdown inside the styled box
    formatted = analysis.replace("### ", "<h3>").replace("\n", "<br>")
    # Simple approach: just render as streamlit markdown
    st.markdown(f'<div class="ai-box">', unsafe_allow_html=True)
    st.markdown(analysis)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Raw data expander ────────────────────────────────────────────────
    with st.expander("📋 Raw Data (last 30 days)"):
        display_cols = ["Open", "High", "Low", "Close", "Volume"]
        if "RSI" in df.columns:
            display_cols.append("RSI")
        if "MACD" in df.columns:
            display_cols.append("MACD")
        st.dataframe(
            df[display_cols].tail(30).style.format("{:.2f}"),
            use_container_width=True,
        )

    with st.expander("📊 Data Summary Sent to AI"):
        st.json(summary)

else:
    # Landing state
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; opacity:0.5;">
      <div style="font-size:4rem;">📈</div>
      <h3 style="font-family:'Space Mono',monospace;">Enter a ticker & click Analyze</h3>
      <p>Get AI-powered technical analysis, valuation, and a Buy/Hold/Sell recommendation.</p>
    </div>
    """, unsafe_allow_html=True)
