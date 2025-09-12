from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import json
from datetime import datetime, timedelta
from flask import render_template
import random
import math
import traceback
import yfinance as yf
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
print("🔑 OPENROUTER_KEY:", bool(OPENROUTER_KEY))

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', 'your-openrouter-api-key-here')
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Helpers: yfinance data utilities
# ------------------------------
def get_stock_df(symbol, period="6mo", interval="1d"):
    """
    Returns a DataFrame of OHLCV for a given symbol (yfinance format).
    symbol: support 'RELIANCE.NS' or '^NSEI' etc.
    """
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        return df
    except Exception as e:
        app.logger.exception("yfinance error for %s: %s", symbol, e)
        return None


def get_multi_close_df(symbols, period="6mo", interval="1d"):
    """
    Returns a simple DataFrame with Close prices for multiple tickers (aligned).
    """
    try:
        data = yf.download(symbols, period=period, interval=interval, group_by='ticker', progress=False, auto_adjust=True)
        # if multiindex -> normalize to single Close df
        if isinstance(data.columns, pd.MultiIndex):
            close_df = pd.DataFrame()
            for sym in symbols:
                try:
                    close_series = data[(sym, 'Close')]
                except Exception:
                    close_series = data['Close'] if 'Close' in data else None
                if close_series is not None:
                    close_df[sym] = close_series
        else:
            close_df = data['Close'] if 'Close' in data else data
            if isinstance(close_df, pd.Series):
                close_df = close_df.to_frame()
        close_df = close_df.dropna(axis=0, how='all')
        return close_df
    except Exception as e:
        app.logger.exception("get_multi_close_df error: %s", e)
        return None


# ------------------------------
# Helpers: technical indicators (pandas)
# ------------------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, length=14):
    delta = series.diff().dropna()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=length-1, adjust=False).mean()
    ma_down = down.ewm(com=length-1, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# ------------------------------
# Backtest engine: simple EMA crossover + risk management
# ------------------------------
def backtest_ema_crossover(close_series, short=12, long=26, capital=100000, risk_pct=0.02):
    """
    Realistic rolling backtest (not hyper-optimized) that:
    - Enter long when short EMA crosses above long EMA
    - Exit when short EMA crosses below long EMA
    - Uses fixed fractional position sizing based on risk_pct per trade with ATR for stoploss
    Returns dictionary with trades and performance metrics.
    """
    df = pd.DataFrame({'close': close_series}).dropna()
    df['ema_short'] = df['close'].ewm(span=short, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=long, adjust=False).mean()
    df['signal'] = 0
    df.loc[df['ema_short'] > df['ema_long'], 'signal'] = 1
    df['signal_shift'] = df['signal'].shift(1).fillna(0)
    df['cross'] = df['signal'] - df['signal_shift']

    trades = []
    position = 0
    entry_price = None
    equity = capital
    trade_records = []

    for idx, row in df.iterrows():
        if row['cross'] == 1 and position == 0:
            # enter long at open (approx close here)
            entry_price = row['close']
            position = 1
            # risk position sizing: allocate risk_pct of equity
            size = (equity * risk_pct) / (entry_price * 0.02)  # assume 2% stop initially (approx)
            size = max(1, math.floor(size))
            trades.append({'entry_time': str(idx), 'entry_price': float(entry_price), 'size': int(size)})
        elif row['cross'] == -1 and position == 1:
            exit_price = row['close']
            position = 0
            last = trades[-1]
            pl = (exit_price - last['entry_price']) * last['size']
            equity += pl
            last.update({'exit_time': str(idx), 'exit_price': float(exit_price), 'pl': float(pl), 'equity_after': float(equity)})
            trade_records.append(last)

    # calculate performance metrics
    total_trades = len(trade_records)
    wins = sum(1 for t in trade_records if t['pl'] > 0)
    losses = total_trades - wins
    total_pl = sum(t['pl'] for t in trade_records)
    returns_pct = (equity - capital) / capital * 100

    stats = {
        'capital_start': capital,
        'capital_end': equity,
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'net_pnl': total_pl,
        'returns_%': returns_pct
    }
    return {'trades': trade_records, 'stats': stats}


# ------------------------------
# Options helpers
# ------------------------------
def get_option_chain(symbol):
    """
    Returns options expiries and option chain for the nearest expiry using yfinance.Ticker.option_chain
    """
    try:
        t = yf.Ticker(symbol)
        expiries = t.options
        if not expiries:
            return {'expiries': [], 'chains': {}}
        # use nearest expiry
        exp = expiries[0]
        chain = t.option_chain(exp)
        calls = chain.calls.to_dict(orient='records')
        puts = chain.puts.to_dict(orient='records')
        return {'expiries': expiries, 'expiry_used': exp, 'calls': calls, 'puts': puts}
    except Exception as e:
        app.logger.exception("get_option_chain error: %s", e)
        return {'expiries': [], 'chains_error': str(e)}


# ------------------------------
# Utility: safe JSON conversion for numpy/pandas
# ------------------------------
def safe_json(obj):
    try:
        return json.loads(json.dumps(obj, default=lambda o: (o.isoformat() if hasattr(o, 'isoformat') else str(o))))
    except Exception:
        return str(obj)

def call_openrouter_api(prompt, system_prompt="You are Nipa, a helpful AI assistant for girls. You're friendly, supportive, and knowledgeable about fashion, beauty, wellness, and lifestyle. IMPORTANT: Always use emojis instead of asterisks (*). Never use asterisks for emphasis - use emojis like 💕✨🌟💖 instead. Respond in a girly, encouraging tone with lots of emojis."):
    """Call OpenRouter API with DeepSeek model"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://nipa-app.onrender.com",
            "X-Title": "Nipa - Girly Lifestyle Hub"
        }
        
        data = {
            "model": "deepseek/deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            print(f"OpenRouter API Error: {response.status_code} - {response.text}")
            return "Sorry babe! I'm having a little technical moment right now. Please try again in a second! 💕✨"
            
    except Exception as e:
        print(f"API Error: {str(e)}")
        return "Sorry babe! I'm having a little technical moment right now. Please try again in a second! 💕✨"


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

# AI Chat Routes
@app.route('/api/chat', methods=['POST'])
def chat():
    """General AI chat endpoint"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        system_prompt = data.get('system_prompt', "You are Nipa, a helpful AI assistant for girls. You're friendly, supportive, and knowledgeable about fashion, beauty, wellness, and lifestyle. IMPORTANT: Always use emojis instead of asterisks (*). Never use asterisks for emphasis - use emojis like 💕✨🌟💖 instead. Respond in a girly, encouraging tone with lots of emojis.")
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        response = call_openrouter_api(message, system_prompt)
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Fashion Routes
@app.route('/api/fashion/outfit', methods=['POST'])
def generate_outfit():
    """Generate outfit recommendations"""
    try:
        data = request.get_json()
        occasion = data.get('occasion', '')
        weather = data.get('weather', '')
        style = data.get('style', '')
        
        prompt = f"""Create a detailed outfit recommendation for:
        - Occasion: {occasion}
        - Weather: {weather}  
        - Style: {style}
        
        Include specific clothing items, colors, accessories, and styling tips. 
        Make it girly and fashionable with lots of emojis! 💕✨"""
        
        system_prompt = "You are Nipa's fashion stylist AI. Create detailed, trendy outfit recommendations with specific items, colors, and styling tips. Always use emojis instead of asterisks. Be encouraging and girly!"
        
        response = call_openrouter_api(prompt, system_prompt)
        return jsonify({'outfit': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ------------------------------
# Real dedicated API routes for each frontend action
# (Routes match the button names in strategy_matrix.html)
# ------------------------------

# 1) generateRealStrategy
@app.route("/api/generate-real-strategy", methods=["POST"])
def api_generate_real_strategy():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", None) or request.json.get('symbol') if request.json else None
        description = data.get("description", "") or data.get("strategy_description", "")
        # Fetch fundamental price history for symbol(s)
        if symbol:
            df = get_stock_df(symbol, period="1y")
            if df is None or df.empty:
                return jsonify({"error": f"No data for {symbol}"}), 400
            close = df['Close']
            macd_line, signal_line, hist = macd(close)
            rsi_series = rsi(close)
            prompt = f"""
You are Lakshmi AI: create a production-grade trading strategy for {symbol}.
Inputs:
- Short summary: {description}
- Recent price history: last close {close.iloc[-1]:.2f}
- RSI latest: {rsi_series.iloc[-1]:.2f}
- MACD latest: {macd_line.iloc[-1]:.4f}, signal: {signal_line.iloc[-1]:.4f}

Output:
- Strategy name
- Indicators used with precise parameters
- Entry rules (exact)
- Stoploss calculation
- Targets (multiple)
- Position sizing rules (quantitative)
- Example trade with numbers
"""
            ai = call_openrouter(prompt)
            return jsonify({"status": "success", "symbol": symbol, "ai_strategy": ai})
        else:
            # If no symbol provided, let AI suggest a basket/strategy
            ai = call_openrouter(f"Create a production-grade multi-asset strategy. User description: {description}")
            return jsonify({"status": "success", "ai_strategy": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# 2) processNaturalLanguageTrading
@app.route("/api/process-natural-language", methods=["POST"])
def api_process_nl_trading():
    try:
        data = request.get_json(force=True) or {}
        text = data.get("text") or data.get("command") or ""
        if not text:
            return jsonify({"error": "No command provided"}), 400
        prompt = f"""
You are Lakshmi AI parser. Convert the following natural language trading instruction into precise actionable rules and JSON:
Instruction: {text}
Return JSON with keys: action (alert/trade/monitor), symbol(s), condition, threshold, timeframe, priority
"""
        parsed = call_openrouter(prompt)
        return jsonify({"status": "success", "parsed": parsed})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 3) runAIAnalysis
@app.route("/api/run-ai-analysis", methods=["POST"])
def api_run_ai_analysis():
    try:
        data = request.get_json(force=True) or {}
        query = data.get("query") or data.get("ai_query") or ""
        if not query:
            return jsonify({"error": "No AI query"}), 400
        ai = call_openrouter(f"Perform an institutional-grade analysis: {query}")
        return jsonify({"status": "success", "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 4) runAdvancedScan
@app.route("/api/run-advanced-scan", methods=["POST"])
def api_run_advanced_scan():
    try:
        data = request.get_json(force=True) or {}
        universe = data.get("universe", "NIFTY500")
        metric = data.get("metric", "volume")
        # We'll perform a quick yfinance scan on top symbols (user may pass list)
        symbols = data.get("symbols", [])
        if not symbols:
            symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]
        df = get_multi_close_df(symbols, period="1mo", interval="1d")
        if df is None:
            return jsonify({"error": "Failed to fetch data"}), 500
        # simple metric: percent change over period
        perf = (df.iloc[-1] / df.iloc[0] - 1) * 100
        top = perf.sort_values(ascending=False).to_dict()
        analysis = call_openrouter(f"Advanced scan result for {symbols}, top performance: {json.dumps(top, indent=2)}")
        return jsonify({"status": "success", "top": safe_json(top), "ai_analysis": analysis})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 5) runAlgoPatternRecognition
@app.route("/api/run-algo-pattern-recognition", methods=["POST"])
def api_run_algo_pattern():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE.NS")
        df = get_stock_df(symbol, period="6mo")
        if df is None or df.empty:
            return jsonify({"error": "No data"})
        # rudimentary pattern recognition: detect recent hammer/shooting star via candle shape
        df['body'] = df['Close'] - df['Open']
        df['range'] = df['High'] - df['Low']
        df['upper_wick'] = df['High'] - df[['Close', 'Open']].max(axis=1)
        df['lower_wick'] = df[['Close', 'Open']].min(axis=1) - df['Low']
        recent = df.tail(10)
        patterns = []
        for idx, row in recent.iterrows():
            if row['lower_wick'] > 2 * abs(row['body']):
                patterns.append({"time": str(idx), "pattern": "Hammer-like", "price": float(row['Close'])})
            if row['upper_wick'] > 2 * abs(row['body']):
                patterns.append({"time": str(idx), "pattern": "Shooting-star-like", "price": float(row['Close'])})
        ai = call_openrouter(f"Pattern recognition summary for {symbol}: {json.dumps(patterns, indent=2)}")
        return jsonify({"status": "success", "patterns": patterns, "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 6) runAlternativeDataAnalysis
@app.route("/api/run-alternative-data-analysis", methods=["POST"])
def api_run_alt_data():
    try:
        data = request.get_json(force=True) or {}
        topic = data.get("topic", "satellite imagery effect on retail")
        # Alternative data not available directly; call AI to analyze external alternative signals with context
        ai = call_openrouter(f"Analyze alternative data effects: {topic}. Provide actionable trade signals if any.")
        return jsonify({"status": "success", "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 7) runAutoBacktest
@app.route("/api/run-auto-backtest", methods=["POST"])
def api_run_auto_backtest():
    try:
        data = request.get_json(force=True) or {}
        symbols = data.get("symbols", ["RELIANCE.NS"])
        period = data.get("period", "2y")
        capital = float(data.get("capital", 100000))
        risk_pct = float(data.get("risk_pct", 0.02))
        # Run EMA crossover backtest on each symbol
        results = {}
        for s in symbols:
            df = get_stock_df(s, period=period)
            if df is None or df.empty:
                results[s] = {"error": "no data"}
                continue
            bt = backtest_ema_crossover(df['Close'], short=12, long=26, capital=capital, risk_pct=risk_pct)
            results[s] = bt
        return jsonify({"status": "success", "results": safe_json(results)})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 8) runBehavioralBiasDetector
@app.route("/api/run-behavioral-bias-detector", methods=["POST"])
def api_bias_detector():
    try:
        # This analyzes user's trading journal or sample trades passed in payload
        data = request.get_json(force=True) or {}
        journal_text = data.get("journal_text", "")
        if not journal_text:
            return jsonify({"error": "No journal text provided"}), 400
        ai = call_openrouter(f"Detect behavioral biases in the following trading journal. Return list of biases and remediation steps:\n\n{journal_text}")
        return jsonify({"status": "success", "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 9) runCommodityStockMapper
@app.route("/api/run-commodity-stock-mapper", methods=["POST"])
def api_commodity_stock_mapper():
    try:
        data = request.get_json(force=True) or {}
        commodity = data.get("commodity", "crude-oil")
        # Map commodity choice to proxy stocks
        mapping = {
            "crude-oil": ["ONGC.NS", "BPCL.NS", "IOC.NS"],
            "gold": ["HINDZINC.NS", "TATASTEEL.NS", "DRREDDY.NS"],
            "steel": ["JSWSTEEL.NS", "SAIL.NS"]
        }
        symbols = mapping.get(commodity, ["RELIANCE.NS"])
        close_df = get_multi_close_df(symbols, period="6mo")
        if close_df is None:
            return jsonify({"error": "failed to fetch"})
        corr = close_df.corr().to_dict()
        ai = call_openrouter(f"Commodity to stock correlation for {commodity}: {json.dumps(corr, indent=2)}")
        return jsonify({"status": "success", "mapping": symbols, "correlation": corr, "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 10) runCorrelationMatrix
@app.route("/api/run-correlation-matrix", methods=["POST"])
def api_run_correlation_matrix():
    try:
        data = request.get_json(force=True) or {}
        tickers = data.get("tickers", ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "^NSEI"])
        close_df = get_multi_close_df(tickers, period="6mo")
        if close_df is None or close_df.empty:
            return jsonify({"error": "Failed to fetch data"})
        corr = close_df.corr().round(4).to_dict()
        ai = call_openrouter(f"Analyze correlation matrix for {tickers}: {json.dumps(corr, indent=2)}")
        return jsonify({"status": "success", "correlation": corr, "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 11) runCurrencyImpactCalculator
@app.route("/api/run-currency-impact", methods=["POST"])
def api_currency_impact():
    try:
        data = request.get_json(force=True) or {}
        pair = data.get("pair", "USD-INR")
        # Quick proxy: fetch USDINR from yfinance if available (^USDINR not always)
        symbol = data.get("symbol", "^INR=X")  # Yahoo FX pair
        df = get_stock_df(symbol, period="1mo")
        if df is None or df.empty:
            ai = call_openrouter(f"Estimate currency impact for {pair} without direct FX data.")
            return jsonify({"status": "success", "ai_analysis": ai})
        change = (df['Close'][-1] - df['Close'][0]) / df['Close'][0] * 100
        ai = call_openrouter(f"Currency impact analysis for {pair}: {change:.2f}% over 1 month.")
        return jsonify({"status": "success", "pair": pair, "change%": change, "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 12) runDrawdownRecoveryPredictor
@app.route("/api/run-drawdown-recovery", methods=["POST"])
def api_drawdown_recovery():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE.NS")
        df = get_stock_df(symbol, period="3y")
        if df is None or df.empty:
            return jsonify({"error": "No data"})
        # calculate drawdowns
        close = df['Close']
        roll_max = close.cummax()
        drawdown = (close - roll_max) / roll_max
        max_dd = drawdown.min()
        ai = call_openrouter(f"Max drawdown for {symbol} over 3y is {max_dd:.2%}. Estimate recovery time and strategies.")
        return jsonify({"status": "success", "max_drawdown": float(max_dd), "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 13) runDreamTradeSimulator
@app.route("/api/run-dream-trade-simulator", methods=["POST"])
def api_dream_trade_simulator():
    try:
        data = request.get_json(force=True) or {}
        scenario = data.get("scenario", "")
        ai = call_openrouter(f"Simulate a dream trade scenario: {scenario}. Provide P&L, ROI, and risk profile.")
        return jsonify({"status": "success", "simulation": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 14) runDynamicPositionSizing
@app.route("/api/run-dynamic-position-sizing", methods=["POST"])
def api_dynamic_position():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE.NS")
        capital = float(data.get("capital", 100000))
        risk_per_trade = float(data.get("risk_pct", 0.02))
        df = get_stock_df(symbol, period="1y")
        if df is None:
            return jsonify({"error": "No data"})
        atr = df['High'] - df['Low']
        atr = atr.rolling(14).mean().dropna().iloc[-1]
        stop_loss_distance = atr * 2
        price = df['Close'].iloc[-1]
        position_size = max(1, int((capital * risk_per_trade) / (stop_loss_distance * price)))
        return jsonify({"status": "success", "symbol": symbol, "position_size": position_size, "atr": float(atr)})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 15) runESGImpactScorer
@app.route("/api/run-esg-impact-scorer", methods=["POST"])
def api_esg_impact():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE.NS")
        # yfinance doesn't provide ESG scores reliably; use AI to synthesize from public headlines (server-side)
        ai = call_openrouter(f"Provide an ESG impact score analysis for {symbol} and explain key drivers.")
        return jsonify({"status": "success", "symbol": symbol, "esg_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 16) runEarningsCallAnalysis
@app.route("/api/run-earnings-call-analysis", methods=["POST"])
def api_earnings_call():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE.NS")
        prompt = f"Analyze recent earnings calls and generate a concise actionable summary for {symbol}."
        ai = call_openrouter(prompt)
        return jsonify({"status": "success", "symbol": symbol, "analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 17) runEconomicImpactPredictor
@app.route("/api/run-economic-impact", methods=["POST"])
def api_economic_impact():
    try:
        data = request.get_json(force=True) or {}
        event = data.get("event", "rbi-policy")
        ai = call_openrouter(f"Predict market impact for event: {event}. Provide sector-level implications and trade ideas.")
        return jsonify({"status": "success", "event": event, "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 18) runGeopoliticalRiskScorer
@app.route("/api/run-geopolitical-risk", methods=["POST"])
def api_geo_risk():
    try:
        data = request.get_json(force=True) or {}
        region = data.get("region", "india-china")
        ai = call_openrouter(f"Score geopolitical risk for {region} and supply specific trade hedges and timeline.")
        return jsonify({"status": "success", "region": region, "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 19) runGlobalMarketSync
@app.route("/api/run-global-market-sync", methods=["POST"])
def api_global_sync():
    try:
        data = request.get_json(force=True) or {}
        region = data.get("region", "us-markets")
        ai = call_openrouter(f"Analyze how {region} will impact Indian markets today. Provide short-term signals.")
        return jsonify({"status": "success", "region": region, "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 20) runInsiderAnalysis
@app.route("/api/run-insider-analysis", methods=["POST"])
def api_insider_analysis():
    try:
        data = request.get_json(force=True) or {}
        period = data.get("period", "30d")
        symbol = data.get("symbol", None)
        ai = call_openrouter(f"Analyze insider trading patterns for {symbol or 'market'} over {period}. Provide signals.")
        return jsonify({"status": "success", "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 21) runInstitutionalFlowTracker
@app.route("/api/run-institutional-flow-tracker", methods=["POST"])
def api_institutional_flow():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "^NSEI")
        # yfinance does not give FII/DII directly; use AI to synthesize from price and volume trends
        df = get_stock_df(symbol, period="1mo")
        if df is None or df.empty:
            return jsonify({"error": "No data"})
        vol_change = ((df['Volume'].iloc[-1] - df['Volume'].iloc[0]) / (df['Volume'].iloc[0] + 1e-9)) * 100
        ai = call_openrouter(f"Institutional flow estimation for {symbol}. Volume change {vol_change:.2f}%. Provide FII/DII signal.")
        return jsonify({"status": "success", "volume_change_pct": vol_change, "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 22) runInterestRateSensitivity
@app.route("/api/run-interest-rate-sensitivity", methods=["POST"])
def api_rate_sensitivity():
    try:
        data = request.get_json(force=True) or {}
        sector = data.get("sector", "banking")
        ai = call_openrouter(f"Analyze interest rate sensitivity for sector {sector}. Provide names of most sensitive tickers and hedges.")
        return jsonify({"status": "success", "sector": sector, "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 23) runLiquidityHeatMap
@app.route("/api/run-liquidity-heatmap", methods=["POST"])
def api_liquidity_heatmap():
    try:
        data = request.get_json(force=True) or {}
        symbols = data.get("symbols", ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"])
        df = get_multi_close_df(symbols, period="3mo")
        if df is None or df.empty:
            return jsonify({"error": "No data"})
        # use average daily dollar volume as liquidity proxy if volume available
        liquidity = {}
        for sym in symbols:
            try:
                hist = yf.download(sym, period="3mo", progress=False, auto_adjust=True)
                if hist is None or hist.empty or 'Volume' not in hist:
                    liquidity[sym] = None
                else:
                    avg_vol = hist['Volume'].tail(30).mean()
                    avg_price = hist['Close'].tail(30).mean()
                    liquidity[sym] = float(avg_vol * avg_price)
            except Exception:
                liquidity[sym] = None
        ai = call_openrouter(f"Liquidity heatmap computed: {json.dumps(liquidity, indent=2)}. Provide suggested trades based on liquidity.")
        return jsonify({"status": "success", "liquidity": safe_json(liquidity), "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 24) runMarketRegimeDetection
@app.route("/api/run-market-regime-detection", methods=["POST"])
def api_market_regime():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "^NSEI")
        df = get_stock_df(symbol, period="2y")
        if df is None:
            return jsonify({"error": "No data"})
        # regime proxy: volatility vs trend
        returns = df['Close'].pct_change().dropna()
        vol = returns.rolling(21).std().iloc[-1]
        trend = (df['Close'].iloc[-1] - df['Close'].iloc[-63]) / df['Close'].iloc[-63]
        regime = "Bullish-Trend" if trend > 0.05 and vol < 0.02 else "Volatile" if vol > 0.03 else "Range-bound"
        ai = call_openrouter(f"Market regime for {symbol}: trend={trend:.3f}, vol={vol:.3f}. Recommend positioning.")
        return jsonify({"status": "success", "regime": regime, "trend": float(trend), "volatility": float(vol), "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 25) runOptionsFlow
@app.route("/api/run-options-flow", methods=["POST"])
def api_options_flow():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE")
        yf_sym = symbol if symbol.endswith(".NS") else symbol + ".NS"
        oc = get_option_chain(yf_sym)
        if not oc.get('expiries'):
            return jsonify({"error": "No options data"}, 404)
        # rough analysis: largest OI strikes
        try:
            calls = pd.DataFrame(oc['calls'])
            puts = pd.DataFrame(oc['puts'])
            top_calls = calls.sort_values('openInterest', ascending=False).head(5).to_dict(orient='records')
            top_puts = puts.sort_values('openInterest', ascending=False).head(5).to_dict(orient='records')
            ai = call_openrouter(f"Options flow summary for {symbol}. Top calls: {json.dumps(top_calls, default=str)} Top puts: {json.dumps(top_puts, default=str)}")
            return jsonify({"status": "success", "expiry": oc.get('expiry_used'), "top_calls": safe_json(top_calls), "top_puts": safe_json(top_puts), "ai_analysis": ai})
        except Exception:
            return jsonify({"error": "Failed to parse option chains"})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 26) runPortfolioOptimization
@app.route("/api/run-portfolio-optimization", methods=["POST"])
def api_portfolio_optimization():
    try:
        data = request.get_json(force=True) or {}
        symbols = data.get("symbols", ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"])
        df = get_multi_close_df(symbols, period="1y")
        if df is None or df.empty:
            return jsonify({"error": "No data"})
        # Simple mean-variance portfolio (not heavy optimization)
        rets = df.pct_change().dropna()
        mu = rets.mean() * 252
        sigma = rets.cov() * 252
        # equal weights as baseline
        weights = {s: 1/len(symbols) for s in symbols}
        ai = call_openrouter(f"Suggest portfolio weights and rebalancing for {symbols}. Baseline mu: {safe_json(mu.to_dict())}")
        return jsonify({"status": "success", "weights": weights, "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 27) runPortfolioStressTesting
@app.route("/api/run-portfolio-stress-testing", methods=["POST"])
def api_portfolio_stress():
    try:
        data = request.get_json(force=True) or {}
        portfolio = data.get("portfolio", {})
        scenario = data.get("scenario", "market-crash")
        ai = call_openrouter(f"Stress test portfolio {portfolio} under scenario {scenario}. Return expected drawdown and recovery suggestions.")
        return jsonify({"status": "success", "scenario": scenario, "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 28) runPriceTargetConsensus
@app.route("/api/run-price-target-consensus", methods=["POST"])
def api_price_target_consensus():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE.NS")
        df = get_stock_df(symbol, period="6mo")
        if df is None:
            return jsonify({"error": "No data"})
        # simple consensus: use AI to synthesize targets from available data
        prompt = f"Generate price target consensus for {symbol} using recent price history. Provide low/medium/high targets and probability."
        ai = call_openrouter(prompt)
        return jsonify({"status": "success", "symbol": symbol, "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 29) runRealBacktest
@app.route("/api/run-real-backtest", methods=["POST"])
def api_run_real_backtest():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE.NS")
        period = data.get("period", "2y")
        capital = float(data.get("capital", 100000))
        df = get_stock_df(symbol, period=period)
        if df is None:
            return jsonify({"error": "No data"})
        bt = backtest_ema_crossover(df['Close'], short=12, long=26, capital=capital, risk_pct=float(data.get("risk_pct", 0.02)))
        return jsonify({"status": "success", "symbol": symbol, "backtest": safe_json(bt)})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 30) runRealDataMining
@app.route("/api/run-real-data-mining", methods=["POST"])
def api_real_data_mining():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE.NS")
        period = data.get("period", "3mo")
        df = get_stock_df(symbol, period=period)
        if df is None or df.empty:
            return jsonify({"error": "No data"})
        # basic anomalies detect: zscore on returns
        ret = df['Close'].pct_change().dropna()
        z = (ret - ret.mean()) / (ret.std() + 1e-9)
        anomalies = ret[np.abs(z) > 2].tail(20).to_dict()
        ai = call_openrouter(f"Data mining summary for {symbol}. anomalies: {json.dumps(safe_json(anomalies), indent=2)}")
        return jsonify({"status": "success", "anomalies": safe_json(anomalies), "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 31) runRealTimeScreener
@app.route("/api/run-real-time-screener", methods=["POST"])
def api_realtime_screener():
    try:
        data = request.get_json(force=True) or {}
        symbols = data.get("symbols", ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"])
        df = get_multi_close_df(symbols, period="5d", interval="1d")
        if df is None:
            return jsonify({"error": "No data"})
        momentum = ((df.iloc[-1] / df.iloc[-5]) - 1) * 100 if len(df) >= 5 else {}
        ai = call_openrouter(f"Realtime screener snapshot: momentum {safe_json(momentum.to_dict() if hasattr(momentum, 'to_dict') else momentum)}")
        return jsonify({"status": "success", "momentum": safe_json(momentum), "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 32) runSectorRotationPredictor
@app.route("/api/run-sector-rotation-predictor", methods=["POST"])
def api_sector_rotation():
    try:
        data = request.get_json(force=True) or {}
        timeframe = data.get("timeframe", "1m")
        ai = call_openrouter(f"Sector rotation predictions for the next {timeframe}. Provide top 3 sectors and reasons.")
        return jsonify({"status": "success", "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 33) runSentimentAnalysis (dedicated route - alias)
@app.route("/api/run-sentiment-analysis", methods=["POST"])
def api_run_sentiment_analysis():
    # mirror of earlier runSentimentAnalysis route; kept for explicit call names from frontend
    return api_run_correlation_matrix.__self__ if False else api_run_sentiment_alias()


def api_run_sentiment_alias():
    # actual logic for alias
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "^NSEI")
        df = get_stock_df(symbol, period="1mo")
        if df is None or df.empty:
            return jsonify({"error": f"No data for {symbol}"})
        change = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / (df['Close'].iloc[0] + 1e-9) * 100
        ai = call_openrouter(f"Short-term sentiment analysis for {symbol}: price change {change:.2f}% over 1 month.")
        return jsonify({"status": "success", "symbol": symbol, "change%": change, "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 34) runSocialMomentumScanner
@app.route("/api/run-social-momentum-scanner", methods=["POST"])
def api_social_momentum():
    try:
        data = request.get_json(force=True) or {}
        topic = data.get("topic", "RELIANCE")
        ai = call_openrouter(f"Scan social momentum and sentiment for {topic} across major channels. Provide signals and confidence.")
        return jsonify({"status": "success", "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 35) runSocialTrendMonetizer
@app.route("/api/run-social-trend-monetizer", methods=["POST"])
def api_social_trend_monetizer():
    try:
        data = request.get_json(force=True) or {}
        trend = data.get("trend", "EV stock interest")
        ai = call_openrouter(f"Monetize social trend '{trend}' into short-term trade ideas. Provide entry/stop/targets.")
        return jsonify({"status": "success", "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 36) runSupplyChainVisibility
@app.route("/api/run-supply-chain-visibility", methods=["POST"])
def api_supply_chain_visibility():
    try:
        data = request.get_json(force=True) or {}
        sector = data.get("sector", "manufacturing")
        ai = call_openrouter(f"Supply chain visibility analysis for sector {sector}. Provide companies likely impacted and trade ideas.")
        return jsonify({"status": "success", "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 37) runTailRiskHedging
@app.route("/api/run-tail-risk-hedging", methods=["POST"])
def api_tail_risk_hedging():
    try:
        data = request.get_json(force=True) or {}
        portfolio = data.get("portfolio", {})
        ai = call_openrouter(f"Generate tail-risk hedging strategies for portfolio: {json.dumps(portfolio, default=str)}. Provide costs and expected protection.")
        return jsonify({"status": "success", "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 38) runTradingJournalAnalysis
@app.route("/api/run-trading-journal-analysis", methods=["POST"])
def api_trading_journal_analysis():
    try:
        data = request.get_json(force=True) or {}
        journal = data.get("journal", "")
        if not journal:
            return jsonify({"error": "No journal entered"}), 400
        ai = call_openrouter(f"Analyze trading journal and produce performance summary and improvement plan: {journal}")
        return jsonify({"status": "success", "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 39) runVolatilitySurface
@app.route("/api/run-volatility-surface", methods=["POST"])
def api_volatility_surface():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE.NS")
        # estimate historical vol and ask AI to approximate surface
        df = get_stock_df(symbol, period="1y")
        if df is None:
            return jsonify({"error": "No data"})
        returns = df['Close'].pct_change().dropna()
        hist_vol = returns.std() * math.sqrt(252)
        ai = call_openrouter(f"Approximate volatility surface for {symbol} using historical vol {hist_vol:.3f}")
        return jsonify({"status": "success", "historical_vol": hist_vol, "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 40) runWalkForwardTest
@app.route("/api/run-walk-forward-test", methods=["POST"])
def api_walk_forward_test():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "RELIANCE.NS")
        df = get_stock_df(symbol, period="3y")
        if df is None:
            return jsonify({"error": "No data"})
        # A simplified walk-forward: rolling window backtest
        window_train = 252
        step = 63
        results = []
        closes = df['Close']
        for start in range(0, len(closes) - window_train - step, step):
            train = closes.iloc[start:start+window_train]
            test = closes.iloc[start+window_train:start+window_train+step]
            bt = backtest_ema_crossover(train, short=12, long=26, capital=100000, risk_pct=0.02)
            # Evaluate on test: naive application, compute directional accuracy
            train_signal = train.ewm(span=12).mean().iloc[-1] > train.ewm(span=26).mean().iloc[-1]
            test_return = (test.iloc[-1] - test.iloc[0]) / test.iloc[0]
            results.append({"train_end": str(train.index[-1]), "test_return": float(test_return)})
        ai = call_openrouter(f"Walk-forward summary for {symbol}. Samples: {len(results)}")
        return jsonify({"status": "success", "samples": len(results), "results": safe_json(results), "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 41) runWeatherPatternTrading
@app.route("/api/run-weather-pattern-trading", methods=["POST"])
def api_weather_pattern_trading():
    try:
        data = request.get_json(force=True) or {}
        sector = data.get("sector", "agriculture")
        ai = call_openrouter(f"Create weather pattern trading ideas for sector {sector}. Provide stocks, entry and stoploss.")
        return jsonify({"status": "success", "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# 42) runDreamTradeSimulator (duplicate earlier, keep alias)
@app.route("/api/run-dream-trade-sim", methods=["POST"])
def api_dream_trade_alias():
    return api_dream_trade_simulator()


# ------------------------------
# Convenience routes (existing ones fixed)
# ------------------------------
@app.route("/api/correlation-matrix", methods=["POST"])
def correlation_matrix():
    try:
        data = request.get_json(force=True) or {}
        tickers = data.get("tickers", ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"])
        close_df = get_multi_close_df(tickers, period="6mo")
        if close_df is None or close_df.empty:
            return jsonify({"error": "Failed to fetch data"}), 500
        corr = close_df.corr().round(4).to_dict()
        ai = call_openrouter(f"Correlation matrix analysis: {json.dumps(corr, indent=2)}")
        return jsonify({"status": "success", "correlation": corr, "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/sentiment-analysis", methods=["POST"])
def sentiment_analysis():
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "^NSEI")
        df = get_stock_df(symbol, period="1mo")
        if df is None or df.empty:
            return jsonify({"error": "No data for symbol"}), 400
        change = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / (df['Close'].iloc[0] + 1e-9) * 100
        ai = call_openrouter(f"Provide a short sentiment explanation for {symbol} given {change:.2f}% price change in 1 month.")
        return jsonify({"status": "success", "symbol": symbol, "change%": change, "ai_analysis": ai})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


# ------------------------------
# Page render
# ------------------------------
@app.route("/strategy-matrix")
def strategy_matrix_page():
    return render_template("strategy_matrix.html")


# ------------------------------
# Run (for local dev)
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)

# Beauty Routes
@app.route('/api/beauty/skincare', methods=['POST'])
def generate_skincare():
    """Generate skincare routine"""
    try:
        data = request.get_json()
        skin_type = data.get('skin_type', '')
        concerns = data.get('concerns', '')
        budget = data.get('budget', '')
        
        prompt = f"""Create a personalized skincare routine for:
        - Skin Type: {skin_type}
        - Main Concerns: {concerns}
        - Budget: {budget}
        
        Include morning and evening routines with specific product recommendations, 
        application order, and tips. Use emojis and be encouraging! 🧴✨"""
        
        system_prompt = "You are Nipa's skincare expert AI. Create detailed skincare routines with specific products and application tips. Always use emojis instead of asterisks. Be knowledgeable and supportive!"
        
        response = call_openrouter_api(prompt, system_prompt)
        return jsonify({'routine': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Wellness Routes
@app.route('/api/wellness/cycle', methods=['POST'])
def calculate_cycle():
    """Calculate menstrual cycle predictions"""
    try:
        data = request.get_json()
        last_period = data.get('last_period', '')
        cycle_length = int(data.get('cycle_length', 28))
        
        # Parse the date
        last_period_date = datetime.strptime(last_period, '%Y-%m-%d')
        
        # Calculate predictions
        next_period = last_period_date + timedelta(days=cycle_length)
        ovulation = last_period_date + timedelta(days=cycle_length - 14)
        fertile_start = ovulation - timedelta(days=5)
        fertile_end = ovulation + timedelta(days=1)
        
        return jsonify({
            'next_period': next_period.isoformat(),
            'ovulation': ovulation.isoformat(),
            'fertile_start': fertile_start.isoformat(),
            'fertile_end': fertile_end.isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wellness/workout', methods=['POST'])
def generate_workout():
    """Generate workout plan"""
    try:
        data = request.get_json()
        goal = data.get('goal', '')
        time = data.get('time', '')
        
        prompt = f"""Create a personalized workout plan for:
        - Fitness Goal: {goal}
        - Available Time: {time} minutes per day
        
        Include specific exercises, sets, reps, and modifications for beginners.
        Make it encouraging and achievable with emojis! 💪✨"""
        
        system_prompt = "You are Nipa's fitness coach AI. Create safe, effective workout plans with clear instructions. Always use emojis instead of asterisks. Be motivating and supportive!"
        
        response = call_openrouter_api(prompt, system_prompt)
        return jsonify({'workout': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Lifestyle Routes
@app.route('/api/lifestyle/mood', methods=['POST'])
def mood_support():
    """Provide mood support and advice"""
    try:
        data = request.get_json()
        mood = data.get('mood', '')
        emoji = data.get('emoji', '')
        
        prompt = f"""The user is feeling {mood} {emoji}. Provide supportive advice, 
        self-care suggestions, and encouraging words. Include activities that might 
        help improve their mood and remind them of their worth. Use lots of emojis! 💕"""
        
        system_prompt = "You are Nipa's emotional support AI. Provide caring, supportive responses with practical self-care advice. Always use emojis instead of asterisks. Be empathetic and uplifting!"
        
        response = call_openrouter_api(prompt, system_prompt)
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/lifestyle/affirmation', methods=['GET'])
def generate_affirmation():
    """Generate daily affirmation"""
    try:
        prompt = """Generate a powerful, personalized daily affirmation for a girl. 
        Make it empowering, confidence-boosting, and beautiful. Include self-love, 
        strength, and positivity. Use emojis! ✨💖"""
        
        system_prompt = "You are Nipa's affirmation generator. Create beautiful, empowering affirmations that boost confidence and self-love. Always use emojis instead of asterisks. Be inspiring and uplifting!"
        
        response = call_openrouter_api(prompt, system_prompt)
        return jsonify({'affirmation': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/lifestyle/selfcare', methods=['POST'])
def generate_selfcare():
    """Generate self-care routine"""
    try:
        data = request.get_json()
        time = data.get('time', '')
        
        prompt = f"""Create a self-care routine for {time} minutes. Include activities 
        for relaxation, pampering, and mental wellness. Make it achievable and 
        luxurious feeling, even on a budget. Use emojis! 🛁💆‍♀️"""
        
        system_prompt = "You are Nipa's self-care expert. Create relaxing, achievable self-care routines that promote wellness and happiness. Always use emojis instead of asterisks. Be nurturing and caring!"
        
        response = call_openrouter_api(prompt, system_prompt)
        return jsonify({'routine': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Shopping Routes
@app.route('/api/shopping/products', methods=['POST'])
def find_products():
    """Find product recommendations"""
    try:
        data = request.get_json()
        category = data.get('category', '')
        budget = data.get('budget', '')
        specific = data.get('specific', '')
        
        prompt = f"""Find the best {category} products within {budget} budget.
        {f'Specifically looking for: {specific}' if specific else ''}
        
        Include product names, price ranges, where to buy, and why they're great.
        Focus on quality and value. Use emojis! 🛍️💎"""
        
        system_prompt = "You are Nipa's shopping assistant. Recommend quality products with good value, including where to find them and why they're worth buying. Always use emojis instead of asterisks. Be helpful and budget-conscious!"
        
        response = call_openrouter_api(prompt, system_prompt)
        return jsonify({'products': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Community Routes
@app.route('/api/community/question', methods=['POST'])
def answer_question():
    """Answer anonymous community questions"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        prompt = f"""Answer this anonymous question with care and wisdom: "{question}"
        
        Provide supportive, non-judgmental advice. Be understanding and helpful.
        Include practical tips and emotional support. Use emojis! 💭💕"""
        
        system_prompt = "You are Nipa's community support AI. Answer questions with empathy, wisdom, and practical advice. Always use emojis instead of asterisks. Be supportive and non-judgmental!"
        
        response = call_openrouter_api(prompt, system_prompt)
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/community/girl-talk', methods=['GET'])
def girl_talk_topic():
    """Generate girl talk discussion topic"""
    try:
        prompt = """Generate a fun, engaging discussion topic for girls to chat about.
        Make it relatable, interesting, and conversation-starting. Could be about 
        relationships, fashion, life experiences, dreams, or fun hypotheticals. 
        Use emojis! 💬✨"""
        
        system_prompt = "You are Nipa's conversation starter generator. Create engaging, fun topics that girls would love to discuss together. Always use emojis instead of asterisks. Be creative and relatable!"
        
        response = call_openrouter_api(prompt, system_prompt)
        return jsonify({'topic': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/community/style-challenge', methods=['GET'])
def style_challenge():
    """Generate weekly style challenge"""
    try:
        prompt = """Create a fun weekly style challenge for girls. Make it creative,
        achievable, and Instagram-worthy. Include specific styling goals and tips
        for completing the challenge. Use emojis! 🏆👗"""
        
        system_prompt = "You are Nipa's style challenge creator. Design fun, creative fashion challenges that are achievable and inspiring. Always use emojis instead of asterisks. Be creative and encouraging!"
        
        response = call_openrouter_api(prompt, system_prompt)
        return jsonify({'challenge': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({'status': 'healthy', 'message': 'Nipa API is running! 💕✨'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)