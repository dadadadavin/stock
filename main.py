# main.py
from fastapi import FastAPI, Query, Body, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from datetime import datetime
import logging
import statistics
import json
import ssl
import sys
import os

# Fix for Railway deployment - add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# SSL FIX: Add this at the TOP of main.py (right after imports)
ssl._create_default_https_context = ssl._create_unverified_context

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TA + Fundamental API", version="1.0")

# ---------- Startup event to initialize backend ----------
@app.on_event("startup")
async def startup_event():
    """Initialize the backend cleanup task"""
    try:
        from backend import start_cleanup_task
        start_cleanup_task()
        logger.info("Backend cleanup task started")
    except Exception as e:
        logger.error(f"Failed to start backend cleanup task: {e}")

# ---------- Try to import TA function ----------
try:
    from ta_engine_api import get_ta_json
except Exception:
    try:
        from ta_engine_final import get_ta_json
    except Exception:
        def get_ta_json(*args, **kwargs):
            raise RuntimeError(
                "get_ta_json not found. Add ta_engine_api.py or ta_engine_final.py with get_ta_json."
            )

# ---------- Try to import Fund/Scraper run() ----------
# Your scraper module should expose run(ticker, days, qty, post=False, api_url=..., pause=...)
try:
    from fund_sentiment import run as run_fundamental  # preferred name
except Exception:
    try:
        from news_sentiment import run as run_fundamental
    except Exception:
        def run_fundamental(*args, **kwargs):
            raise RuntimeError(
                "run() not found. Add fund_sentiment.py or news_sentiment.py exposing run(ticker, days, qty, post=False, api_url=...)"
            )

# ---------- Try to import historical fetcher (the module you built earlier) ----------
try:
    from historical_fetcher import fetch_history
except Exception:
    def fetch_history(*args, **kwargs):
        raise RuntimeError(
            "fetch_history not found. Add historical_fetcher.py with fetch_history(...) function."
        )

# ---------- Import backend module for WebSocket streaming ----------
try:
    from backend import handle_client_message, remove_client_from_all_subscriptions
except Exception:
    def handle_client_message(*args, **kwargs):
        raise RuntimeError(
            "handle_client_message not found. Add backend.py with handle_client_message(...) function."
        )
    
    async def remove_client_from_all_subscriptions(*args, **kwargs):
        raise RuntimeError(
            "remove_client_from_all_subscriptions not found. Add backend.py with remove_client_from_all_subscriptions(...) function."
        )

# ---------- Connection manager for WebSocket ----------
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}  # {client_id: websocket}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

manager = ConnectionManager()

# ---------- Root ----------
@app.get("/")
def root():
    return {
        "message": "TA + Fundamental API running",
        "usage": [
            "/analyze?ticker=AAPL",
            "/fundamental?ticker=BBCA&days=7&quantity=10",
            "POST /news?ticker=BBCA  (body = list of news items)",
            "/history?ticker=BBCA.JK&period=5y  (historical data)",
            "WS /ws/stream/{symbol}/{client_id}  (WebSocket streaming)"
        ]
    }

# ---------- TA analyze endpoint ----------
@app.get("/analyze")
def analyze(
    ticker: str = Query(...),
    range_opt: str = Query("last5d"),
    max_points: int = Query(150),
    downsample_method: str = Query("uniform"),
    resample_rule: str = Query("1D"),
    debug: bool = Query(False)
):
    try:
        result = get_ta_json(
            ticker=ticker,
            range_opt=range_opt,
            debug=debug,
            max_points=max_points,
            downsample=True,
            downsample_method=downsample_method,
            resample_rule=resample_rule
        )
        return JSONResponse(content=result)
    except Exception as e:
        logging.exception("TA analyze failed")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ---------- Run fundamental scraper endpoint ----------
@app.get("/fundamental")
def fundamental(
    ticker: str = Query(..., description="Stock code, e.g. BBCA or GOTO.JK"),
    days: int = Query(7, description="Lookback in days"),
    quantity: int = Query(10, description="How many items to return (trimmed)"),
    post: bool = Query(False, description="If true, scraper will POST results to configured API URL"),
    api_url: Optional[str] = Query(None, description="Override API URL used by scraper when post=true")
):
    """
    Runs your standalone scraper/sentiment (in-process). Blocking call.
    Calls run_fundamental(ticker, days, quantity, post=post, api_url=api_url) and returns result.
    """
    try:
        kwargs = {}
        if api_url:
            kwargs["api_url"] = api_url
        result = run_fundamental(ticker, days, quantity, post=post, **kwargs)
        return JSONResponse(content=result)
    except Exception as e:
        logging.exception("Fundamental run failed")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ---------- News ingest endpoint (keeps compatibility with scraper POST) ----------
class NewsItem(BaseModel):
    title: str
    link: Optional[str] = None
    sentiment: str
    positive: Optional[float] = None
    neutral: Optional[float] = None
    negative: Optional[float] = None
    query: Optional[str] = None
    timestamp: Optional[str] = None

@app.post("/news", status_code=201)
def ingest_news(ticker: Optional[str] = Query(None), items: List[NewsItem] = Body(...)):
    """
    Request a list of news items posted by the scraper.
    Stateless: returns a compact summary (no DB).
    """
    try:
        count = len(items)
        if count == 0:
            return {"ticker": ticker, "received": 0, "summary": None}

        positives = []
        neutrals = []
        negatives = []
        label_counts = {}
        titles = []
        links = []

        for it in items:
            if not it.timestamp:
                it.timestamp = datetime.utcnow().isoformat() + "Z"
            p = float(it.positive) if it.positive is not None else 0.0
            n = float(it.neutral) if it.neutral is not None else 0.0
            neg = float(it.negative) if it.negative is not None else 0.0
            positives.append(p)
            neutrals.append(n)
            negatives.append(neg)

            lab = (it.sentiment or "Neutral")
            label_counts[lab] = label_counts.get(lab, 0) + 1

            titles.append(it.title)
            links.append(it.link or "")

        def safe_avg(lst):
            try:
                return float(statistics.mean(lst)) if lst else 0.0
            except Exception:
                return 0.0

        avg_scores = {
            "positive": round(safe_avg(positives), 4),
            "neutral": round(safe_avg(neutrals), 4),
            "negative": round(safe_avg(negatives), 4)
        }

        majority = max(label_counts.items(), key=lambda x: (x[1], avg_scores["positive"]))[0] if label_counts else "Neutral"

        sample = [{"title": t, "link": l, "sentiment": s.sentiment, "timestamp": s.timestamp}
                  for t, l, s in zip(titles, links, items)][:5]

        summary = {
            "received": count,
            "ticker": (ticker or "").upper(),
            "majority_sentiment": majority,
            "by_label": label_counts,
            "avg_scores": avg_scores,
            "sample": sample
        }

        return {"status": "ok", "summary": summary}
    except Exception as e:
        logging.exception("News ingest failed")
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------------------------------
# NEW: Historical data endpoint (uses fetch_history from historical_fetcher)
# --------------------------------------------------------------------------
@app.get("/history")
def history_endpoint(
    ticker: str = Query(..., description="Ticker symbol, e.g. BBCA.JK"),
    period: Optional[str] = Query(None, description="Period like 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max"),
    start: Optional[str] = Query(None, description="Start date (YYYY-MM-DD) - takes precedence over period"),
    end: Optional[str] = Query(None, description="End date (YYYY-MM-DD) - used with start"),
    interval: Optional[str] = Query(None, description="Interval override like 1m,15m,1h,1d,1wk,1mo"),
    include_index: bool = Query(True, description="Include Date index in each row"),
    adjust_close: bool = Query(True, description="Include Adj Close if available"),
    force_refresh: bool = Query(False, description="Ignore cache and fetch fresh data"),
):
    """
    Return historical OHLCV data for a ticker.
    This endpoint calls fetch_history(...) from historical_fetcher.py and returns its JSON result.
    - start/end override period when provided.
    - interval may override automatic interval selection.
    - force_refresh bypasses cache in fetch_history.
    Note: historical_fetcher handles caching and TTL; this endpoint only forwards results.
    """
    try:
        # call the data provider function from your module
        example = fetch_history(
            ticker=ticker,
            period=period,
            start=start,
            end=end,
            interval=interval,
            include_index=include_index,
            adjust_close=adjust_close,
            force_refresh=force_refresh,
        )
        return JSONResponse(content=example)
    except Exception as e:
        logging.exception("History fetch failed")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# --------------------------------------------------------------------------
# NEW: WebSocket endpoint for streaming data from backend.py
# --------------------------------------------------------------------------
@app.websocket("/ws/stream/{symbol}/{client_id}")
async def websocket_stream(websocket: WebSocket, symbol: str, client_id: str):
    """
    WebSocket endpoint for streaming live stock data.
    Connect to: ws://localhost:8000/ws/stream/BBCA.JK/unique_client_id
    
    Client messages:
    - Initial request: {"req": "BBCA.JK", "id": "unique_client_id"}
    - Heartbeat: {"beat": "BBCA.JK", "id": "unique_client_id"}
    
    Server messages:
    - Day data: {"type": "day", "symbol": "BBCA.JK", "data": [...], "timestamp": "..."}
    - Live data: {"type": "live", "symbol": "BBCA.JK", "data": {...}, "timestamp": "..."}
    - Error: {"type": "error", "symbol": "BBA.JK", "message": "..."}
    - Timeout: {"type": "timeout", "symbol": "BBCA.JK", "message": "No heartbeat received"}
    """
    await manager.connect(websocket, client_id)
    logging.info(f"WebSocket connected for symbol: {symbol}, client_id: {client_id}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Process the message using the backend
            await handle_client_message(message, client_id, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logging.info(f"WebSocket disconnected for symbol: {symbol}, client_id: {client_id}")
        
        # Remove from all subscriptions
        await remove_client_from_all_subscriptions(client_id)
            
    except Exception as e:
        logging.exception(f"WebSocket error for symbol {symbol}, client_id {client_id}: {str(e)}")
        try:
            await manager.send_personal_message(
                json.dumps({"type": "error", "message": str(e)}), 
                websocket
            )
        except:
            pass  # Client might be disconnected
        finally:
            manager.disconnect(client_id)
            
            # Remove from all subscriptions
            await remove_client_from_all_subscriptions(client_id)

