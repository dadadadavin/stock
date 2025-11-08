# backend.py
"""
Advanced backend helper for live streaming data from Yahoo Finance.

This module provides a subscription-based streaming system for Indonesia Stock Exchange (IDX) data:
1. Clients can request full day data with {"req": "CODE", "id": "client_id"}
2. Clients can maintain subscriptions with {"beat": "CODE", "id": "client_id"}
3. Data is shared across multiple clients for the same stock
4. Automatic cleanup of inactive subscriptions
5. Data is filtered to IDX trading hours (09:00-11:30, 13:30-15:00 Jakarta time)
6. Sends minute-by-minute updates with OHLCV data

Public API:
- SubscriptionManager: Global subscription management
- handle_client_message(message, client_id, websocket): Process client messages
- remove_client_from_all_subscriptions(client_id): Remove client from all subscriptions
- start_cleanup_task(): Start the cleanup task when event loop is running
"""

from typing import Dict, Any, List, Optional, Set
import asyncio
import datetime
import json
import logging
import time

import pandas as pd
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function to check if a timestamp is within IDX trading hours
def in_trading_hours(ts):
    """Check if timestamp is within Indonesia Stock Exchange trading hours"""
    try:
        # Convert to Jakarta time if not already
        if ts.tz is None:
            ts = ts.tz_localize('UTC')
        if ts.tzinfo.zone != 'Asia/Jakarta':
            ts = ts.tz_convert('Asia/Jakarta')
        
        # Morning session: 09:00 to 11:30
        morning_start = ts.replace(hour=9, minute=0, second=0, microsecond=0)
        morning_end = ts.replace(hour=11, minute=30, second=0, microsecond=0)
        
        # Afternoon session: 13:30 to 15:00
        afternoon_start = ts.replace(hour=13, minute=30, second=0, microsecond=0)
        afternoon_end = ts.replace(hour=15, minute=0, second=0, microsecond=0)
        
        return (morning_start <= ts <= morning_end) or (afternoon_start <= ts <= afternoon_end)
    except Exception as e:
        logger.error(f"Error in trading hours check: {e}")
        return False

# Helper function to normalize DataFrame columns
def normalize_dataframe(df):
    """Normalize DataFrame with multi-index columns to simple columns"""
    if df is None or df.empty:
        return df
    
    # Check if we have multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten multi-index columns
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    
    # Extract ticker symbol from column names if present
    ticker = None
    for col in df.columns:
        if 'BBCA.JK' in col or 'TICKER' in col:
            ticker = col.split('_')[1] if '_' in col else col
            break
    
    # Rename columns to standard names
    column_mapping = {}
    for col in df.columns:
        if 'Close' in col:
            column_mapping[col] = 'Close'
        elif 'Open' in col:
            column_mapping[col] = 'Open'
        elif 'High' in col:
            column_mapping[col] = 'High'
        elif 'Low' in col:
            column_mapping[col] = 'Low'
        elif 'Volume' in col:
            column_mapping[col] = 'Volume'
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    return df

# Global subscription manager
class SubscriptionManager:
    def __init__(self):
        # Structure: {symbol: SubscriptionData}
        self.active_stocks: Dict[str, 'SubscriptionData'] = {}
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
    
    async def get_subscription(self, symbol: str) -> 'SubscriptionData':
        """Get or create subscription data for a symbol"""
        symbol = symbol.strip().upper()
        
        async with self._lock:
            if symbol not in self.active_stocks:
                self.active_stocks[symbol] = SubscriptionData(symbol)
            return self.active_stocks[symbol]
    
    async def cleanup_inactive_subscriptions(self):
        """Remove subscriptions with no active subscribers"""
        async with self._lock:
            symbols_to_remove = []
            for symbol, sub in self.active_stocks.items():
                if not sub.has_active_subscribers() and sub.is_inactive():
                    symbols_to_remove.append(symbol)
            
            for symbol in symbols_to_remove:
                del self.active_stocks[symbol]
                logger.info(f"Cleaned up subscription for {symbol}")

# Global instance
subscription_manager = SubscriptionManager()

class SubscriptionData:
    def __init__(self, symbol: str):
        self.symbol = symbol.strip().upper()
        self.subscribers: Dict[str, Any] = {}  # {client_id: websocket}
        self.last_heartbeat: Optional[datetime.datetime] = None
        self.day_cache: Optional[pd.DataFrame] = None
        self.live_cache: Optional[pd.DataFrame] = None
        self.timer_running: bool = False
        self.last_fetch_time: Optional[datetime.datetime] = None
        self.data_fetch_task: Optional[asyncio.Task] = None
        self.heartbeat_monitor_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def add_subscriber(self, client_id: str, websocket):
        """Add a new subscriber"""
        async with self._lock:
            self.subscribers[client_id] = websocket
            self.update_heartbeat()
            logger.info(f"Added subscriber {client_id} for {self.symbol}")
    
    async def remove_subscriber(self, client_id: str):
        """Remove a subscriber"""
        async with self._lock:
            if client_id in self.subscribers:
                del self.subscribers[client_id]
                logger.info(f"Removed subscriber {client_id} for {self.symbol}")
    
    def update_heartbeat(self):
        """Update the last heartbeat timestamp"""
        self.last_heartbeat = datetime.datetime.utcnow()
    
    def has_active_subscribers(self) -> bool:
        """Check if there are any active subscribers"""
        return len(self.subscribers) > 0
    
    def is_inactive(self) -> bool:
        """Check if the subscription is inactive (no heartbeat for 20 seconds)"""
        if self.last_heartbeat is None:
            return True
        
        inactive_time = (datetime.datetime.utcnow() - self.last_heartbeat).total_seconds()
        return inactive_time > 20  # Changed from 60 to 20 seconds as requested
    
    async def broadcast_to_subscribers(self, message: Dict[str, Any]):
        """Broadcast a message to all subscribers"""
        if not self.subscribers:
            return
        
        message_str = json.dumps(message)
        disconnected_clients = []
        
        for client_id, websocket in self.subscribers.items():
            try:
                await websocket.send_text(message_str)
            except Exception as e:
                logger.error(f"Error sending to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.remove_subscriber(client_id)
    
    async def send_day_data_to_subscriber(self, client_id: str, websocket):
        """Send day data to a specific subscriber"""
        if self.day_cache is None or self.day_cache.empty:
            error_msg = {
                "type": "error",
                "symbol": self.symbol,
                "message": "No day data available",
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            }
            try:
                await websocket.send_text(json.dumps(error_msg))
            except:
                pass
            return
        
        # Send day data to requesting client only
        day_data = []
        for _, row in self.day_cache.iterrows():
            # Format the data in the desired OHLCV format
            day_data.append({
                "Date": row["Datetime"],
                "Open": float(row["Open"]),
                "High": float(row["High"]),
                "Low": float(row["Low"]),
                "Close": float(row["Close"]),
                "Volume": int(row["Volume"])
            })
        
        day_msg = {
            "type": "day",
            "symbol": self.symbol,
            "data": day_data,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }
        
        try:
            await websocket.send_text(json.dumps(day_msg))
            logger.info(f"Sent day data for {self.symbol} to {client_id} ({len(day_data)} points)")
        except Exception as e:
            logger.error(f"Error sending day data to {client_id}: {e}")
    
    async def start_timers(self):
        """Start the data fetch and heartbeat monitor timers"""
        if not self.timer_running:
            self.timer_running = True
            self.data_fetch_task = asyncio.create_task(self._data_fetch_loop())
            self.heartbeat_monitor_task = asyncio.create_task(self._heartbeat_monitor_loop())
            logger.info(f"Started timers for {self.symbol}")
    
    async def stop_timers(self):
        """Stop the data fetch and heartbeat monitor timers"""
        self.timer_running = False
        if self.data_fetch_task:
            self.data_fetch_task.cancel()
        if self.heartbeat_monitor_task:
            self.heartbeat_monitor_task.cancel()
        logger.info(f"Stopped timers for {self.symbol}")
    
    async def _data_fetch_loop(self):
        """Fetch live data every 60 seconds (1 minute)"""
        while self.timer_running and self.has_active_subscribers():
            try:
                # Fetch 1-minute interval data
                live_data = await fetch_live_data(self.symbol)
                if live_data is not None and not live_data.empty:
                    self.live_cache = live_data
                    
                    # Get the latest data point
                    latest_row = live_data.iloc[-1]
                    
                    # Create and broadcast live data message with OHLCV format
                    message = {
                        "type": "live",
                        "symbol": self.symbol,
                        "data": {
                            "Date": latest_row["Datetime"],
                            "Open": float(latest_row["Open"]),
                            "High": float(latest_row["High"]),
                            "Low": float(latest_row["Low"]),
                            "Close": float(latest_row["Close"]),
                            "Volume": int(latest_row["Volume"])
                        },
                        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
                    }
                    await self.broadcast_to_subscribers(message)
                    logger.info(f"Sent live data for {self.symbol}: OHLCV data")
                else:
                    # Send a message indicating no new data
                    message = {
                        "type": "live",
                        "symbol": self.symbol,
                        "data": None,
                        "message": "No new data available (market closed or no data)",
                        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
                    }
                    await self.broadcast_to_subscribers(message)
                    logger.info(f"No live data available for {self.symbol}")
                
                self.last_fetch_time = datetime.datetime.utcnow()
                await asyncio.sleep(60)  # Wait 60 seconds (1 minute)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in data fetch loop for {self.symbol}: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _heartbeat_monitor_loop(self):
        """Monitor heartbeats and clean up inactive subscriptions"""
        while self.timer_running:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                if self.is_inactive():
                    # Send timeout message to all subscribers
                    message = {
                        "type": "timeout",
                        "symbol": self.symbol,
                        "message": "No heartbeat received for 20 seconds",
                        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
                    }
                    await self.broadcast_to_subscribers(message)
                    
                    # Stop timers and clean up
                    await self.stop_timers()
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat monitor for {self.symbol}: {e}")

# Helper functions
def _now_utc() -> datetime.datetime:
    return datetime.datetime.utcnow()

async def fetch_day_data(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch full day data with 1-minute intervals for IDX"""
    symbol = symbol.strip().upper()
    
    def _sync_download():
        try:
            # Get 1-minute data for the current day
            df = yf.download(tickers=symbol, period="1d", interval="1m", progress=False, auto_adjust=True)
            logger.info(f"Downloaded raw data for {symbol}: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {e}")
            return None
    
    try:
        df = await asyncio.to_thread(_sync_download)
        if df is not None and not df.empty:
            logger.info(f"Processing {len(df)} rows for {symbol}")
            
            # Normalize DataFrame columns (handle multi-index)
            df = normalize_dataframe(df)
            
            # Convert to Jakarta time
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('Asia/Jakarta')
            
            # Filter to trading hours only
            trading_data = df[df.index.map(in_trading_hours)]
            logger.info(f"After trading hours filter: {len(trading_data)} rows")
            
            if trading_data.empty:
                logger.warning(f"No trading data available for {symbol} within IDX trading hours")
                return None
            
            # Reset index to make Datetime a column
            trading_data = trading_data.reset_index()
            
            # Rename the time column
            trading_data = trading_data.rename(columns={'index': 'Datetime'})
            
            # Convert Datetime to ISO format for JSON serialization
            trading_data['Datetime'] = trading_data['Datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S%z')
            
            logger.info(f"Successfully processed day data for {symbol}: {len(trading_data)} rows")
            return trading_data
        else:
            logger.warning(f"No data returned for {symbol}")
            return None
    except Exception as e:
        logger.error(f"Error fetching day data for {symbol}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

async def fetch_live_data(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch recent 1-minute interval data for IDX"""
    symbol = symbol.strip().upper()
    
    def _sync_download():
        try:
            # Get 1-minute data for the current day
            df = yf.download(tickers=symbol, period="1d", interval="1m", progress=False, auto_adjust=True)
            logger.info(f"Downloaded live data for {symbol}: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error downloading live data for {symbol}: {e}")
            return None
    
    try:
        df = await asyncio.to_thread(_sync_download)
        if df is not None and not df.empty:
            # Normalize DataFrame columns (handle multi-index)
            df = normalize_dataframe(df)
            
            # Convert to Jakarta time
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('Asia/Jakarta')
            
            # Filter to trading hours only
            trading_data = df[df.index.map(in_trading_hours)]
            
            if trading_data.empty:
                logger.warning(f"No live trading data available for {symbol} within IDX trading hours")
                return None
            
            # Reset index to make Datetime a column
            trading_data = trading_data.reset_index()
            
            # Rename the time column
            trading_data = trading_data.rename(columns={'index': 'Datetime'})
            
            # Convert Datetime to ISO format for JSON serialization
            trading_data['Datetime'] = trading_data['Datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S%z')
            
            return trading_data
        else:
            logger.warning(f"No live data returned for {symbol}")
            return None
    except Exception as e:
        logger.error(f"Error fetching live data for {symbol}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

async def handle_client_message(message: Dict[str, Any], client_id: str, websocket):
    """Process incoming client messages"""
    try:
        logger.info(f"Processing message from {client_id}: {message}")
        
        if "req" in message:
            # Initial request for day data
            symbol = message.get("req", "").strip().upper()
            if not symbol:
                error_msg = {
                    "type": "error",
                    "symbol": "",
                    "message": "Invalid symbol in request",
                    "timestamp": _now_utc().isoformat() + "Z"
                }
                await websocket.send_text(json.dumps(error_msg))
                return
            
            logger.info(f"Processing request for symbol: {symbol}")
            
            # Get or create subscription
            subscription = await subscription_manager.get_subscription(symbol)
            await subscription.add_subscriber(client_id, websocket)
            
            # Check if we already have day data cached
            if subscription.day_cache is None:
                logger.info(f"Fetching day data for {symbol} (first request)")
                subscription.day_cache = await fetch_day_data(symbol)
            
            # Send day data to the requesting client
            await subscription.send_day_data_to_subscriber(client_id, websocket)
            
            # Start timers if not already running
            await subscription.start_timers()
        
        elif "beat" in message:
            # Heartbeat message
            symbol = message.get("beat", "").strip().upper()
            if not symbol:
                return
            
            # Get subscription and update heartbeat
            subscription = await subscription_manager.get_subscription(symbol)
            await subscription.add_subscriber(client_id, websocket)
            subscription.update_heartbeat()
            
            # If this is a new subscriber and we have day data cached, send it
            if subscription.day_cache is not None and not subscription.day_cache.empty:
                await subscription.send_day_data_to_subscriber(client_id, websocket)
            
            # Start timers if not already running
            await subscription.start_timers()
        
        else:
            # Unknown message type
            error_msg = {
                "type": "error",
                "symbol": "",
                "message": "Unknown message type",
                "timestamp": _now_utc().isoformat() + "Z"
            }
            await websocket.send_text(json.dumps(error_msg))
    
    except Exception as e:
        logger.error(f"Error handling client message: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        error_msg = {
            "type": "error",
            "symbol": message.get("symbol", ""),
            "message": f"Server error processing message: {str(e)}",
            "timestamp": _now_utc().isoformat() + "Z"
        }
        try:
            await websocket.send_text(json.dumps(error_msg))
        except:
            pass  # Client might be disconnected

async def remove_client_from_all_subscriptions(client_id: str):
    """Remove client from all subscriptions they're part of"""
    async with subscription_manager._lock:
        for symbol, sub in subscription_manager.active_stocks.items():
            if client_id in sub.subscribers:
                await sub.remove_subscriber(client_id)
                logger.info(f"Removed client {client_id} from {symbol}")

async def cleanup_inactive_subscriptions():
    """Periodically clean up inactive subscriptions"""
    while True:
        try:
            await subscription_manager.cleanup_inactive_subscriptions()
            await asyncio.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)

def start_cleanup_task():
    """Start the cleanup task when the event loop is running"""
    global cleanup_task
    cleanup_task = asyncio.create_task(cleanup_inactive_subscriptions())
    return cleanup_task

# Initialize cleanup_task as None
cleanup_task = None
