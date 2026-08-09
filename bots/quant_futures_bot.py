"""
Quant Futures Trading Bot

Multi-strategy crypto futures engine using CCXT.
Supports Binance Futures, Bybit, OKX with live, paper, and backtest modes.

Setup:
  1. pip install ccxt pandas numpy python-dotenv
  2. Copy .env.futures.example to .env.futures and fill in credentials
  3. python quant_futures_bot.py --mode live

Modes:
  --mode live      Live trading on mainnet
  --mode paper     Paper trading on testnet
  --mode backtest  Historical backtesting
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_FILE = Path(__file__).parent / ".env.futures"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("quant_futures_bot.log"),
    ],
)
log = logging.getLogger("quant_futures")


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    timestamp: datetime
    symbol: str
    momentum_score: int = 0
    mean_reversion_score: int = 0
    funding_score: int = 0

    @property
    def composite_score(self) -> int:
        return self.momentum_score + self.mean_reversion_score + self.funding_score

    @property
    def confidence(self) -> int:
        directions = []
        for s in [self.momentum_score, self.mean_reversion_score, self.funding_score]:
            if s > 0:
                directions.append(1)
            elif s < 0:
                directions.append(-1)
            else:
                directions.append(0)
        if all(d == directions[0] for d in directions) and directions[0] != 0:
            return 3
        agreeing = max(
            sum(1 for d in directions if d == 1),
            sum(1 for d in directions if d == -1),
        )
        return agreeing

    @property
    def action(self) -> str:
        score = self.composite_score
        if score >= 3:
            return "long"
        elif score <= -3:
            return "short"
        elif score <= 0 and any(s > 0 for s in [self.momentum_score, self.mean_reversion_score, self.funding_score]):
            return "exit_long"
        elif score >= 0 and any(s < 0 for s in [self.momentum_score, self.mean_reversion_score, self.funding_score]):
            return "exit_short"
        return "hold"


@dataclass
class Position:
    symbol: str
    side: Side
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    exchange_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None

    @property
    def notional(self) -> float:
        return self.entry_price * self.quantity


@dataclass
class Trade:
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    fees: float = 0.0

    @property
    def duration(self) -> timedelta:
        return self.exit_time - self.entry_time


@dataclass
class PerformanceMetrics:
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_hours: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    total_trades: int = 0
    avg_trade_duration_hours: float = 0.0
    monthly_returns: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ExchangeClient
# ---------------------------------------------------------------------------

class ExchangeClient:
    EXCHANGE_MAP = {
        "binance": ccxt.binanceusdm,
        "bybit": ccxt.bybit,
        "okx": ccxt.okx,
    }

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        testnet: bool = True,
    ):
        cls = self.EXCHANGE_MAP.get(exchange_id)
        if cls is None:
            raise ValueError(f"Unsupported exchange: {exchange_id}. Use: {list(self.EXCHANGE_MAP)}")

        config: dict = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }
        if passphrase:
            config["password"] = passphrase

        self.exchange: ccxt.Exchange = cls(config)

        if testnet:
            self.exchange.set_sandbox_mode(True)
            log.info("Exchange %s initialized in TESTNET mode", exchange_id)
        else:
            log.info("Exchange %s initialized in MAINNET mode", exchange_id)

        self.exchange_id = exchange_id

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 200, since: Optional[int] = None
    ) -> pd.DataFrame:
        raw = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    def fetch_ticker(self, symbol: str) -> dict:
        return self.exchange.fetch_ticker(symbol)

    def fetch_funding_rate(self, symbol: str) -> float:
        try:
            if hasattr(self.exchange, "fetch_funding_rate"):
                data = self.exchange.fetch_funding_rate(symbol)
                return float(data.get("fundingRate", 0.0))
            ticker = self.exchange.fetch_ticker(symbol)
            info = ticker.get("info", {})
            for key in ["fundingRate", "funding_rate", "lastFundingRate"]:
                if key in info:
                    return float(info[key])
        except Exception as e:
            log.warning("Failed to fetch funding rate for %s: %s", symbol, e)
        return 0.0

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[dict] = None,
    ) -> dict:
        params = params or {}
        try:
            order = self.exchange.create_order(symbol, order_type, side, amount, price, params)
            log.info(
                "Order placed: %s %s %s qty=%.6f price=%s id=%s",
                order_type, side, symbol, amount, price, order.get("id"),
            )
            return order
        except ccxt.BaseError as e:
            log.error("Order failed: %s %s %s — %s", order_type, side, symbol, e)
            raise

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        try:
            result = self.exchange.cancel_order(order_id, symbol)
            log.info("Cancelled order %s on %s", order_id, symbol)
            return result
        except ccxt.BaseError as e:
            log.warning("Cancel failed for %s: %s", order_id, e)
            raise

    def get_position(self, symbol: str) -> Optional[dict]:
        try:
            positions = self.exchange.fetch_positions([symbol])
            for pos in positions:
                contracts = float(pos.get("contracts", 0) or 0)
                if contracts > 0:
                    return pos
        except ccxt.BaseError as e:
            log.warning("Failed to fetch position for %s: %s", symbol, e)
        return None

    def get_balance(self) -> dict:
        balance = self.exchange.fetch_balance()
        usdt = balance.get("USDT", balance.get("total", {}))
        return {
            "total": float(usdt.get("total", 0) if isinstance(usdt, dict) else 0),
            "free": float(usdt.get("free", 0) if isinstance(usdt, dict) else 0),
            "used": float(usdt.get("used", 0) if isinstance(usdt, dict) else 0),
        }

    def set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            self.exchange.set_leverage(leverage, symbol)
            log.info("Leverage set to %dx for %s", leverage, symbol)
        except ccxt.BaseError as e:
            log.warning("Set leverage failed for %s: %s", symbol, e)

    def fetch_ohlcv_since(
        self, symbol: str, timeframe: str, start_ms: int, end_ms: int
    ) -> pd.DataFrame:
        all_candles = []
        since = start_ms
        tf_ms = self._timeframe_to_ms(timeframe)

        while since < end_ms:
            batch = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not batch:
                break
            all_candles.extend(batch)
            last_ts = batch[-1][0]
            if last_ts >= end_ms:
                break
            since = last_ts + tf_ms
            time.sleep(self.exchange.rateLimit / 1000)

        if not all_candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df[df.index <= pd.Timestamp(end_ms, unit="ms", tz="UTC")]
        df = df[~df.index.duplicated(keep="first")]
        return df

    @staticmethod
    def _timeframe_to_ms(tf: str) -> int:
        multipliers = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
        unit = tf[-1]
        value = int(tf[:-1])
        return value * multipliers.get(unit, 60_000)


# ---------------------------------------------------------------------------
# DataEngine
# ---------------------------------------------------------------------------

class DataEngine:
    @staticmethod
    def compute_indicators(df: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
        p = {
            "ema_fast": 12,
            "ema_slow": 26,
            "sma_period": 50,
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "adx_period": 14,
            "atr_period": 14,
            "vol_sma_period": 20,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
        }
        if params:
            p.update(params)

        c = df["close"]
        h = df["high"]
        l = df["low"]

        df["ema_fast"] = c.ewm(span=p["ema_fast"], adjust=False).mean()
        df["ema_slow"] = c.ewm(span=p["ema_slow"], adjust=False).mean()
        df["sma"] = c.rolling(p["sma_period"]).mean()

        bb_sma = c.rolling(p["bb_period"]).mean()
        bb_std = c.rolling(p["bb_period"]).std()
        df["bb_upper"] = bb_sma + p["bb_std"] * bb_std
        df["bb_mid"] = bb_sma
        df["bb_lower"] = bb_sma - p["bb_std"] * bb_std
        df["bb_zscore"] = (c - bb_sma) / bb_std.replace(0, np.nan)

        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / p["rsi_period"], min_periods=p["rsi_period"], adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / p["rsi_period"], min_periods=p["rsi_period"], adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=p["atr_period"], adjust=False).mean()

        plus_dm = (h - h.shift(1)).clip(lower=0)
        minus_dm = (l.shift(1) - l).clip(lower=0)
        mask_plus = plus_dm > minus_dm
        mask_minus = minus_dm > plus_dm
        plus_dm = plus_dm.where(mask_plus, 0)
        minus_dm = minus_dm.where(mask_minus, 0)

        atr_smooth = tr.ewm(span=p["adx_period"], adjust=False).mean()
        df["plus_di"] = 100 * plus_dm.ewm(span=p["adx_period"], adjust=False).mean() / atr_smooth.replace(0, np.nan)
        df["minus_di"] = 100 * minus_dm.ewm(span=p["adx_period"], adjust=False).mean() / atr_smooth.replace(0, np.nan)
        di_sum = df["plus_di"] + df["minus_di"]
        dx = 100 * (df["plus_di"] - df["minus_di"]).abs() / di_sum.replace(0, np.nan)
        df["adx"] = dx.ewm(span=p["adx_period"], adjust=False).mean()

        df["vol_sma"] = df["volume"].rolling(p["vol_sma_period"]).mean()

        ema_fast_macd = c.ewm(span=p["macd_fast"], adjust=False).mean()
        ema_slow_macd = c.ewm(span=p["macd_slow"], adjust=False).mean()
        df["macd"] = ema_fast_macd - ema_slow_macd
        df["macd_signal"] = df["macd"].ewm(span=p["macd_signal"], adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        sma_for_z = c.rolling(p["sma_period"]).mean()
        sma_std = c.rolling(p["sma_period"]).std()
        df["price_zscore"] = (c - sma_for_z) / sma_std.replace(0, np.nan)

        return df


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class MomentumStrategy:
    def score(self, df: pd.DataFrame) -> int:
        if len(df) < 2:
            return 0

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        ema_cross_up = prev["ema_fast"] <= prev["ema_slow"] and latest["ema_fast"] > latest["ema_slow"]
        ema_cross_down = prev["ema_fast"] >= prev["ema_slow"] and latest["ema_fast"] < latest["ema_slow"]
        ema_bullish = latest["ema_fast"] > latest["ema_slow"]
        ema_bearish = latest["ema_fast"] < latest["ema_slow"]

        adx_ok = latest.get("adx", 0) > 20
        rsi = latest.get("rsi", 50)
        vol_ok = latest["volume"] > 1.5 * latest.get("vol_sma", latest["volume"])
        macd_bull = latest.get("macd_hist", 0) > 0
        macd_bear = latest.get("macd_hist", 0) < 0

        if ema_cross_up and adx_ok and 50 <= rsi <= 80 and vol_ok:
            return 2
        if ema_bullish and adx_ok and 50 <= rsi <= 80 and macd_bull:
            return 1
        if ema_cross_down and adx_ok and 20 <= rsi <= 50 and vol_ok:
            return -2
        if ema_bearish and adx_ok and 20 <= rsi <= 50 and macd_bear:
            return -1
        return 0


class MeanReversionStrategy:
    def score(self, df: pd.DataFrame) -> int:
        if len(df) < 6:
            return 0

        latest = df.iloc[-1]
        z = latest.get("bb_zscore", 0)
        rsi = latest.get("rsi", 50)

        recent_extreme_low = any(df["bb_zscore"].iloc[-6:-1] < -2.0)
        recent_extreme_high = any(df["bb_zscore"].iloc[-6:-1] > 2.0)

        if z < -2.0 and rsi < 30 and recent_extreme_low:
            return 2
        if z < -1.5 and rsi < 35:
            return 1
        if z > 2.0 and rsi > 70 and recent_extreme_high:
            return -2
        if z > 1.5 and rsi > 65:
            return -1
        return 0


class FundingRateStrategy:
    def score(self, funding_rate: float) -> int:
        if funding_rate > 0.001:
            return -2
        elif funding_rate > 0.0005:
            return -1
        elif funding_rate < -0.001:
            return 2
        elif funding_rate < -0.0005:
            return 1
        return 0


# ---------------------------------------------------------------------------
# CompositeSignal
# ---------------------------------------------------------------------------

class CompositeSignal:
    LONG_THRESHOLD = 3
    SHORT_THRESHOLD = -3

    def __init__(self):
        self.momentum = MomentumStrategy()
        self.mean_reversion = MeanReversionStrategy()
        self.funding = FundingRateStrategy()

    def generate(self, df: pd.DataFrame, funding_rate: float, symbol: str) -> Signal:
        now = datetime.now(timezone.utc)
        sig = Signal(
            timestamp=now,
            symbol=symbol,
            momentum_score=self.momentum.score(df),
            mean_reversion_score=self.mean_reversion.score(df),
            funding_score=self.funding.score(funding_rate),
        )
        log.debug(
            "Signal %s: mom=%+d mr=%+d fund=%+d => composite=%+d confidence=%d action=%s",
            symbol, sig.momentum_score, sig.mean_reversion_score,
            sig.funding_score, sig.composite_score, sig.confidence, sig.action,
        )
        return sig

    def should_enter_long(self, sig: Signal) -> bool:
        return sig.composite_score >= self.LONG_THRESHOLD

    def should_enter_short(self, sig: Signal) -> bool:
        return sig.composite_score <= self.SHORT_THRESHOLD

    def should_exit_long(self, sig: Signal) -> bool:
        return sig.composite_score <= 0

    def should_exit_short(self, sig: Signal) -> bool:
        return sig.composite_score >= 0


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------

class RiskManager:
    def __init__(
        self,
        max_position_pct: float = 5.0,
        max_drawdown_pct: float = 10.0,
        max_daily_loss_pct: float = 5.0,
        max_concurrent_positions: int = 3,
        max_total_exposure_pct: float = 20.0,
        max_daily_trades: int = 50,
        default_leverage: int = 3,
        max_leverage: int = 10,
        kelly_lookback: int = 50,
        cooldown_seconds: int = 300,
    ):
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_concurrent = max_concurrent_positions
        self.max_total_exposure_pct = max_total_exposure_pct
        self.max_daily_trades = max_daily_trades
        self.default_leverage = default_leverage
        self.max_leverage = max_leverage
        self.kelly_lookback = kelly_lookback
        self.cooldown_seconds = cooldown_seconds

        self.equity_peak: float = 0.0
        self.daily_start_equity: float = 0.0
        self.daily_trade_count: int = 0
        self.last_daily_reset: Optional[datetime] = None
        self.circuit_breaker_active: bool = False
        self.trade_history: list[Trade] = []
        self.cooldowns: dict[str, datetime] = {}

    def update_equity_peak(self, equity: float) -> None:
        if equity > self.equity_peak:
            self.equity_peak = equity

    def reset_daily_if_needed(self, equity: float) -> None:
        now = datetime.now(timezone.utc)
        if self.last_daily_reset is None or now.date() != self.last_daily_reset.date():
            self.daily_start_equity = equity
            self.daily_trade_count = 0
            self.last_daily_reset = now
            log.info("Daily counters reset. Starting equity: %.2f", equity)

    def check_circuit_breaker(self, equity: float) -> bool:
        if self.equity_peak <= 0:
            return False

        drawdown = 1.0 - (equity / self.equity_peak)

        if self.circuit_breaker_active:
            recovery_threshold = self.equity_peak * (1.0 - self.max_drawdown_pct / 100.0 * 0.5)
            if equity >= recovery_threshold:
                self.circuit_breaker_active = False
                log.info("Circuit breaker RELEASED. Equity: %.2f", equity)
                return False
            return True

        if drawdown >= self.max_drawdown_pct / 100.0:
            self.circuit_breaker_active = True
            log.warning(
                "Circuit breaker ACTIVATED. Drawdown: %.2f%% Equity: %.2f Peak: %.2f",
                drawdown * 100, equity, self.equity_peak,
            )
            return True

        return False

    def check_daily_loss_limit(self, equity: float) -> bool:
        if self.daily_start_equity <= 0:
            return False
        daily_loss = 1.0 - (equity / self.daily_start_equity)
        if daily_loss >= self.max_daily_loss_pct / 100.0:
            log.warning("Daily loss limit hit: %.2f%%", daily_loss * 100)
            return True
        return False

    def check_daily_trade_limit(self) -> bool:
        if self.daily_trade_count >= self.max_daily_trades:
            log.warning("Daily trade limit reached: %d", self.daily_trade_count)
            return True
        return False

    def check_cooldown(self, symbol: str) -> bool:
        if symbol in self.cooldowns:
            if datetime.now(timezone.utc) < self.cooldowns[symbol]:
                return True
        return False

    def set_cooldown(self, symbol: str) -> None:
        self.cooldowns[symbol] = datetime.now(timezone.utc) + timedelta(seconds=self.cooldown_seconds)

    def compute_kelly_fraction(self) -> float:
        recent = self.trade_history[-self.kelly_lookback:]
        if len(recent) < 10:
            return 0.02

        wins = [t for t in recent if t.pnl > 0]
        losses = [t for t in recent if t.pnl <= 0]

        if not losses:
            return 0.10
        if not wins:
            return 0.01

        win_rate = len(wins) / len(recent)
        avg_win = sum(t.pnl_pct for t in wins) / len(wins)
        avg_loss = abs(sum(t.pnl_pct for t in losses) / len(losses))

        if avg_loss == 0:
            return 0.02

        kelly = win_rate - (1.0 - win_rate) / (avg_win / avg_loss)
        half_kelly = kelly * 0.5
        return max(0.01, min(0.10, half_kelly))

    def compute_position_size(
        self,
        equity: float,
        price: float,
        atr: float,
        existing_positions: list[Position],
    ) -> float:
        if len(existing_positions) >= self.max_concurrent:
            log.info("Max concurrent positions reached (%d)", self.max_concurrent)
            return 0.0

        current_exposure = sum(p.notional for p in existing_positions)
        max_total = equity * self.max_total_exposure_pct / 100.0
        remaining = max_total - current_exposure
        if remaining <= 0:
            log.info("Max total exposure reached")
            return 0.0

        kelly_frac = self.compute_kelly_fraction()
        position_pct = min(kelly_frac, self.max_position_pct / 100.0)
        position_value = equity * position_pct

        position_value = min(position_value, remaining)

        quantity = position_value / price
        if quantity <= 0:
            return 0.0

        log.info(
            "Position sizing: kelly=%.3f pct=%.3f value=%.2f qty=%.6f",
            kelly_frac, position_pct, position_value, quantity,
        )
        return quantity

    def effective_leverage(self, equity: float) -> int:
        if self.equity_peak <= 0:
            return self.default_leverage

        drawdown = 1.0 - (equity / self.equity_peak)
        if drawdown > 0.05:
            reduced = max(1, self.default_leverage - 1)
            log.info("Leverage reduced to %dx due to %.1f%% drawdown", reduced, drawdown * 100)
            return reduced
        return self.default_leverage

    def can_trade(self, equity: float, symbol: str, positions: list[Position]) -> tuple[bool, str]:
        self.reset_daily_if_needed(equity)
        self.update_equity_peak(equity)

        if self.check_circuit_breaker(equity):
            return False, "circuit_breaker"
        if self.check_daily_loss_limit(equity):
            return False, "daily_loss_limit"
        if self.check_daily_trade_limit():
            return False, "daily_trade_limit"
        if self.check_cooldown(symbol):
            return False, "cooldown"
        if len(positions) >= self.max_concurrent:
            return False, "max_positions"
        return True, "ok"

    def record_trade(self, trade: Trade) -> None:
        self.trade_history.append(trade)
        self.daily_trade_count += 1


# ---------------------------------------------------------------------------
# TradeExecutor
# ---------------------------------------------------------------------------

class TradeExecutor:
    def __init__(
        self,
        client: ExchangeClient,
        sl_atr_mult: float = 2.0,
        tp_atr_mult: float = 4.0,
        trailing_atr_mult: float = 2.5,
        limit_timeout_seconds: int = 30,
        use_trailing_stop: bool = True,
    ):
        self.client = client
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.trailing_atr_mult = trailing_atr_mult
        self.limit_timeout = limit_timeout_seconds
        self.use_trailing_stop = use_trailing_stop

    def open_position(
        self, symbol: str, side: Side, quantity: float, price: float, atr: float, leverage: int
    ) -> Optional[Position]:
        try:
            self.client.set_leverage(symbol, leverage)
        except Exception:
            pass

        order = None
        try:
            order = self.client.create_order(symbol, "limit", "buy" if side == Side.LONG else "sell", quantity, price)
            filled = self._wait_for_fill(order, symbol)
            if not filled:
                log.info("Limit order not filled, falling back to market")
                self.client.cancel_order(order["id"], symbol)
                order = self.client.create_order(symbol, "market", "buy" if side == Side.LONG else "sell", quantity)
        except ccxt.BaseError:
            order = self.client.create_order(symbol, "market", "buy" if side == Side.LONG else "sell", quantity)

        if not order:
            return None

        fill_price = float(order.get("average", order.get("price", price)) or price)
        fill_qty = float(order.get("filled", quantity) or quantity)

        if fill_qty <= 0:
            return None

        if side == Side.LONG:
            sl = fill_price - atr * self.sl_atr_mult
            tp = fill_price + atr * self.tp_atr_mult
            trail = fill_price - atr * self.trailing_atr_mult if self.use_trailing_stop else None
        else:
            sl = fill_price + atr * self.sl_atr_mult
            tp = fill_price - atr * self.tp_atr_mult
            trail = fill_price + atr * self.trailing_atr_mult if self.use_trailing_stop else None

        sl_order_id = self._place_stop_loss(symbol, side, fill_qty, sl)
        tp_order_id = self._place_take_profit(symbol, side, fill_qty, tp)

        pos = Position(
            symbol=symbol,
            side=side,
            entry_price=fill_price,
            quantity=fill_qty,
            entry_time=datetime.now(timezone.utc),
            stop_loss=sl,
            take_profit=tp,
            trailing_stop=self.trailing_atr_mult * atr if self.use_trailing_stop else None,
            trailing_stop_price=trail,
            exchange_order_id=order.get("id"),
            sl_order_id=sl_order_id,
            tp_order_id=tp_order_id,
        )

        log.info(
            "Opened %s %s @ %.4f qty=%.6f SL=%.4f TP=%.4f",
            side.value, symbol, fill_price, fill_qty, sl, tp,
        )
        return pos

    def close_position(self, pos: Position, current_price: float) -> Optional[Trade]:
        close_side = "sell" if pos.side == Side.LONG else "buy"
        try:
            for oid in [pos.sl_order_id, pos.tp_order_id]:
                if oid:
                    try:
                        self.client.cancel_order(oid, pos.symbol)
                    except Exception:
                        pass

            order = self.client.create_order(pos.symbol, "market", close_side, pos.quantity)
            exit_price = float(order.get("average", order.get("price", current_price)) or current_price)
        except ccxt.BaseError as e:
            log.error("Failed to close %s %s: %s", pos.side.value, pos.symbol, e)
            return None

        if pos.side == Side.LONG:
            pnl = (exit_price - pos.entry_price) * pos.quantity
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        trade = Trade(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            entry_time=pos.entry_time,
            exit_time=datetime.now(timezone.utc),
            pnl=pnl,
            pnl_pct=pnl_pct,
        )
        log.info(
            "Closed %s %s @ %.4f PnL=%.4f (%.2f%%)",
            pos.side.value, pos.symbol, exit_price, pnl, pnl_pct * 100,
        )
        return trade

    def update_trailing_stop(self, pos: Position, current_price: float) -> Position:
        if not self.use_trailing_stop or pos.trailing_stop is None:
            return pos

        if pos.side == Side.LONG:
            new_trail = current_price - pos.trailing_stop
            if pos.trailing_stop_price is None or new_trail > pos.trailing_stop_price:
                pos.trailing_stop_price = new_trail
        else:
            new_trail = current_price + pos.trailing_stop
            if pos.trailing_stop_price is None or new_trail < pos.trailing_stop_price:
                pos.trailing_stop_price = new_trail

        return pos

    def check_stop_hit(self, pos: Position, current_price: float) -> bool:
        if pos.side == Side.LONG:
            if current_price <= pos.stop_loss:
                return True
            if pos.trailing_stop_price and current_price <= pos.trailing_stop_price:
                return True
        else:
            if current_price >= pos.stop_loss:
                return True
            if pos.trailing_stop_price and current_price >= pos.trailing_stop_price:
                return True
        return False

    def check_tp_hit(self, pos: Position, current_price: float) -> bool:
        if pos.side == Side.LONG:
            return current_price >= pos.take_profit
        return current_price <= pos.take_profit

    def _wait_for_fill(self, order: dict, symbol: str) -> bool:
        deadline = time.time() + self.limit_timeout
        order_id = order.get("id")
        while time.time() < deadline:
            try:
                fetched = self.client.exchange.fetch_order(order_id, symbol)
                if fetched.get("status") == "closed":
                    return True
                if fetched.get("status") == "canceled":
                    return False
            except Exception:
                pass
            time.sleep(2)
        return False

    def _place_stop_loss(self, symbol: str, side: Side, qty: float, price: float) -> Optional[str]:
        close_side = "sell" if side == Side.LONG else "buy"
        try:
            order = self.client.create_order(
                symbol, "stop", close_side, qty, price,
                params={"stopPrice": price, "reduceOnly": True},
            )
            return order.get("id")
        except Exception as e:
            log.warning("Failed to place SL for %s: %s", symbol, e)
            return None

    def _place_take_profit(self, symbol: str, side: Side, qty: float, price: float) -> Optional[str]:
        close_side = "sell" if side == Side.LONG else "buy"
        try:
            order = self.client.create_order(
                symbol, "limit", close_side, qty, price,
                params={"reduceOnly": True},
            )
            return order.get("id")
        except Exception as e:
            log.warning("Failed to place TP for %s: %s", symbol, e)
            return None


# ---------------------------------------------------------------------------
# PerformanceTracker
# ---------------------------------------------------------------------------

class PerformanceTracker:
    def __init__(self, log_file: str = "performance_log.json"):
        self.log_file = Path(log_file)
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.initial_equity: float = 0.0

    def record_trade(self, trade: Trade) -> None:
        self.trades.append(trade)

    def record_equity(self, equity: float) -> None:
        self.equity_curve.append((datetime.now(timezone.utc), equity))

    def compute_metrics(self) -> PerformanceMetrics:
        m = PerformanceMetrics()

        if not self.trades:
            return m

        m.total_trades = len(self.trades)
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        m.win_rate = len(wins) / m.total_trades if m.total_trades > 0 else 0.0

        gross_profit = sum(t.pnl for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
        m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

        m.avg_win_pct = (sum(t.pnl_pct for t in wins) / len(wins) * 100) if wins else 0.0
        m.avg_loss_pct = (sum(t.pnl_pct for t in losses) / len(losses) * 100) if losses else 0.0

        durations = [t.duration.total_seconds() / 3600 for t in self.trades]
        m.avg_trade_duration_hours = sum(durations) / len(durations) if durations else 0.0

        if self.equity_curve and len(self.equity_curve) > 1:
            equities = [e for _, e in self.equity_curve]
            m.total_return_pct = (equities[-1] / equities[0] - 1.0) * 100 if equities[0] > 0 else 0.0

            first_time = self.equity_curve[0][0]
            last_time = self.equity_curve[-1][0]
            days = max((last_time - first_time).total_seconds() / 86400, 1)
            years = days / 365.25
            if years > 0 and equities[0] > 0:
                m.annual_return_pct = ((equities[-1] / equities[0]) ** (1 / years) - 1.0) * 100

            returns = pd.Series(equities).pct_change().dropna()
            if len(returns) > 1 and returns.std() > 0:
                m.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(365 * 24)
                downside = returns[returns < 0]
                if len(downside) > 0 and downside.std() > 0:
                    m.sortino_ratio = (returns.mean() / downside.std()) * np.sqrt(365 * 24)

            peak = equities[0]
            max_dd = 0.0
            dd_start = 0
            max_dd_duration = 0
            current_dd_start = 0
            for i, eq in enumerate(equities):
                if eq > peak:
                    duration = i - current_dd_start
                    if duration > max_dd_duration and max_dd > 0:
                        max_dd_duration = duration
                    peak = eq
                    current_dd_start = i
                dd = 1.0 - eq / peak
                if dd > max_dd:
                    max_dd = dd
                    dd_start = current_dd_start
            m.max_drawdown_pct = max_dd * 100

            if len(self.equity_curve) > 1:
                avg_interval = (last_time - first_time).total_seconds() / len(self.equity_curve) / 3600
                m.max_drawdown_duration_hours = max_dd_duration * avg_interval

        monthly = {}
        for t in self.trades:
            key = t.exit_time.strftime("%Y-%m")
            monthly[key] = monthly.get(key, 0.0) + t.pnl
        m.monthly_returns = monthly

        return m

    def rolling_sharpe(self, window_days: int = 30) -> float:
        if len(self.equity_curve) < 2:
            return 0.0

        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        recent = [(t, e) for t, e in self.equity_curve if t >= cutoff]

        if len(recent) < 2:
            return 0.0

        equities = [e for _, e in recent]
        returns = pd.Series(equities).pct_change().dropna()
        if len(returns) < 2 or returns.std() == 0:
            return 0.0

        return float((returns.mean() / returns.std()) * np.sqrt(365 * 24))

    def save(self) -> None:
        data = {
            "trades": [
                {
                    "symbol": t.symbol,
                    "side": t.side.value,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                }
                for t in self.trades
            ],
            "equity_curve": [
                {"time": t.isoformat(), "equity": e}
                for t, e in self.equity_curve
            ],
            "metrics": asdict(self.compute_metrics()),
        }
        self.log_file.write_text(json.dumps(data, indent=2, default=str))
        log.info("Performance data saved to %s", self.log_file)


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    def __init__(
        self,
        initial_equity: float = 10000.0,
        fee_rate: float = 0.0006,
        slippage_pct: float = 0.0005,
        sl_atr_mult: float = 2.0,
        tp_atr_mult: float = 4.0,
        max_positions: int = 3,
        max_position_pct: float = 5.0,
        leverage: int = 3,
    ):
        self.initial_equity = initial_equity
        self.fee_rate = fee_rate
        self.slippage_pct = slippage_pct
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct
        self.leverage = leverage

    def run(self, df: pd.DataFrame, symbol: str) -> PerformanceMetrics:
        df = DataEngine.compute_indicators(df.copy())
        df = df.dropna(subset=["ema_fast", "ema_slow", "rsi", "adx", "atr", "bb_zscore"])

        if df.empty:
            log.warning("No data after indicator computation")
            return PerformanceMetrics()

        composite = CompositeSignal()
        risk = RiskManager(
            max_position_pct=self.max_position_pct,
            max_concurrent_positions=self.max_positions,
            default_leverage=self.leverage,
        )
        tracker = PerformanceTracker()

        equity = self.initial_equity
        tracker.initial_equity = equity
        positions: list[Position] = []

        log.info("Backtest starting: %d bars, initial equity: %.2f", len(df), equity)

        for i in range(50, len(df)):
            window = df.iloc[max(0, i - 100) : i + 1]
            bar = df.iloc[i]
            bar_time = df.index[i]
            price = float(bar["close"])
            atr = float(bar["atr"])
            high = float(bar["high"])
            low = float(bar["low"])

            risk.update_equity_peak(equity)
            tracker.record_equity(equity)

            closed = []
            for pos in positions:
                hit_sl = False
                hit_tp = False

                if pos.side == Side.LONG:
                    hit_sl = low <= pos.stop_loss
                    hit_tp = high >= pos.take_profit
                else:
                    hit_sl = high >= pos.stop_loss
                    hit_tp = low <= pos.take_profit

                if hit_sl or hit_tp:
                    if hit_sl:
                        exit_price = pos.stop_loss
                    else:
                        exit_price = pos.take_profit

                    slippage = exit_price * self.slippage_pct
                    if pos.side == Side.LONG:
                        exit_price -= slippage if hit_sl else -slippage
                    else:
                        exit_price += slippage if hit_sl else -slippage

                    if pos.side == Side.LONG:
                        pnl_raw = (exit_price - pos.entry_price) * pos.quantity
                    else:
                        pnl_raw = (pos.entry_price - exit_price) * pos.quantity

                    fees = pos.entry_price * pos.quantity * self.fee_rate + exit_price * pos.quantity * self.fee_rate
                    pnl = pnl_raw - fees

                    trade = Trade(
                        symbol=symbol,
                        side=pos.side,
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        quantity=pos.quantity,
                        entry_time=pos.entry_time,
                        exit_time=bar_time if isinstance(bar_time, datetime) else bar_time.to_pydatetime(),
                        pnl=pnl,
                        pnl_pct=pnl / (pos.entry_price * pos.quantity) if pos.entry_price * pos.quantity > 0 else 0,
                        fees=fees,
                    )
                    tracker.record_trade(trade)
                    risk.record_trade(trade)
                    equity += pnl
                    closed.append(pos)

            positions = [p for p in positions if p not in closed]

            sig = Signal(
                timestamp=bar_time if isinstance(bar_time, datetime) else bar_time.to_pydatetime(),
                symbol=symbol,
                momentum_score=composite.momentum.score(window),
                mean_reversion_score=composite.mean_reversion.score(window),
                funding_score=0,
            )

            for pos in list(positions):
                should_exit = False
                if pos.side == Side.LONG and composite.should_exit_long(sig):
                    should_exit = True
                elif pos.side == Side.SHORT and composite.should_exit_short(sig):
                    should_exit = True

                if should_exit:
                    exit_price = price * (1 - self.slippage_pct if pos.side == Side.LONG else 1 + self.slippage_pct)
                    if pos.side == Side.LONG:
                        pnl_raw = (exit_price - pos.entry_price) * pos.quantity
                    else:
                        pnl_raw = (pos.entry_price - exit_price) * pos.quantity

                    fees = pos.entry_price * pos.quantity * self.fee_rate + exit_price * pos.quantity * self.fee_rate
                    pnl = pnl_raw - fees

                    trade = Trade(
                        symbol=symbol,
                        side=pos.side,
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        quantity=pos.quantity,
                        entry_time=pos.entry_time,
                        exit_time=bar_time if isinstance(bar_time, datetime) else bar_time.to_pydatetime(),
                        pnl=pnl,
                        pnl_pct=pnl / (pos.entry_price * pos.quantity) if pos.entry_price * pos.quantity > 0 else 0,
                        fees=fees,
                    )
                    tracker.record_trade(trade)
                    risk.record_trade(trade)
                    equity += pnl
                    positions.remove(pos)

            if risk.check_circuit_breaker(equity):
                continue
            if risk.check_daily_loss_limit(equity):
                continue

            if composite.should_enter_long(sig) and not any(p.side == Side.LONG and p.symbol == symbol for p in positions):
                qty = risk.compute_position_size(equity, price, atr, positions)
                if qty > 0:
                    entry_price = price * (1 + self.slippage_pct)
                    sl = entry_price - atr * self.sl_atr_mult
                    tp = entry_price + atr * self.tp_atr_mult
                    pos = Position(
                        symbol=symbol,
                        side=Side.LONG,
                        entry_price=entry_price,
                        quantity=qty,
                        entry_time=bar_time if isinstance(bar_time, datetime) else bar_time.to_pydatetime(),
                        stop_loss=sl,
                        take_profit=tp,
                    )
                    positions.append(pos)
                    equity -= entry_price * qty * self.fee_rate

            elif composite.should_enter_short(sig) and not any(p.side == Side.SHORT and p.symbol == symbol for p in positions):
                qty = risk.compute_position_size(equity, price, atr, positions)
                if qty > 0:
                    entry_price = price * (1 - self.slippage_pct)
                    sl = entry_price + atr * self.sl_atr_mult
                    tp = entry_price - atr * self.tp_atr_mult
                    pos = Position(
                        symbol=symbol,
                        side=Side.SHORT,
                        entry_price=entry_price,
                        quantity=qty,
                        entry_time=bar_time if isinstance(bar_time, datetime) else bar_time.to_pydatetime(),
                        stop_loss=sl,
                        take_profit=tp,
                    )
                    positions.append(pos)
                    equity -= entry_price * qty * self.fee_rate

        for pos in positions:
            final_price = float(df.iloc[-1]["close"])
            if pos.side == Side.LONG:
                pnl_raw = (final_price - pos.entry_price) * pos.quantity
            else:
                pnl_raw = (pos.entry_price - final_price) * pos.quantity
            fees = pos.entry_price * pos.quantity * self.fee_rate + final_price * pos.quantity * self.fee_rate
            pnl = pnl_raw - fees
            trade = Trade(
                symbol=symbol,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=final_price,
                quantity=pos.quantity,
                entry_time=pos.entry_time,
                exit_time=df.index[-1] if isinstance(df.index[-1], datetime) else df.index[-1].to_pydatetime(),
                pnl=pnl,
                pnl_pct=pnl / (pos.entry_price * pos.quantity) if pos.entry_price * pos.quantity > 0 else 0,
                fees=fees,
            )
            tracker.record_trade(trade)
            equity += pnl

        tracker.record_equity(equity)
        metrics = tracker.compute_metrics()

        log.info("=" * 60)
        log.info("BACKTEST RESULTS — %s", symbol)
        log.info("=" * 60)
        log.info("Total Return:        %+.2f%%", metrics.total_return_pct)
        log.info("Annual Return:       %+.2f%%", metrics.annual_return_pct)
        log.info("Sharpe Ratio:        %.3f", metrics.sharpe_ratio)
        log.info("Sortino Ratio:       %.3f", metrics.sortino_ratio)
        log.info("Max Drawdown:        %.2f%%", metrics.max_drawdown_pct)
        log.info("Win Rate:            %.1f%%", metrics.win_rate * 100)
        log.info("Profit Factor:       %.2f", metrics.profit_factor)
        log.info("Avg Win:             %+.2f%%", metrics.avg_win_pct)
        log.info("Avg Loss:            %+.2f%%", metrics.avg_loss_pct)
        log.info("Total Trades:        %d", metrics.total_trades)
        log.info("Avg Duration:        %.1f hours", metrics.avg_trade_duration_hours)
        log.info("Final Equity:        %.2f", equity)
        log.info("=" * 60)

        if metrics.monthly_returns:
            log.info("Monthly PnL:")
            for month, pnl in sorted(metrics.monthly_returns.items()):
                log.info("  %s: %+.2f", month, pnl)

        return metrics


# ---------------------------------------------------------------------------
# Live Trading Engine
# ---------------------------------------------------------------------------

class TradingEngine:
    def __init__(
        self,
        client: ExchangeClient,
        symbols: list[str],
        timeframe: str = "1h",
        leverage: int = 3,
        poll_interval: int = 60,
        max_position_pct: float = 5.0,
        max_drawdown_pct: float = 10.0,
        max_daily_loss_pct: float = 5.0,
    ):
        self.client = client
        self.symbols = symbols
        self.timeframe = timeframe
        self.poll_interval = poll_interval

        self.risk = RiskManager(
            max_position_pct=max_position_pct,
            max_drawdown_pct=max_drawdown_pct,
            max_daily_loss_pct=max_daily_loss_pct,
            default_leverage=leverage,
        )
        self.executor = TradeExecutor(client)
        self.composite = CompositeSignal()
        self.tracker = PerformanceTracker()
        self.positions: dict[str, Position] = {}
        self.running = False

    def start(self) -> None:
        self.running = True
        log.info("Trading engine started. Symbols: %s Timeframe: %s", self.symbols, self.timeframe)

        balance = self.client.get_balance()
        initial_equity = balance["total"]
        self.risk.equity_peak = initial_equity
        self.risk.daily_start_equity = initial_equity
        self.tracker.initial_equity = initial_equity
        log.info("Initial equity: %.2f USDT", initial_equity)

        while self.running:
            try:
                self._tick()
            except KeyboardInterrupt:
                self.stop()
                break
            except Exception as e:
                log.error("Tick error: %s", e, exc_info=True)

            time.sleep(self.poll_interval)

    def stop(self) -> None:
        self.running = False
        log.info("Shutting down trading engine")
        self.tracker.save()
        metrics = self.tracker.compute_metrics()
        log.info(
            "Session metrics — Trades: %d Win rate: %.1f%% PnL: %+.2f%%",
            metrics.total_trades, metrics.win_rate * 100, metrics.total_return_pct,
        )

    def _tick(self) -> None:
        balance = self.client.get_balance()
        equity = balance["total"]
        self.tracker.record_equity(equity)
        self.risk.reset_daily_if_needed(equity)
        self.risk.update_equity_peak(equity)

        active_positions = list(self.positions.values())

        for symbol in self.symbols:
            try:
                self._process_symbol(symbol, equity, active_positions)
            except Exception as e:
                log.error("Error processing %s: %s", symbol, e, exc_info=True)

        if len(self.tracker.trades) % 10 == 0 and self.tracker.trades:
            self.tracker.save()

    def _process_symbol(self, symbol: str, equity: float, all_positions: list[Position]) -> None:
        df = self.client.fetch_ohlcv(symbol, self.timeframe, limit=200)
        df = DataEngine.compute_indicators(df)

        if df.empty or len(df) < 50:
            return

        current_price = float(df.iloc[-1]["close"])
        atr = float(df.iloc[-1]["atr"])
        funding_rate = self.client.fetch_funding_rate(symbol)

        if symbol in self.positions:
            pos = self.positions[symbol]
            pos = self.executor.update_trailing_stop(pos, current_price)
            self.positions[symbol] = pos

            if self.executor.check_stop_hit(pos, current_price) or self.executor.check_tp_hit(pos, current_price):
                trade = self.executor.close_position(pos, current_price)
                if trade:
                    self.tracker.record_trade(trade)
                    self.risk.record_trade(trade)
                    self.risk.set_cooldown(symbol)
                    del self.positions[symbol]
                return

            sig = self.composite.generate(df, funding_rate, symbol)
            should_exit = (
                (pos.side == Side.LONG and self.composite.should_exit_long(sig))
                or (pos.side == Side.SHORT and self.composite.should_exit_short(sig))
            )
            if should_exit:
                trade = self.executor.close_position(pos, current_price)
                if trade:
                    self.tracker.record_trade(trade)
                    self.risk.record_trade(trade)
                    self.risk.set_cooldown(symbol)
                    del self.positions[symbol]
            return

        can_trade, reason = self.risk.can_trade(equity, symbol, all_positions)
        if not can_trade:
            log.debug("Cannot trade %s: %s", symbol, reason)
            return

        sig = self.composite.generate(df, funding_rate, symbol)
        leverage = self.risk.effective_leverage(equity)

        if self.composite.should_enter_long(sig):
            qty = self.risk.compute_position_size(equity, current_price, atr, all_positions)
            if qty > 0:
                pos = self.executor.open_position(symbol, Side.LONG, qty, current_price, atr, leverage)
                if pos:
                    self.positions[symbol] = pos

        elif self.composite.should_enter_short(sig):
            qty = self.risk.compute_position_size(equity, current_price, atr, all_positions)
            if qty > 0:
                pos = self.executor.open_position(symbol, Side.SHORT, qty, current_price, atr, leverage)
                if pos:
                    self.positions[symbol] = pos


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quant Futures Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["live", "paper", "backtest"],
        default="paper",
        help="Trading mode (default: paper)",
    )
    parser.add_argument("--symbol", default=None, help="Override symbol for backtest")
    parser.add_argument("--timeframe", default=None, help="Override timeframe")
    parser.add_argument("--start", default=None, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--equity", type=float, default=10000.0, help="Backtest initial equity")
    parser.add_argument("--output", default=None, help="Backtest output JSON file")
    return parser.parse_args()


def build_client(testnet: bool) -> ExchangeClient:
    return ExchangeClient(
        exchange_id=os.getenv("EXCHANGE", "binance"),
        api_key=os.getenv("API_KEY", ""),
        api_secret=os.getenv("API_SECRET", ""),
        passphrase=os.getenv("API_PASSPHRASE", ""),
        testnet=testnet,
    )


def run_backtest(args: argparse.Namespace) -> None:
    symbol = args.symbol or os.getenv("SYMBOLS", "BTC/USDT").split(",")[0].strip()
    timeframe = args.timeframe or os.getenv("TIMEFRAME", "1h")
    leverage = int(os.getenv("LEVERAGE", "3"))
    max_pos_pct = float(os.getenv("MAX_POSITION_PCT", "5.0"))

    if not args.start or not args.end:
        log.error("Backtest requires --start and --end dates (YYYY-MM-DD)")
        sys.exit(1)

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    log.info("Fetching historical data for %s %s from %s to %s", symbol, timeframe, args.start, args.end)

    client = build_client(testnet=True)
    df = client.fetch_ohlcv_since(symbol, timeframe, start_ms, end_ms)

    if df.empty:
        log.error("No data fetched for backtest")
        sys.exit(1)

    log.info("Fetched %d candles", len(df))

    engine = BacktestEngine(
        initial_equity=args.equity,
        leverage=leverage,
        max_position_pct=max_pos_pct,
    )
    metrics = engine.run(df, symbol)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(asdict(metrics), indent=2, default=str))
        log.info("Results saved to %s", output_path)


def run_live(testnet: bool) -> None:
    symbols = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT").split(",")]
    timeframe = os.getenv("TIMEFRAME", "1h")
    leverage = int(os.getenv("LEVERAGE", "3"))
    poll_interval = int(os.getenv("POLL_INTERVAL", "60"))
    max_pos_pct = float(os.getenv("MAX_POSITION_PCT", "5.0"))
    max_dd_pct = float(os.getenv("MAX_DRAWDOWN_PCT", "10.0"))
    max_daily_pct = float(os.getenv("MAX_DAILY_LOSS_PCT", "5.0"))

    client = build_client(testnet=testnet)
    engine = TradingEngine(
        client=client,
        symbols=symbols,
        timeframe=timeframe,
        leverage=leverage,
        poll_interval=poll_interval,
        max_position_pct=max_pos_pct,
        max_drawdown_pct=max_dd_pct,
        max_daily_loss_pct=max_daily_pct,
    )

    def handle_signal(signum, frame):
        log.info("Received signal %d, stopping...", signum)
        engine.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    mode_label = "PAPER (testnet)" if testnet else "LIVE (mainnet)"
    log.info("Starting %s trading: %s on %s", mode_label, symbols, timeframe)
    engine.start()


def main() -> None:
    args = parse_args()

    if args.mode == "backtest":
        run_backtest(args)
    elif args.mode == "paper":
        run_live(testnet=True)
    elif args.mode == "live":
        log.warning("=" * 60)
        log.warning("  LIVE TRADING MODE — REAL MONEY AT RISK")
        log.warning("=" * 60)
        run_live(testnet=False)


if __name__ == "__main__":
    main()
